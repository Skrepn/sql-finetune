"""Execution-based reward for SQL agent training.

This module scores a candidate SQL query by executing it in a read-only
SQLite sandbox and comparing result sets against a gold reference query.

The reward is designed for:
- Verifier-guided data collection (rollout -> preferences).
- Offline preference optimization.

It returns a structured ``RewardResult`` with reward value and diagnostics.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Optional, Sequence

from sql_agent.env.sql_sandbox import (
    ExecutionResult,
    SqlErrorType,
    SqlSandbox,
    normalize_result_rows,
)

__all__ = [
    "RewardConfig",
    "RewardResult",
    "compare_result_sets",
    "compute_execution_reward",
]


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Reward configuration for execution-based scoring.

    Attributes:
        match: Reward when candidate result set matches gold result set.
        exec_mismatch: Reward when candidate executes but result set differs.
        syntax_error: Reward (penalty) for syntax errors.
        runtime_error: Reward (penalty) for runtime errors.
        forbidden: Reward (penalty) for forbidden queries.
        timeout: Reward (penalty) for timeouts.
        unknown_error: Reward (penalty) for unknown errors.
        length_penalty_weight: Per-token penalty applied to candidate SQL.
        select_star_penalty: Penalty if query uses ``SELECT *``.
        slow_query_ms: If > 0, queries slower than this are penalized.
        slow_query_penalty: Penalty applied when ``slow_query_ms`` is exceeded.
    """

    match: float = 1.0
    exec_mismatch: float = 0.1
    syntax_error: float = -0.2
    runtime_error: float = -0.4
    forbidden: float = -1.0
    timeout: float = -0.6
    unknown_error: float = -0.6

    length_penalty_weight: float = 0.0
    select_star_penalty: float = 0.0
    slow_query_ms: int = 0
    slow_query_penalty: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RewardConfig:
        """Builds a ``RewardConfig`` from a mapping, ignoring unknown keys.

        Args:
            data: Mapping with keys matching ``RewardConfig`` fields.

        Returns:
            New ``RewardConfig`` instance.
        """
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)


@dataclasses.dataclass(frozen=True)
class RewardResult:
    """Structured output of execution-based reward computation.

    Attributes:
        reward: Final scalar reward.
        matched: True if candidate and gold result sets match.
        candidate: ``ExecutionResult`` for candidate SQL.
        gold: ``ExecutionResult`` for gold SQL (``None`` on early failure).
        reason: Short string describing the main outcome.
    """

    reward: float
    matched: bool
    candidate: ExecutionResult
    gold: Optional[ExecutionResult]
    reason: str


_SELECT_STAR_RE = re.compile(r"\bselect\s+\*\b", re.IGNORECASE)


def compute_execution_reward(
    sandbox: SqlSandbox,
    db_id: str,
    candidate_sql: str,
    gold_sql: str,
    config: RewardConfig,
    *,
    strict_gold: bool = True,
) -> RewardResult:
    """Computes reward for a candidate SQL by execution against gold SQL.

    Args:
        sandbox: ``SqlSandbox`` used for safe execution.
        db_id: Database identifier used to locate the SQLite database.
        candidate_sql: Candidate SQL string to score.
        gold_sql: Reference SQL string for the same prompt.
        config: ``RewardConfig`` specifying outcome rewards and shaping
            penalties.
        strict_gold: If ``True``, raises ``ValueError`` when the gold
            query fails to run. If ``False``, returns a ``RewardResult``
            with reason ``"gold_failed"``.

    Returns:
        ``RewardResult`` containing reward scalar and diagnostics.

    Raises:
        ValueError: If ``strict_gold`` is ``True`` and the gold query fails.
    """
    gold_res = sandbox.execute(db_id=db_id, sql=gold_sql)
    if not gold_res.ok:
        if strict_gold:
            raise ValueError(
                f"Gold SQL failed for db_id={db_id}: "
                f"{gold_res.error_type.value}: {gold_res.error_message}"
            )
        return RewardResult(
            reward=float(config.unknown_error),
            matched=False,
            candidate=sandbox.execute(db_id=db_id, sql=candidate_sql),
            gold=gold_res,
            reason="gold_failed",
        )

    cand_res = sandbox.execute(db_id=db_id, sql=candidate_sql)
    base_reward, matched, reason = _base_reward_from_results(
        cand_res=cand_res,
        gold_res=gold_res,
        config=config,
    )

    shaped_reward = base_reward + _shaping_penalties(
        candidate_sql=candidate_sql,
        candidate_exec=cand_res,
        config=config,
    )

    return RewardResult(
        reward=float(shaped_reward),
        matched=matched,
        candidate=cand_res,
        gold=gold_res,
        reason=reason,
    )


def compare_result_sets(
    gold_rows: Sequence[Sequence[Any]],
    candidate_rows: Sequence[Sequence[Any]],
) -> bool:
    """Compares result sets in a stable, order-insensitive way.

    Args:
        gold_rows: Rows from executing gold SQL.
        candidate_rows: Rows from executing candidate SQL.

    Returns:
        ``True`` if normalized result sets match.
    """
    return normalize_result_rows(gold_rows) == normalize_result_rows(
        candidate_rows
    )


def _base_reward_from_results(
    cand_res: ExecutionResult,
    gold_res: ExecutionResult,
    config: RewardConfig,
) -> tuple[float, bool, str]:
    """Computes the base reward from execution outcomes.

    Args:
        cand_res: Candidate execution result.
        gold_res: Gold execution result.
        config: Reward configuration.

    Returns:
        A ``(reward, matched, reason)`` tuple.
    """
    if cand_res.ok:
        matched = compare_result_sets(gold_res.rows, cand_res.rows)
        if matched:
            return float(config.match), True, "match"
        return float(config.exec_mismatch), False, "exec_mismatch"

    error_map = {
        SqlErrorType.FORBIDDEN: (config.forbidden, "forbidden"),
        SqlErrorType.TIMEOUT: (config.timeout, "timeout"),
        SqlErrorType.SYNTAX_ERROR: (config.syntax_error, "syntax_error"),
        SqlErrorType.RUNTIME_ERROR: (config.runtime_error, "runtime_error"),
    }

    reward_value, reason = error_map.get(
        cand_res.error_type,
        (config.unknown_error, "unknown_error"),
    )
    return float(reward_value), False, reason


def _shaping_penalties(
    candidate_sql: str,
    candidate_exec: ExecutionResult,
    config: RewardConfig,
) -> float:
    """Computes optional shaping penalties.

    Args:
        candidate_sql: Candidate SQL string.
        candidate_exec: Candidate execution result.
        config: Reward configuration.

    Returns:
        A (possibly negative) shaping term to add to the base reward.
    """
    penalty = 0.0

    if config.length_penalty_weight != 0.0:
        penalty -= config.length_penalty_weight * _count_sql_tokens(
            candidate_sql
        )

    if config.select_star_penalty != 0.0 and _SELECT_STAR_RE.search(
        candidate_sql
    ):
        penalty -= config.select_star_penalty

    if (
        config.slow_query_ms > 0
        and config.slow_query_penalty != 0.0
        and candidate_exec.ok
        and candidate_exec.elapsed_ms > config.slow_query_ms
    ):
        penalty -= config.slow_query_penalty

    return float(penalty)


def _count_sql_tokens(sql: str) -> int:
    """Counts approximate whitespace-delimited tokens in SQL.

    Args:
        sql: SQL query text.

    Returns:
        Integer token count approximation.
    """
    return len(re.findall(r"\S+", sql.strip()))
