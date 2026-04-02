"""Tests for execution-based reward computation."""

from __future__ import annotations

from sql_agent.env.sql_sandbox import ExecutionResult, SqlErrorType
from sql_agent.reward.execution_reward import (
    RewardConfig,
    _base_reward_from_results,
    _shaping_penalties,
    compare_result_sets,
    compute_execution_reward,
)


def test_compare_result_sets_is_order_insensitive() -> None:
    """compare_result_sets ignores row order."""
    gold = [(1, "a"), (2, "b")]
    cand = [(2, "b"), (1, "a")]
    assert compare_result_sets(gold, cand)


def test_reward_config_from_dict_ignores_unknown_keys() -> None:
    """RewardConfig.from_dict ignores unexpected keys."""
    config = RewardConfig.from_dict({"match": 2.0, "unknown": 123})
    assert config.match == 2.0
    assert not hasattr(config, "unknown")


def test_base_reward_for_success_match() -> None:
    """_base_reward_from_results returns match reward for identical rows."""
    # Build two OK results with identical rows.
    config = RewardConfig()
    gold_res = ExecutionResult(
        ok=True,
        error_type=SqlErrorType.OK,
        error_message="",
        rows=((1,),),
        truncated=False,
        elapsed_ms=1,
    )
    cand_res = ExecutionResult(
        ok=True,
        error_type=SqlErrorType.OK,
        error_message="",
        rows=((1,),),
        truncated=False,
        elapsed_ms=1,
    )
    reward, matched, reason = _base_reward_from_results(
        cand_res=cand_res,
        gold_res=gold_res,
        config=config,
    )
    assert reward == config.match
    assert matched is True
    assert reason == "match"


def test_base_reward_for_syntax_error() -> None:
    """_base_reward_from_results maps syntax errors to the correct reward."""
    config = RewardConfig(syntax_error=-0.3)
    gold_res = ExecutionResult(
        ok=True,
        error_type=SqlErrorType.OK,
        error_message="",
        rows=((1,),),
        truncated=False,
        elapsed_ms=1,
    )
    cand_res = ExecutionResult(
        ok=False,
        error_type=SqlErrorType.SYNTAX_ERROR,
        error_message="syntax error",
        rows=(),
        truncated=False,
        elapsed_ms=1,
    )
    reward, matched, reason = _base_reward_from_results(
        cand_res=cand_res,
        gold_res=gold_res,
        config=config,
    )
    assert reward == -0.3
    assert matched is False
    assert reason == "syntax_error"


def test_shaping_penalties_apply() -> None:
    """_shaping_penalties applies length, select-star, and slow penalties."""
    config = RewardConfig(
        length_penalty_weight=0.1,
        select_star_penalty=0.5,
        slow_query_ms=10,
        slow_query_penalty=0.7,
    )
    exec_res = ExecutionResult(
        ok=True,
        error_type=SqlErrorType.OK,
        error_message="",
        rows=(),
        truncated=False,
        elapsed_ms=20,
    )
    penalty = _shaping_penalties(
        candidate_sql="SELECT * FROM users",
        candidate_exec=exec_res,
        config=config,
    )
    assert penalty < 0
    assert abs(penalty) >= 1.0


def test_compute_execution_reward_match(
    sandbox,
) -> None:
    """compute_execution_reward returns match reward for correct SQL."""
    config = RewardConfig(match=1.5)
    res = compute_execution_reward(
        sandbox=sandbox,
        db_id="test_db",
        candidate_sql="SELECT COUNT(*) FROM users",
        gold_sql="SELECT COUNT(*) FROM users",
        config=config,
    )
    assert res.matched is True
    assert res.reward == 1.5
    assert res.reason == "match"


def test_compute_execution_reward_mismatch(
    sandbox,
) -> None:
    """compute_execution_reward returns mismatch reward for wrong SQL."""
    config = RewardConfig(exec_mismatch=0.25)
    res = compute_execution_reward(
        sandbox=sandbox,
        db_id="test_db",
        candidate_sql="SELECT COUNT(*) FROM orders WHERE amount > 100",
        gold_sql="SELECT COUNT(*) FROM users",
        config=config,
    )
    assert res.matched is False
    assert res.reward == 0.25
    assert res.reason == "exec_mismatch"


def test_compute_execution_reward_gold_failure_non_strict(
    sandbox,
) -> None:
    """compute_execution_reward can return gold_failed when strict off."""
    config = RewardConfig(unknown_error=-0.9)
    res = compute_execution_reward(
        sandbox=sandbox,
        db_id="test_db",
        candidate_sql="SELECT COUNT(*) FROM users",
        gold_sql="SELECT FROM missing",
        config=config,
        strict_gold=False,
    )
    assert res.matched is False
    assert res.reward == -0.9
    assert res.reason == "gold_failed"
