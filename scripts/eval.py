"""Evaluate SQL generation models on execution accuracy.

This script is the final evaluation step in the pipeline. It generates
SQL with greedy decoding (deterministic) and measures execution accuracy
by running both candidate and gold SQL against the actual database.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from sql_agent.dataset_utils.prompts import build_chatml_prompt
from sql_agent.env.sql_sandbox import SqlErrorType, SqlSandbox
from sql_agent.models.tokenizer import load_tokenizer
from sql_agent.generation import extract_sql_from_generation
from sql_agent.reward.execution_reward import (
    RewardConfig,
    compute_execution_reward,
)
from sql_agent.utils import (
    build_quantization_config,
    load_config,
    resolve_dtype,
    save_json,
    setup_logging,
    timestamp,
    write_jsonl_line,
)


def _classify_difficulty(gold_sql: str) -> str:
    """Classifies SQL query difficulty based on structural complexity.

    Categories follow benchmark conventions:
    - easy: single table, no JOIN, no subquery.
    - medium: one JOIN or GROUP BY or ORDER BY.
    - hard: multiple JOINs or subquery or HAVING.
    - extra: nested subqueries of INTERSECT/UNION/EXCEPT.

    Args:
        gold_sql: Gold SQL query text.

    Returns:
         One of ``easy``, ``medium``, ``hard``, ``extra``.
    """
    sql_lower = gold_sql.lower()

    # Count structural elements.
    num_joins = len(re.findall(r"\bjoin\b", sql_lower))
    has_subquery = (
        "select" in sql_lower[sql_lower.find("from") + 4:]
        if "from" in sql_lower
        else False
    )
    has_group_by = "group by" in sql_lower
    has_order_by = "order by" in sql_lower
    has_having = "having" in sql_lower
    has_set_op = any(
        op in sql_lower
        for op in ("intersect", "union", "except")
    )
    has_nested = sql_lower.count("select") >= 3

    if has_nested or has_set_op:
        return "extra"
    if num_joins >= 2 or has_subquery or has_having:
        return "hard"
    if num_joins >= 1 or has_group_by or has_order_by:
        return "medium"
    return "easy"


def _load_model(
    model_name: str,
    dtype: torch.dtype,
    adapter_dir: Optional[Path] = None,
) -> tuple[Any, Any]:
    """Loads quantized base model, optionally with a LoRA adapter.

    Args:
        model_name: HuggingFace model name.
        dtype: Compute dtype.
        adapter_dir: Path to saved adapter, or ``None`` for base model.

    Returns:
        ``(model, device)`` tuple.
    """
    bnb_config = build_quantization_config(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(
            base_model, adapter_dir.as_posix(),
        )
    else:
        model = base_model

    model.eval()

    device = next(model.parameters()).device
    return model, device


def _configure_generation_tokens(
    model: Any,
    tokenizer: Any,
) -> Optional[list[int]]:
    """Configures EOS and PAD tokens for generation.

    Args:
        model: Loaded causal language model.
        tokenizer: Model tokenizer.

    Returns:
        List of EOS token IDs, or ``None`` if no special tokens found.
    """
    eos_token_ids: list[int] = []

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    if (
        isinstance(im_end_id, int)
        and im_end_id >= 0
        and im_end_id != unk_token_id
    ):
        eos_token_ids.append(im_end_id)

    if (
        tokenizer.eos_token_id is not None
        and tokenizer.eos_token_id not in eos_token_ids
    ):
        eos_token_ids.append(int(tokenizer.eos_token_id))

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if eos_token_ids:
        model.config.eos_token_id = eos_token_ids
        model.generation_config.eos_token_id = eos_token_ids

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return eos_token_ids or None


def _iter_examples(
    data_file: str,
    *,
    max_examples: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Yields dataset examples from a JSON file.

    Args:
        data_file: Path to a processed JSON file.
        max_examples: Optional cap for debugging.

    Yields:
        Dataset row dictionaries.
    """
    ds = load_dataset("json", data_files=data_file, split="train")
    n = len(ds) if max_examples is None else min(len(ds), max_examples)
    for i in range(n):
        yield dict(ds[i])


def _generate_sql_batch(
    model: Any,
    tokenizer: Any,
    device: Any,
    eos_token_ids: Optional[list[int]],
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
) -> list[str]:
    """Greedy-generates SQL for a batch of prompts.

    Requires ``tokenizer.padding_side == "left"`` so generated tokens
    start at the same position for every row.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer matching the model.
        device: Device to run generation on.
        eos_token_ids: List of EOS token IDs.
        prompts: Fully formatted prompts, one per example.
        max_length: Max prompt length.
        max_new_tokens: Max tokens to generate.
        repetition_penalty: Penalty for repeated tokens, 1.0 means no penalty.

    Returns:
        Extracted SQL string, one per prompt, in the same order.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.inference_mode():
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if eos_token_ids is not None:
            generate_kwargs["eos_token_id"] = eos_token_ids

        outputs = model.generate(**generate_kwargs)

    gen_ids = outputs[:, input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
    return [extract_sql_from_generation(text) for text in decoded]


def _score_example(
    sandbox: SqlSandbox,
    reward_config: RewardConfig,
    example: dict[str, Any],
    predicted_sql: str,
) -> dict[str, Any]:
    """Executes predicted SQL against gold and builds the result row.

    Args:
        sandbox: SQL execution sandbox.
        reward_config: Settings for reward computation.
        example: Dataset example with "question", "sql" and "db_id" keys.
        predicted_sql: SQL generation by the model.

    Returns:
        Result row with match status, reward, error type and difficulty.
    """
    question = example.get("question", "")
    gold_sql = example.get("sql", "")
    db_id = example.get("db_id", "")
    difficulty = _classify_difficulty(gold_sql)

    try:
        reward_res = compute_execution_reward(
            sandbox=sandbox,
            db_id=db_id,
            candidate_sql=predicted_sql,
            gold_sql=gold_sql,
            config=reward_config,
            strict_gold=False,
        )
        return {
            "question": question,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "matched": reward_res.matched,
            "reward": reward_res.reward,
            "reason": reward_res.reason,
            "error_type": reward_res.candidate.error_type.value,
            "difficulty": difficulty,
            "gold_failed": reward_res.reason == "gold_failed",
        }
    except Exception as exc:
        logging.warning("Evaluation failed for db_id=%s: %s", db_id, exc)
        return {
            "question": question,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "matched": False,
            "reward": 0.0,
            "reason": "eval_error",
            "error_type": "unknown_error",
            "difficulty": difficulty,
            "gold_failed": False,
        }


def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes comprehensive evaluation metrics.

    Args:
        results: List of per-example result dictionaries.

    Returns:
        Metrics dictionary with overall and per-difficulty breakdown.
    """
    total = len(results)
    if total == 0:
        return {"execution_accuracy": 0.0, "total": 0}

    # Overall execution accuracy.
    matched = sum(1 for r in results if r["matched"])
    exec_accuracy = matched / total

    # Exclude examples where gold SQL itself failed.
    valid_results = [r for r in results if not r.get("gold_failed", False)]
    valid_total = len(valid_results)
    valid_matched = sum(1 for r in valid_results if r["matched"])
    valid_accuracy = valid_matched / valid_total if valid_total > 0 else 0.0

    # Per-difficulty breakdown.
    difficulty_metrics: dict[str, dict[str, Any]] = {}
    for diff in ("easy", "medium", "hard", "extra"):
        diff_results = [r for r in valid_results if r["difficulty"] == diff]
        diff_total = len(diff_results)
        diff_matched = sum(1 for r in diff_results if r["matched"])
        difficulty_metrics[diff] = {
            "total": diff_total,
            "matched": diff_matched,
            "accuracy": diff_matched / diff_total if diff_total > 0 else 0.0,
        }

    # Error breakdown.
    error_counts: dict[str, int] = {}
    for r in results:
        reason = r.get("reason", "unknown")
        error_counts[reason] = error_counts.get(reason, 0) + 1

    return {
        "execution_accuracy": exec_accuracy,
        "execution_accuracy_valid": valid_accuracy,
        "total": total,
        "matched": matched,
        "gold_failed": total - valid_total,
        "by_difficulty": difficulty_metrics,
        "by_reason": error_counts,
    }


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SQL generation model on execution accuracy."
    )
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--reward_config", default="configs/reward.yaml")

    parser.add_argument(
        "--adapter_dir",
        nargs="+",
        default=[],
        help="One or more adapter directories to evaluate.",
    )
    parser.add_argument(
        "--include_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the raw pretrained model (no adapter) for comparison. "
             "Use --no-include_base to skip.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/eval",
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty at eval decoding (1.0 = off). "
             "Old runs used a hardcoded 1.2.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Prompts per generation batch.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Environment controls.
    parser.add_argument("--database_root", default=None)
    parser.add_argument("--timeout_s", type=float, default=2.0)
    parser.add_argument("--max_rows", type=int, default=200)

    return parser.parse_args()


def _evaluate_adapter(
    adapter_dir: Optional[Path],
    model_cfg: dict[str, Any],
    data_file: str,
    sandbox: SqlSandbox,
    reward_config: RewardConfig,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluates one adapter (or base model) and saves results.

    Args:
        adapter_dir: Path to the adapter checkpoint, or ``None`` to
            evaluate the raw pretrained model without any adapter.
        model_cfg: Model configuration.
        data_file: Path to evaluation data.
        sandbox: SQL execution sandbox.
        reward_config: Reward configuration.
        run_dir: Output directory for this evaluation.
        args: CLI arguments.

    Returns:
        Summary metrics dictionary.
    """
    if adapter_dir is not None:
        label = adapter_dir.as_posix()
        safe_name = label.replace("/", "_").replace("\\", "_")
    else:
        label = f"base/{model_cfg['model_name']}"
        safe_name = "base_pretrained"

    logging.info("=" * 60)
    logging.info("Evaluating: %s", label)
    logging.info("=" * 60)

    # Load model.
    model_name = model_cfg["model_name"]
    dtype = resolve_dtype(model_cfg.get("dtype", "float16"))

    model, device = _load_model(model_name, dtype, adapter_dir)
    tokenizer = load_tokenizer(model_name)
    tokenizer.padding_side = "left"
    eos_token_ids = _configure_generation_tokens(model, tokenizer)
    max_length = int(model_cfg.get("max_length", 2048))

    logging.info("Model loaded. Device: %s", device)

    results: list[dict[str, Any]] = []
    start_time = time.time()
    results_path = run_dir / f"results_{safe_name}.jsonl"
    skipped_incomplete = 0

    def _flush_batch(batch: list[dict[str, Any]], out_fp: Any) -> None:
        """Generates and scores SQL for the batch."""
        prompts = [
            build_chatml_prompt(schema=ex["schema"], question=ex["question"])
            for ex in batch
        ]
        predicted = _generate_sql_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            eos_token_ids=eos_token_ids,
            prompts=prompts,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        for ex, sql in zip(batch, predicted):
            result = _score_example(sandbox, reward_config, ex, sql)
            result["example_idx"] = ex["_idx"]
            results.append(result)
            write_jsonl_line(out_fp, result)

    with results_path.open("w", encoding="utf-8") as f:
        batch: list[dict[str, Any]] = []
        for idx, example in enumerate(
            _iter_examples(data_file, max_examples=args.max_examples)
        ):
            if not all(
                example.get(k) for k in ("question", "schema", "sql", "db_id")
            ):
                skipped_incomplete += 1
                continue

            example["_idx"] = idx
            batch.append(example)

            if len(batch) >= args.eval_batch_size:
                _flush_batch(batch, f)
                batch = []
                # Free unused GPU memory.
                torch.cuda.empty_cache()

                current_acc = sum(
                    1 for r in results if r["matched"]
                ) / len(results)
                elapsed = time.time() - start_time
                logging.info(
                    "Progress: %d examples | accuracy=%.3f | speed=%.2f ex/s",
                    len(results),
                    current_acc,
                    len(results) / elapsed
                )

        if batch:
            _flush_batch(batch, f)

    elapsed = time.time() - start_time

    if skipped_incomplete:
        logging.warning(
            "Skipped %d examples with missing fields.", skipped_incomplete
        )

    # Compute metrics.
    metrics = _compute_metrics(results)
    metrics["adapter_dir"] = adapter_dir.as_posix() if adapter_dir else None
    metrics["label"] = label
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["examples_per_second"] = (
        round(len(results) / elapsed, 2) if elapsed > 0 else 0.0
    )

    # Log summary.
    logging.info("-" * 60)
    logging.info("Results for: %s", label)
    logging.info(
        "Execution accuracy: %.1f%% (%d/%d)",
        metrics["execution_accuracy"] * 100,
        metrics["matched"],
        metrics["total"],
    )
    logging.info(
        "Accuracy (excl. gold failures): %.1f%% ",
        metrics["execution_accuracy_valid"] * 100,
    )

    for diff in ("easy", "medium", "hard", "extra"):
        d = metrics["by_difficulty"][diff]
        if d["total"] > 0:
            logging.info(
                "  %-8s: %.1f%% (%d/%d)",
                diff,
                d["accuracy"] * 100,
                d["matched"],
                d["total"],
            )

    logging.info("Error breakdown: %s", metrics["by_reason"])
    logging.info(
        "Time: %.1fs (%.1f examples/s)",
        elapsed,
        metrics["examples_per_second"],
    )
    logging.info("-" * 60)

    # Free GPU memory before next adapter.
    del model
    torch.cuda.empty_cache()

    return metrics


def _fmt_delta(value: float, base_value: float) -> str:
    """Formats a metric value with delta vs base."""
    delta = value - base_value
    sign = "+" if delta >= 0 else ""
    return f"{value * 100:5.1f}% ({sign}{delta * 100:.1f})"


def _log_comparison_table(
    all_metrics: list[dict[str, Any]],
    base_metrics: dict[str, Any] | None,
) -> None:
    """Logs a detailed comparison table across all evaluated models.

    Shows overall accuracy, per-difficulty breakdown, and deltas
    relative to the base model when available.

    Args:
        all_metrics: List of metrics dicts from each evaluated model.
        base_metrics: Base model metrics, or ``None`` if not evaluated.
    """
    logging.info("=" * 72)
    logging.info("COMPARISON SUMMARY")
    logging.info("=" * 72)

    # Overall accuracy table.
    logging.info("")
    logging.info("%-42s %12s %12s", "Model", "Accuracy", "Valid Acc")
    logging.info("-" * 72)

    for m in sorted(
        all_metrics,
        key=lambda x: x["execution_accuracy"],
        reverse=True,
    ):
        label = m["label"]
        if len(label) > 40:
            label = "..." + label[-37:]

        if base_metrics is not None and m is not base_metrics:
            acc_str = _fmt_delta(
                m["execution_accuracy"],
                base_metrics["execution_accuracy"],
            )
            vacc_str = _fmt_delta(
                m["execution_accuracy_valid"],
                base_metrics["execution_accuracy_valid"],
            )
        else:
            acc_str = f"{m['execution_accuracy'] * 100:5.1f}%"
            vacc_str = f"{m['execution_accuracy_valid'] * 100:5.1f}%"

        logging.info("  %-40s %12s %12s", label, acc_str, vacc_str)

    # Per-difficulty comparison.
    if base_metrics is not None:
        finetuned = [m for m in all_metrics if m is not base_metrics]
        if finetuned:
            logging.info("")
            logging.info(
                "  %-40s %8s %8s %8s %8s",
                "Model", "easy", "medium", "hard", "extra",
            )
            logging.info("-" * 72)

            for m in finetuned:
                label = m["label"]
                if len(label) > 40:
                    label = "..." + label[-37:]

                diffs = []
                for diff in ("easy", "medium", "hard", "extra"):
                    m_acc = m["by_difficulty"][diff]["accuracy"]
                    b_acc = base_metrics["by_difficulty"][diff]["accuracy"]
                    delta = (m_acc - b_acc) * 100
                    sign = "+" if delta >= 0 else ""
                    diffs.append(f"{sign}{delta:.1f}%")

                logging.info(
                    "  %-40s %8s %8s %8s %8s",
                    label, *diffs,
                )

    logging.info("=" * 72)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    model_cfg = load_config(args.model_config)
    data_cfg = load_config(args.data_config)
    reward_cfg_raw = load_config(args.reward_config)

    run_dir = Path(args.output_dir) / f"eval_{timestamp()}"
    setup_logging(run_dir, "eval.log")

    logging.info("Evaluation run directory: %s", run_dir)

    # Resolve data file.
    data_file = (
        data_cfg["val_file"]
        if args.split == "val"
        else data_cfg["train_file"]
    )
    logging.info("Split: %s (%s)", args.split, data_file)

    # Resolve database root.
    database_root = args.database_root or data_cfg.get("database_file")
    if database_root is None:
        raise ValueError("Database root not found.")

    reward_config = RewardConfig.from_dict(reward_cfg_raw)

    sandbox = SqlSandbox(
        database_root=database_root,
        timeout_s=args.timeout_s,
        max_rows=args.max_rows,
    )

    # Evaluate base model first (default) unless --no-include-base.
    all_metrics: list[dict[str, Any]] = []

    if not args.adapter_dir and not args.include_base:
        raise ValueError(
            "Provide --adapter_dir and/or --include_base."
        )

    base_metrics: dict[str, Any] | None = None
    if args.include_base:
        logging.info("Evaluating base model (no adapter)...")
        base_metrics = _evaluate_adapter(
            adapter_dir=None,
            model_cfg=model_cfg,
            data_file=data_file,
            sandbox=sandbox,
            reward_config=reward_config,
            run_dir=run_dir,
            args=args,
        )
        all_metrics.append(base_metrics)

    # Evaluate each adapter.
    for adapter_path in args.adapter_dir:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            logging.error("Adapter not found: %s", adapter_dir)
            continue

        metrics = _evaluate_adapter(
            adapter_dir=adapter_dir,
            model_cfg=model_cfg,
            data_file=data_file,
            sandbox=sandbox,
            reward_config=reward_config,
            run_dir=run_dir,
            args=args,
        )
        all_metrics.append(metrics)

    # Save comparison summary.
    if len(all_metrics) > 1:
        _log_comparison_table(all_metrics, base_metrics)

    # Compute deltas vs base for JSON summary.
    comparison: list[dict[str, Any]] = []
    if base_metrics is not None:
        for m in all_metrics:
            if m is base_metrics:
                continue
            delta_overall = (
                m["execution_accuracy"] - base_metrics["execution_accuracy"]
            )
            delta_valid = (
                m["execution_accuracy_valid"]
                - base_metrics["execution_accuracy_valid"]
            )
            delta_by_diff = {}
            for diff in ("easy", "medium", "hard", "extra"):
                delta_by_diff[diff] = round(
                    m["by_difficulty"][diff]["accuracy"]
                    - base_metrics["by_difficulty"][diff]["accuracy"],
                    4,
                )
            comparison.append({
                "label": m["label"],
                "delta_accuracy": round(delta_overall, 4),
                "delta_accuracy_valid": round(delta_valid, 4),
                "delta_by_difficulty": delta_by_diff,
            })

    summary = {
        "timestamp": run_dir.name,
        "split": args.split,
        "data_file": data_file,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "results": all_metrics,
        "comparison_vs_base": comparison if comparison else None,
        "eval_batch_size": args.eval_batch_size,
        "repetition_penalty": args.repetition_penalty,
    }
    save_json(run_dir / "eval_summary.json", summary)

    logging.info("Evaluation saved to: %s", run_dir)


if __name__ == "__main__":
    main()
