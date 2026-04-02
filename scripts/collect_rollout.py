"""Collect rollouts (multiple sampled SQL candidates) for DPO training."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from sql_agent.dataset_utils.prompts import build_chatml_prompt
from sql_agent.env.sql_sandbox import SqlErrorType, SqlSandbox
from sql_agent.models.tokenizer import load_tokenizer
from sql_agent.reward.execution_reward import RewardConfig, compute_execution_reward
from sql_agent.utils import (
    build_quantization_config,
    load_config,
    resolve_dtype,
    resolve_sft_adapter_dir,
    save_run_meta,
    setup_logging,
    timestamp,
    write_jsonl_line,
)


def _strip_code_fences(text: str) -> str:
    """Removes markdown code fences if present.

    Args:
        text: Decoded model output.

    Returns:
        Cleaned text without surrounding ````` fences.
    """
    t = text.strip()
    if "```" not in t:
        return t
    t = t.replace("```sql", "```")
    parts = t.split("```")
    blocks = [p.strip() for p in parts if p.strip()]
    if not blocks:
        return text.strip()
    return max(blocks, key=len)


_NON_SQL_RE = re.compile(r"[^\x00-\x7F]")

def _extract_sql_from_generated(
    generated_text: str,
    *,
    im_end_token: str = "<|im_end|>",
) -> str:
    """Extracts SQL from raw model-generated text.

    Applies multiple cleaning stages to isolate valid SQL:
    1. Strip code fences.
    2. Cut at ChatML end token.
    3. Cut at first non-ASCII character.
    4. Cut at corruption markers (special tokens, repeated newlines).
    5. Cut at semicolon (take only first statement).
    6. Validate that result looks like SQL.

    Args:
        generated_text: Raw decoded text from the model.
        im_end_token: End-of-turn token to split on.

    Returns:
        Cleaned SQL string, or an empty string if no valid SQL found.
    """
    text = _strip_code_fences(generated_text)

    if im_end_token in text:
        text = text.split(im_end_token, 1)[0]

    text = text.strip()

    # Strip assistant prefix if model echoed it.
    for marker in ("<|im_start|>assistant\n", "assistant\n"):
        if text.startswith(marker):
            text = text[len(marker):].strip()

    non_ascii_match = _NON_SQL_RE.search(text)
    if non_ascii_match:
        text = text[:non_ascii_match.start()].strip()

    # Cut at corruption markers.
    corruption_markers = ["<|", "<quote", "```", "\n\n", "\nSELECT", "\nWITH"]
    cut_positions = [
        text.find(m) for m in corruption_markers if m in text
    ]
    if cut_positions:
        text = text[:min(cut_positions)].strip()

    # Take only the first SQL statement.
    if ";" in text:
        text = text.split(";", 1)[0].strip()

    # Validate: must start with SELECT or WITH.
    lowered = text.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return ""

    return text.strip()


def _load_policy_model(
    model_name: str,
    dtype: torch.dtype,
    sft_adapter_dir: Path,
) -> PeftModel:
    """Loads the quantized base model + SFT LoRA adapter for generation.

    The model is placed on GPU automatically by bitsandbytes during
    ``from_pretrained`` — no explicit ``.to(device)`` call is needed.

    Args:
        model_name: HuggingFace model identifier.
        dtype: Torch dtype for compute.
        sft_adapter_dir: Directory containing PEFT adapter weights/config.

    Returns:
        Loaded PEFT model in eval mode.
    """
    bnb_config = build_quantization_config(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, sft_adapter_dir.as_posix())
    model.eval()
    return model


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


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Collect rollouts for SQL DPO training."
    )
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--sft_config", default="configs/sft.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--reward_config", default="configs/reward.yaml")

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Which split to use from data_config (train_file or val_file).",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/rollouts",
        help="Root directory for rollout runs.",
    )
    parser.add_argument(
        "--sft_adapter_dir",
        default=None,
        help="Optional override for SFT adapter directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)

    # Sampling controls.
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # Environment controls.
    parser.add_argument("--database_root", default=None)
    parser.add_argument("--timeout_s", type=float, default=2.0)
    parser.add_argument("--max_rows", type=int, default=200)

    return parser.parse_args()


def main() -> None:
    """Entry point for rollout collection."""
    args = _parse_args()
    set_seed(args.seed)

    model_cfg = load_config(args.model_config)
    sft_cfg = load_config(args.sft_config)
    data_cfg = load_config(args.data_config)
    reward_cfg_raw = load_config(args.reward_config)

    run_dir = Path(args.output_dir) / f"rollouts_{timestamp()}"
    setup_logging(run_dir, "collect_rollouts.log")
    logging.info("Run directory: %s", run_dir)

    # Resolve dataset path.
    data_file = (
        data_cfg["train_file"]
        if args.split == "train"
        else data_cfg["val_file"]
    )
    logging.info("Using data split: %s (%s)", args.split, data_file)

    database_root = args.database_root or data_cfg.get("database_file")
    if database_root is None:
        raise ValueError("Database root not found.")

    reward_config = RewardConfig.from_dict(reward_cfg_raw)
    tokenizer = load_tokenizer(model_cfg["model_name"])

    dtype = resolve_dtype(model_cfg.get("dtype", "float16"))
    sft_adapter_dir = resolve_sft_adapter_dir(sft_cfg, args.sft_adapter_dir)
    logging.info("Loading SFT adapter from: %s", sft_adapter_dir)

    model = _load_policy_model(model_cfg["model_name"], dtype, sft_adapter_dir)
    model.config.pad_token_id = tokenizer.pad_token_id

    eos_token_ids = _configure_generation_tokens(model, tokenizer)

    sandbox = SqlSandbox(
        database_root=database_root,
        timeout_s=args.timeout_s,
        max_rows=args.max_rows,
    )

    rollouts_path = run_dir / "rollouts.jsonl"
    logging.info("Writing rollouts to: %s", rollouts_path)

    device = next(model.parameters()).device

    meta = {
        "timestamp": run_dir.name,
        "seed": args.seed,
        "model_name": model_cfg["model_name"],
        "sft_adapter_dir": sft_adapter_dir.as_posix(),
        "data_file": data_file,
        "database_root": str(database_root),
        "sampling": {
            "num_candidates": args.num_candidates,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "env": {
            "timeout_s": args.timeout_s,
            "max_rows": args.max_rows,
        },
        "reward": dataclasses.asdict(reward_config),
    }
    save_run_meta(run_dir, meta)

    total = 0
    kept = 0
    matches = 0

    with rollouts_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(
            _iter_examples(data_file, max_examples=args.max_examples)
        ):
            question = ex.get("question", "")
            schema = ex.get("schema", "")
            gold_sql = ex.get("sql", "")
            db_id = ex.get("db_id", "")

            if not question or not schema or not gold_sql or not db_id:
                logging.warning(
                    "Skipping example %d due to missing fields.", idx
                )
                continue

            prompt = build_chatml_prompt(schema=schema, question=question)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=int(model_cfg.get("max_length", 2048)),
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Generate all candidates in a single forward pass.
            with torch.inference_mode():
                generate_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "do_sample": True,
                    "num_return_sequences": args.num_candidates,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if eos_token_ids is not None:
                    generate_kwargs["eos_token_id"] = eos_token_ids

                outputs = model.generate(**generate_kwargs)

            # Score each candidate.
            for k in range(outputs.shape[0]):
                total += 1
                gen_ids = outputs[k, input_ids.shape[1] :]
                decoded = tokenizer.decode(
                    gen_ids, skip_special_tokens=False
                )
                candidate_sql = _extract_sql_from_generated(decoded)

                try:
                    reward_res = compute_execution_reward(
                        sandbox=sandbox,
                        db_id=db_id,
                        candidate_sql=candidate_sql,
                        gold_sql=gold_sql,
                        config=reward_config,
                        strict_gold=True,
                    )
                except ValueError as exc:
                    logging.warning(
                        "Gold execution failed at example %d: %s",
                        idx,
                        str(exc),
                    )
                    break

                if reward_res.matched:
                    matches += 1

                record = {
                    "id": f"{args.split}-{idx:06d}",
                    "candidate_id": k,
                    "db_id": db_id,
                    "prompt": prompt,
                    "gold_sql": gold_sql,
                    "candidate_sql": candidate_sql,
                    "reward": reward_res.reward,
                    "matched": reward_res.matched,
                    "valid": reward_res.candidate.error_type not in (
                        SqlErrorType.FORBIDDEN,
                    ),
                    "exec_ok": reward_res.candidate.ok,
                    "error_type": reward_res.candidate.error_type.value,
                    "error_message": reward_res.candidate.error_message,
                    "exec_time_ms": reward_res.candidate.elapsed_ms,
                    "truncated": reward_res.candidate.truncated,
                    "reason": reward_res.reason,
                }
                write_jsonl_line(f, record)
                kept += 1

            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

            if (idx + 1) % 25 == 0:
                match_rate = (matches / kept) if kept else 0.0
                logging.info(
                    "Processed %d examples | candidates=%d | kept=%d "
                    "| match_rate=%.3f",
                    idx + 1,
                    total,
                    kept,
                    match_rate,
                )

    match_rate = (matches / kept) if kept else 0.0
    logging.info(
        "Done. candidates=%d kept=%d match_rate=%.3f",
        total,
        kept,
        match_rate,
    )
    logging.info("Rollouts saved to: %s", rollouts_path)


if __name__ == "__main__":
    main()
