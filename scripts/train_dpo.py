"""Trains a DPO policy for the SQL agent using QLoRA."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

from sql_agent.models.tokenizer import load_tokenizer
from sql_agent.utils import (
    build_quantization_config,
    resolve_dtype,
    resolve_sft_adapter_dir,
    save_json,
    setup_logging,
    timestamp,
)
from sql_agent.utils.config import load_config


@dataclasses.dataclass
class DatasetStats:
    """Basic dataset statistics.

    Attributes:
        rows_read: Total rows read from file.
        rows_kept: Rows that passed validation.
        rows_dropped_missing: Rows dropped due to missing fields.
    """

    rows_read: int = 0
    rows_kept: int = 0
    rows_dropped_missing: int = 0

    def to_dict(self) -> dict[str, int]:
        """Returns stats as a plain dictionary."""
        return dataclasses.asdict(self)


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SQL DPO model from preferences."
    )

    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--sft_config", default="configs/sft.yaml")
    parser.add_argument("--dpo_config", default="configs/dpo.yaml")

    parser.add_argument(
        "--train_preferences_path",
        required=True,
        help="Path to training preferences.jsonl.",
    )
    parser.add_argument(
        "--eval_preferences_path",
        default=None,
        help="Optional path to evaluation preferences.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional override for the DPO run output directory.",
    )
    parser.add_argument(
        "--sft_adapter_dir",
        default=None,
        help="Optional override for SFT adapter directory.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional Trainer checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional limit on number of training preference rows.",
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="Optional limit on number of evaluation preference rows.",
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If a required input file is missing.
        ValueError: If an argument is invalid.
    """
    train_path = Path(args.train_preferences_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training preferences file not found: {train_path}"
        )

    if args.eval_preferences_path is not None:
        eval_path = Path(args.eval_preferences_path)
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Evaluation preferences file not found: {eval_path}"
            )

    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max_train_examples must be > 0")
    if args.max_eval_examples is not None and args.max_eval_examples <= 0:
        raise ValueError("--max_eval_examples must be > 0")


def _cfg_get(
    cfg: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Reads a config value from either top level or ``cfg["train"]``.

    Args:
        cfg: Loaded config dictionary.
        key: Config key to look up.
        default: Default value if key is absent at both levels.

    Returns:
        Config value or ``default``.
    """
    if key in cfg:
        return cfg[key]
    train_cfg = cfg.get("train", {})
    if isinstance(train_cfg, dict) and key in train_cfg:
        return train_cfg[key]
    return default


def _resolve_output_dir(
    dpo_cfg: dict[str, Any],
    override_output_dir: Optional[str],
) -> Path:
    """Builds the run output directory.

    Args:
        dpo_cfg: Loaded DPO config.
        override_output_dir: Optional CLI override.

    Returns:
        Output run directory path.
    """
    if override_output_dir:
        return Path(override_output_dir)

    base_dir = _cfg_get(dpo_cfg, "output_dir", "runs/dpo")
    run_name = _cfg_get(dpo_cfg, "run_name", "sql_dpo")
    return Path(base_dir) / f"{run_name}_{timestamp()}"


def _load_preferences_dataset(
    path: str,
    *,
    max_examples: Optional[int] = None,
) -> tuple[Dataset, DatasetStats]:
    """Loads and validates a preference dataset from JSONL.

    Args:
        path: Path to ``preferences.jsonl``.
        max_examples: Optional row limit for debugging.

    Returns:
        A ``(dataset, stats)`` tuple.
    """
    raw_dataset = load_dataset("json", data_files=path, split="train")
    if max_examples is not None:
        raw_dataset = raw_dataset.select(
            range(min(len(raw_dataset), max_examples))
        )

    stats = DatasetStats(rows_read=len(raw_dataset))
    kept_rows: list[dict[str, str]] = []

    for row in raw_dataset:
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()

        if not prompt or not chosen or not rejected:
            stats.rows_dropped_missing += 1
            continue

        kept_rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
        stats.rows_kept += 1

    dataset = Dataset.from_list(kept_rows)
    return dataset, stats


def _load_trainable_policy_model(
    model_name: str,
    dtype: torch.dtype,
    sft_adapter_dir: Path,
) -> PreTrainedModel:
    """Loads the DPO policy model with a trainable LoRA adapter.

    Args:
        model_name: Base model name or HuggingFace path.
        dtype: Compute dtype.
        sft_adapter_dir: Path to the saved SFT adapter.

    Returns:
        Trainable PEFT model with both adapters loaded.
    """
    quant_config = build_quantization_config(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_dir.as_posix(),
        is_trainable=True,
    )
    model.config.use_cache = False

    return model


def _load_reference_model(
    model_name: str,
    dtype: torch.dtype,
    sft_adapter_dir: Path,
) -> PreTrainedModel:
    """Loads a frozen reference model for DPO.

    The reference model mirrors the SFT adapter but remains fully frozen.
    """
    quant_config = build_quantization_config(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    ref_model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_dir.as_posix(),
        is_trainable=False,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.config.use_cache = False
    return ref_model


def _build_training_args(
    dpo_cfg: dict[str, Any],
    output_dir: Path,
    tokenizer: Any,
) -> DPOConfig:
    """Builds TRL ``DPOConfig`` from the loaded config dictionary.

    Args:
        dpo_cfg: Loaded DPO config dictionary.
        output_dir: Output directory for this run.
        tokenizer: Tokenizer instance (used to resolve pad token).

    Returns:
        Configured ``DPOConfig`` object.

    Raises:
        ValueError: If tokenizer has neither ``pad_token`` nor
            ``eos_token`` defined.
    """
    pad_token = tokenizer.pad_token
    if pad_token is None:
        pad_token = tokenizer.eos_token
    if pad_token is None:
        raise ValueError(
            "Tokenizer must define either pad_token or eos_token."
        )

    max_length = _cfg_get(dpo_cfg, "max_length", 512)
    if max_length is not None:
        max_length = int(max_length)

    gradient_checkpointing = bool(
        _cfg_get(dpo_cfg, "gradient_checkpointing", True)
    )
    gradient_checkpointing_kwargs = _cfg_get(
        dpo_cfg,
        "gradient_checkpointing_kwargs",
        None,
    )

    return DPOConfig(
        output_dir=output_dir.as_posix(),
        run_name=_cfg_get(dpo_cfg, "run_name", output_dir.name),

        # Core optimization hyperparameters.
        learning_rate=float(_cfg_get(dpo_cfg, "learning_rate", 5e-6)),
        per_device_train_batch_size=int(
            _cfg_get(dpo_cfg, "per_device_train_batch_size", 1)
        ),
        per_device_eval_batch_size=int(
            _cfg_get(dpo_cfg, "per_device_eval_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            _cfg_get(dpo_cfg, "gradient_accumulation_steps", 16)
        ),

        # Training length controls.
        num_train_epochs=float(
            _cfg_get(dpo_cfg, "num_train_epochs", 1.0)
        ),
        max_steps=int(_cfg_get(dpo_cfg, "max_steps", -1)),

        # Warmup and learning rate schedule.
        warmup_ratio=float(_cfg_get(dpo_cfg, "warmup_ratio", 0.0)),
        lr_scheduler_type=str(
            _cfg_get(dpo_cfg, "lr_scheduler_type", "linear")
        ),

        # Optimizer configuration.
        optim=str(_cfg_get(dpo_cfg, "optim", "paged_adamw_8bit")),
        weight_decay=float(_cfg_get(dpo_cfg, "weight_decay", 0.0)),
        max_grad_norm=float(_cfg_get(dpo_cfg, "max_grad_norm", 1.0)),

        # Logging and checkpointing frequency.
        logging_steps=int(_cfg_get(dpo_cfg, "logging_steps", 10)),
        save_steps=int(_cfg_get(dpo_cfg, "save_steps", 200)),
        save_total_limit=int(_cfg_get(dpo_cfg, "save_total_limit", 2)),
        save_strategy=str(_cfg_get(dpo_cfg, "save_strategy", "steps")),

        # Evaluation settings.
        eval_strategy=str(_cfg_get(dpo_cfg, "eval_strategy", "no")),
        eval_steps=_cfg_get(dpo_cfg, "eval_steps", None),
        load_best_model_at_end=bool(
            _cfg_get(dpo_cfg, "load_best_model_at_end", False)
        ),
        metric_for_best_model=str(
            _cfg_get(dpo_cfg, "metric_for_best_model", "eval_loss")
        ),
        greater_is_better=bool(
            _cfg_get(dpo_cfg, "greater_is_better", False)
        ),
        report_to=str(_cfg_get(dpo_cfg, "report_to", "none")),

        # Reproducibility and mixed precision settings.
        seed=int(_cfg_get(dpo_cfg, "seed", 42)),
        fp16=bool(_cfg_get(dpo_cfg, "fp16", True)),
        bf16=bool(_cfg_get(dpo_cfg, "bf16", False)),

        # Memory-saving flags.
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,

        # Drop unused dataset columns before passing to the model.
        remove_unused_columns=bool(
            _cfg_get(dpo_cfg, "remove_unused_columns", False)
        ),

        # DPO-specific parameters.
        beta=float(_cfg_get(dpo_cfg, "beta", 0.1)),
        label_smoothing=float(
            _cfg_get(dpo_cfg, "label_smoothing", 0.0)
        ),
        loss_type=str(_cfg_get(dpo_cfg, "loss_type", "sigmoid")),
        max_length=max_length,
        truncation_mode=str(
            _cfg_get(dpo_cfg, "truncation_mode", "keep_start")
        ),
        pad_token=pad_token,
        precompute_ref_log_probs=bool(
            _cfg_get(dpo_cfg, "precompute_ref_log_probs", True)
        ),
        dataset_num_proc=_cfg_get(dpo_cfg, "dataset_num_proc", None),
        disable_dropout=bool(
            _cfg_get(dpo_cfg, "disable_dropout", True)
        ),
        logging_first_step=bool(
            _cfg_get(dpo_cfg, "logging_first_step", True)
        ),
    )


def main() -> None:
    """Entry point for DPO training."""
    args = _parse_args()
    _validate_args(args)

    model_cfg = load_config(args.model_config)
    sft_cfg = load_config(args.sft_config)
    dpo_cfg = load_config(args.dpo_config)

    run_dir = _resolve_output_dir(dpo_cfg, args.output_dir)
    setup_logging(run_dir, "train_dpo.log")

    seed = int(_cfg_get(dpo_cfg, "seed", 42))
    set_seed(seed)

    logging.info("Run directory: %s", run_dir)
    logging.info("Loading configs:")
    logging.info("  model: %s", args.model_config)
    logging.info("  sft:   %s", args.sft_config)
    logging.info("  dpo:   %s", args.dpo_config)

    dtype = resolve_dtype(
        _cfg_get(dpo_cfg, "compute_dtype", model_cfg.get("dtype", "float16"))
    )

    sft_adapter_dir = resolve_sft_adapter_dir(sft_cfg, args.sft_adapter_dir)
    if not sft_adapter_dir.exists():
        raise FileNotFoundError(
            f"SFT adapter directory not found: {sft_adapter_dir}"
        )

    logging.info("Loading tokenizer: %s", model_cfg["model_name"])
    tokenizer = load_tokenizer(model_cfg["model_name"])
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(
        "Loading training preferences: %s",
        args.train_preferences_path,
    )
    train_dataset, train_stats = _load_preferences_dataset(
        args.train_preferences_path,
        max_examples=args.max_train_examples,
    )
    logging.info(
        "Train dataset rows kept: %d / %d",
        train_stats.rows_kept,
        train_stats.rows_read,
    )

    eval_dataset = None
    eval_stats = None
    if args.eval_preferences_path:
        logging.info(
            "Loading eval preferences: %s", args.eval_preferences_path
        )
        eval_dataset, eval_stats = _load_preferences_dataset(
            args.eval_preferences_path,
            max_examples=args.max_eval_examples,
        )
        logging.info(
            "Eval dataset rows kept: %d / %d",
            eval_stats.rows_kept,
            eval_stats.rows_read,
        )

    logging.info(
        "Loading base model + trainable SFT adapter from: %s",
        sft_adapter_dir,
    )
    model = _load_trainable_policy_model(
        model_name=model_cfg["model_name"],
        dtype=dtype,
        sft_adapter_dir=sft_adapter_dir,
    )
    ref_model = _load_reference_model(
        model_name=model_cfg["model_name"],
        dtype=dtype,
        sft_adapter_dir=sft_adapter_dir,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_args = _build_training_args(dpo_cfg, run_dir, tokenizer)
    if training_args.eval_strategy != "no" and eval_dataset is None:
        raise ValueError(
            "Evaluation is enabled in dpo config, but --eval_preferences_path "
            "was not provided."
        )

    run_meta = {
        "timestamp": run_dir.name,
        "model_name": model_cfg["model_name"],
        "dtype": model_cfg.get("dtype", "float16"),
        "sft_adapter_dir": sft_adapter_dir.as_posix(),
        "train_preferences_path": Path(
            args.train_preferences_path
        ).as_posix(),
        "eval_preferences_path": (
            Path(args.eval_preferences_path).as_posix()
            if args.eval_preferences_path
            else None
        ),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "train_dataset_stats": train_stats.to_dict(),
        "eval_dataset_stats": (
            eval_stats.to_dict() if eval_stats is not None else None
        ),
        "dpo_args": training_args.to_dict(),
        "versions": {"torch": torch.__version__},
    }
    save_json(run_dir / "run_meta.json", run_meta)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logging.info("Starting DPO training.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logging.info("Saving final model.")
    final_dir = run_dir / "final"
    trainer.save_model(final_dir.as_posix())
    tokenizer.save_pretrained(final_dir.as_posix())

    train_result = trainer.state.log_history
    save_json(
        run_dir / "train_log_history.json",
        {"log_history": train_result},
    )

    logging.info("Training completed.")
    logging.info("Final adapter saved to: %s", final_dir)


if __name__ == "__main__":
    main()
