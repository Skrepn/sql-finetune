"""Supervised fine-tuning (SFT) script for the SQL agent.

Trains a QLoRA adapter on top of a quantized base model using
prompt/completion pairs formatted in ChatML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sql_agent.dataset_utils.prompts import build_chatml_prompt, format_completion
from sql_agent.models.tokenizer import load_tokenizer
from sql_agent.utils import (
    build_quantization_config,
    load_config,
    resolve_dtype,
    setup_logging,
)


def _format_example(example: dict[str, str]) -> dict[str, str]:
    """Formats a dataset example into a ChatML prompt/completion pair.

    Args:
        example: A dictionary with ``question``, ``sql``, and
            ``schema`` keys.

    Returns:
        A dictionary with ``prompt`` and ``completion`` keys.
    """
    prompt = build_chatml_prompt(
        schema=example["schema"],
        question=example["question"],
    )
    completion = format_completion(example["sql"])
    return {"prompt": prompt, "completion": completion}


def _tokenize_batch(
    batch: dict[str, list[str]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenizes a batch and applies label masking on prompt tokens.

    The prompt portion and any padding tokens are masked with ``-100``
    in the labels so that the training loss is computed only on the
    SQL completion.

    Args:
        batch: Batch dictionary with ``prompt`` and ``completion``
            string lists.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum total sequence length.

    Returns:
        Tokenized dictionary with ``input_ids``, ``attention_mask``,
        and ``labels``.
    """
    prompts = batch["prompt"]
    completions = batch["completion"]

    full_texts = [p + c for p, c in zip(prompts, completions)]

    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding=False,
        add_special_tokens=False,
        max_length=max_length,
    )

    input_ids = tokenized["input_ids"]
    labels = [ids.copy() for ids in input_ids]

    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            add_special_tokens=False,
            max_length=max_length,
        )["input_ids"]

        prompt_len = len(prompt_ids)

        # Mask prompt tokens.
        for j in range(prompt_len):
            labels[i][j] = -100

    tokenized["labels"] = labels
    return tokenized


def _process_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    """Applies formatting and tokenization to a dataset split.

    Args:
        dataset: Raw HuggingFace dataset.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset ready for ``Trainer``.
    """
    dataset = dataset.map(
        _format_example,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.map(
        lambda batch: _tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["prompt", "completion"],
    )
    return dataset


def main() -> None:
    """Runs supervised fine-tuning."""
    set_seed(42)

    model_cfg = load_config("configs/model.yaml")
    sft_cfg = load_config("configs/sft.yaml")
    data_cfg = load_config("configs/data.yaml")

    # Set up logging.
    output_dir = Path(sft_cfg["output_dir"])
    setup_logging(output_dir, "train_sft.log")

    logging.info("Model: %s", model_cfg["model_name"])
    logging.info("Output: %s", output_dir)

    # Load tokenizer.
    tokenizer = load_tokenizer(model_cfg["model_name"])

    # Load and process datasets.
    train_dataset = load_dataset(
        "json", data_files=data_cfg["train_file"], split="train"
    )
    val_dataset = load_dataset(
        "json", data_files=data_cfg["val_file"], split="train"
    )

    max_length = int(model_cfg["max_length"])
    train_dataset = _process_dataset(train_dataset, tokenizer, max_length)
    val_dataset = _process_dataset(val_dataset, tokenizer, max_length)

    logging.info("Train examples: %d", len(train_dataset))
    logging.info("Val examples: %d", len(val_dataset))

    # Load quantized base model.
    dtype = resolve_dtype(model_cfg.get("dtype", "float16"))
    bnb_config = build_quantization_config(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        quantization_config=bnb_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    # Configure LoRA adapters.
    lora_cfg = sft_cfg.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Build training arguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(sft_cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(
            sft_cfg.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(sft_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(sft_cfg.get("warmup_ratio", 0.05)),
        num_train_epochs=float(sft_cfg.get("num_epochs", 3)),
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        save_steps=int(sft_cfg.get("save_steps", 200)),
        save_total_limit=int(sft_cfg.get("save_total_limit", 3)),
        eval_strategy="steps",
        eval_steps=int(sft_cfg.get("eval_steps", 200)),
        logging_steps=int(sft_cfg.get("logging_steps", 10)),
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    # Dynamic padding.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logging.info("Starting SFT training.")
    trainer.train()

    # Save final adapter.
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    logging.info("Training completed. Adapter saved to: %s", final_dir)


if __name__ == "__main__":
    main()
