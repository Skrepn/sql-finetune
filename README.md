# sql-finetune

Fine-tuning a 0.5B language model for text-to-SQL generation using **SFT -> DPO** pipeline on the [Spider](https://huggingface.co/datasets/xlangai/spider) benchmark - trained entirely on a single GPU (GTX 1660 SUPER, 6 GB VRAM).

## Results

Execution accuracy on the full Spider validation set (1034 examples, greedy decoding):

| Model | Accuracy | vs Base |
|:------|:--------:|:---------:|
| Qwen2.5-Coder-0.5B (base) | 0.4% | - |
| + SFT (QLoRA) | 24.3% | +23.9% |
| + DPO | **36.9%** | **+36.5%** |

Breakdown by query difficulty:

| Difficulty | Base | SFT | DPO |
|:-----------|:----:|:---:|:---:|
| Easy | 0.6% | 36.0% | **53.8%** |
| Medium | 0.5% | 22.0% | **32.6%** |
| Hard | 0.0% | 17.0% | **27.8%** |
| Extra | 0.0% | 5.0% | **12.5%** |

## Approach

**Stage 1 - Supervised Fine-Tuning (SFT).** The base model is fine-tuned on 7000 (question, schema) -> SQL pairs using QLoRA (4-bit quantization, LoRA rank 8). Prompts are formatted in ChatML with CREATE TABLE DDL as schema representation. Training uses `load_best_model_at_end` with eval loss to select the optimal checkpoint.

**Stage 2 - Rollout Collection.** The SFT model generates 8 candidate SQL queries per training example (temperature 0.8). Each candidate is executed in a read-only SQLite sandbox and scored by comparing result sets against the gold query (execution accuracy, not exact match).

**Stage 3 - Preference Pair Construction.** Candidates are grouped by example. The highest-reward candidate becomes "chosen", the lowest becomes "rejected". Only pairs with a reward gap > 0.3 are kept. Invalid candidates (forbidden SQL) are filtered out; syntax and runtime errors are kept as useful negative signal.

**Stage 4 - Direct Preference Optimization (DPO).** The SFT is used as both the initialization and the reference model. Key hyperparameters: β=0.3 (strong KL penalty to prevent overfitting), lr=2e-6, 1 epoch. Reference log-probabilities are precomputed to save GPU memory.

### Key Design Decisions

**Execution-based reward.** Candidate SQL is evaluated by running it against the actual database and comparing result sets with the gold query. This is more robust than exact string match - semantically equivalent queries (e.g., `COUNT(*)` vs `COUNT(id)`) are correctly scored as matches.

**SQL extraction from model output.** A 0.5B model often generates correct SQL followed by unicode hallucinations or repeated text. A dedicated algorithm removes unwanted tokens (non-ASCII characters, ChatML tokens, and corrupted fragments) and keeps only valid SQL.

**Conservative DPO.** Early experiments with aggressive settings (β=0.05, lr=5e-6) degraded the model. The final configuration (β=0.3, lr=2e-6, 1 epoch) keeps the policy close to the SFT reference while still learning from preferences.

**SFT as reference model.** I experimented with multiple rounds of DPO using the SFT adapter as a fixed reference model. However, performance degradation was observed already after the first DPO round when attempting further iterations, likely due to drift accumulation. New rollouts were generated after the first round, but additional DPO training did not lead to improvements.

## Project Structure

```
sql-finetune/
├── configs/
│   ├── model.yaml            # Model name, dtype, max_length
│   ├── sft.yaml              # SFT hyperparameters + LoRA config
│   ├── dpo.yaml              # DPO hyperparameters
│   ├── data.yaml             # Dataset paths
│   └── reward.yaml           # Reward function weights
│
├── scripts/
│   ├── download_data.py      # Download raw Spider dataset
│   ├── preprocess_data.py    # Convert to SFT format (schema + question -> SQL)
│   ├── train_sft.py          # QLoRA supervised fine-tuning
│   ├── collect_rollout.py    # Generate candidates + compute rewards
│   ├── build_preferences.py  # Construct chosen/rejected pairs
│   ├── train_dpo.py          # DPO training
│   └── eval.py               # Final evaluation
│
├── sql_agent/
│   ├── dataset_utils/
│   │   ├── prompts.py        # ChatML prompt template
│   │   └── preprocessing.py  # Tokenization and collation
│   ├── env/
│   │   └── sql_sandbox.py    # Read-only SQLite execution sandbox
│   ├── models/
│   │   └── tokenizer.py      # Tokenizer loading
│   ├── reward/
│   │   └── execution_reward.py  # Reward function (match/mismatch/error)
│   └── utils/                # Config, I/O, logging, timestamps, dtype helpers
│
├── tests/                    # Unit tests (pytest)
└── pyproject.toml            # Package config + tool settings
```

## Quick Start

### Installation

```bash
git clone https://github.com/<username>/sql-finetune.git
cd sql-finetune
pip install -e ".[dev]"
```

### Data Preparation

```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

### Training Pipeline

```bash
# Stage 1: SFT
python scripts/train_sft.py

# Stage 2: Collect rollouts
python -m scripts.collect_rollout \
    --sft_adapter_dir runs/sft/.../final \

# Stage 3: Build preference pairs
python -m scripts.build_preferences \
    --rollouts_path runs/rollouts/.../rollouts.jsonl \

# Stage 4: DPO
python -m scripts.train_dpo \
    --train_preferences_path runs/preferences/.../preferences.jsonl \
    --sft_adapter_dir runs/sft/.../final
```

### Evaluation

```bash
# Full comparison: base -> SFT -> DPO
python -m scripts.eval \
    --include_base \
    --adapter_dir runs/sft/.../final \
                  runs/dpo/.../final \
    --split val
```

### Tests

```bash
pytest
```

## Hardware

All experiments were run on a single machine:

| Component | Spec |
|:----------|:-----|
| GPU | NVIDIA GTX 1660 SUPER (6 GB VRAM) |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Precision | float16 |
| SFT time | ~14 hours (2 epochs) |
| DPO time | ~4 hours (1 epoch, 4490 pairs) |
| Evaluation | ~5 hours (1034 examples, greedy) |

## Lessons Learned

1. **DPO hyperparameters matter more than data quantity.** 4490 preference pairs with β=0.3 and lr=2e-6 outperformed 4563 pairs with β=0.05 and lr=5e-6 - the aggressive configuration degraded the model below SFT baseline.

2. **The `valid` field definition is critical.** Initially, only candidates with `exec_ok=True` were eligible for preferences, leaving only reward 1.0 vs 0.1 pairs. Changing the filter to exclude only `FORBIDDEN` errors (while keeping syntax/runtime errors as valid rejected candidates) tripled the number of preference pairs and improved diversity.

3. **SQL extraction saves ~20% accuracy.** The 0.5B model often generates correct SQL followed by unicode garbage. Without the extraction pipeline, match rate drops from ~60% to ~4% on training datasets.

4. **`load_best_model_at_end` is essential for SFT.** Train loss keeps decreasing, but eval loss plateaus around epoch 0.8–0.9 and rises after - a classic sign of overfitting. Without this flag, the saved checkpoint is the last (overfit) one rather than the best.

5. **Iterative DPO doesn't work at 0.5B scale.** Running multiple DPO rounds with SFT as a fixed reference degraded performance after the first iteration, even with fresh rollouts. The model is likely too small to absorb repeated distribution shifts without drifting too far from the reference.

## Tech Stack

- **Model:** [Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- **Fine-tuning:** [PEFT](https://github.com/huggingface/peft) (QLoRA), [TRL](https://github.com/huggingface/trl) (DPO)
- **Dataset:** [Spider](https://huggingface.co/datasets/xlangai/spider) (7000 train / 1034 val)
- **Evaluation:** Execution accuracy via SQLite sandbox
- **Code quality:** pytest

## License

MIT
