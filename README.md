# DeepSynth Harbor Parity

Original-side parity scripts for the [DeepSynth Harbor adapter](https://github.com/laude-institute/harbor/tree/main/adapters/deepsynth).

These scripts run the DeepSynth benchmark **outside** Harbor (directly via Claude Code CLI) to produce original-side scores for parity comparison.

## Dataset

[DeepSynthesisTeam/deepsynth-bench](https://huggingface.co/datasets/DeepSynthesisTeam/deepsynth-bench) — 40 dev questions with gold answers. Each question requires synthesizing information from multiple web sources into a structured JSON dictionary.

A pre-downloaded copy of the dev set is included as `dev.json`.

## Setup

```bash
pip install -r requirements.txt

# API keys
export ANTHROPIC_API_KEY="your_key"
export HF_TOKEN="your_token"  # only needed if re-downloading the dataset
```

## Usage

### 1. Run original-side (3 trials)

```bash
python run_original_claude_code.py \
    --dataset-json dev.json \
    --prompt-template original_prompt.md \
    --output-dir results/original/trial_1 \
    --model anthropic/claude-haiku-4-5
```

Repeat for `trial_2` and `trial_3`.

### 2. Evaluate each trial

```bash
python eval_deepsynth.py \
    --predictions results/original/trial_1/predictions.json \
    --dataset-json dev.json \
    --output-dir results/original/trial_1
```

### 3. Run Harbor-side (in the Harbor repo)

```bash
# From the Harbor repo root
uv run harbor jobs start -c adapters/deepsynth/deepsynth-parity.yaml

# Export predictions from each Harbor job
python export_harbor_predictions.py \
    --jobs-dir /path/to/harbor/jobs/<job_id> \
    --output-dir results/harbor/trial_1

# Evaluate
python eval_deepsynth.py \
    --predictions results/harbor/trial_1/predictions.json \
    --dataset-json dev.json \
    --output-dir results/harbor/trial_1
```

### 4. Compare

```bash
python compare_scores.py \
    --original-results results/original/trial_{1,2,3}/eval_results.json \
    --harbor-results results/harbor/trial_{1,2,3}/eval_results.json \
    --output-dir results/comparison
```

## Metric

**F1 over key-value pairs**: precision and recall of exact key-value matches between the model's JSON answer and the gold answer. Deterministic only (no LLM judge) to ensure reproducibility.

## Files

| File | Description |
|------|-------------|
| `run_original_claude_code.py` | Run Claude Code CLI directly on each question |
| `eval_deepsynth.py` | Batch F1 evaluator for predictions |
| `compare_scores.py` | Compare original vs Harbor trial results |
| `export_harbor_predictions.py` | Extract predictions from Harbor job dirs |
| `prepare_dataset.py` | Download dataset from HuggingFace to JSON |
| `dev.json` | Pre-downloaded dev set (40 questions) |
| `original_prompt.md` | Prompt template for original-side runs |
| `deepsynth_parity.yaml` | Harbor-side job config (reference copy) |

## Author

[Chao Beyond Zhou](mailto:thinkstepbystep@gmail.com)
