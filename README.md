# DeepSynth Parity Experiments

This repository evaluates language models on the DeepSynth benchmark, including agent-based evaluation using Claude Code following Harbor's approach.

## Quick Start

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Run DeepSynth with Claude Code (agent-based, direct CLI)
python run_original_claude_code.py \
    --dataset-json dev.json \
    --prompt-template original_prompt.md \
    --output-dir results/original/trial_1 \
    --model claude-haiku-4-5 \
    --concurrency 20
```

## Experimental Results

### DeepSynth Benchmark (n=40)

| Model | Type | Mean F1 | Notes |
|-------|------|---------|-------|
| claude-haiku-4-5 | Agent (Claude Code) | 0.275 | 40 questions, 1 run |

### Notes:

- claude-code runs the specified model via Claude Code CLI on the host, following Harbor's experimental setup
- Each question requires synthesizing information from multiple web sources into a structured JSON dictionary
- Grading is deterministic F1 over key-value pairs (no LLM judge) to ensure reproducibility
- Many questions failed due to web access limitations in CLI mode — the model returned apologetic text instead of JSON answers

## Evaluation Details

- **Models:** claude-haiku-4-5 (via Claude Code agent)
- **Timeout:** 1200 seconds per task
- **Concurrency:** 20
- **Grader:** deterministic F1 over key-value pairs
- **Dataset:** [DeepSynthesisTeam/deepsynth-bench](https://huggingface.co/datasets/DeepSynthesisTeam/deepsynth-bench) (40 dev questions)

## Author

[Chao Beyond Zhou](mailto:thinkstepbystep@gmail.com)
