# DeepSynth Parity Experiments

Original-side parity scripts for the [DeepSynth Harbor adapter](https://github.com/harbor-framework/harbor/pull/1112). These scripts run the DeepSynth benchmark **outside** Harbor (directly via Claude Code CLI) to produce original-side scores for parity comparison.

## Quick Start

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Run DeepSynth with Claude Code (agent-based, direct CLI)
python run_original_claude_code.py --dataset-json dev.json --prompt-template original_prompt.md --output-dir results/original/trial_1 --model claude-haiku-4-5 --concurrency 20
```

## Experimental Results

### DeepSynth Benchmark (n=40)

| Trial | Mean F1 | Parse Errors | Missing | Scored > 0 |
|-------|---------|--------------|---------|------------|
| 4     | 0.075   | 29 (72.5%)   | 2       | 3 (Q11=1.0, Q29=1.0, Q37=1.0) |
| 5     | 0.095   | 23 (57.5%)   | 1       | 4 (Q24=1.0, Q29=0.8, Q30=1.0, Q49=1.0) |
| 6     | 0.108   | 22 (55.0%)   | 1       | 5 (Q29=1.0, Q36=0.33, Q53=1.0, Q68=1.0, Q74=1.0) |
| **Avg** | **0.093 ± 0.010** | **24.7 (61.7%)** | **1.3** | |

All trials use **claude-haiku-4-5** via Claude Code CLI (`claude-code@2.1.73`) with LLM judge enabled (0 judge upgrades across all runs).

### About the Earlier 0.275 Result

An earlier version of this README reported Mean F1 = 0.275 from a single trial. That result was produced by a buggy output extraction pipeline: the runner parsed JSON directly from the raw `stream-json` event output instead of first extracting the agent's text responses. This caused it to match JSON fragments from stream metadata events rather than the agent's actual answers, inflating the score. The pipeline was corrected in commit `a4df409` by adding `_extract_text_from_stream()`, which properly reconstructs the agent's text from `assistant` message blocks before JSON extraction. Trials 4-6 above use the corrected pipeline and are the authoritative results.

### Key Observations

- **Parse errors are the dominant failure mode** (55-82% of tasks), indicating the model frequently returns non-JSON output (apologetic text, malformed responses, or incomplete answers)
- When output parses successfully, scores are still mostly 0 — the judge rarely upgrades F1=0 answers
- High variance across trials: different questions succeed each run, with Q29 being the most consistent (scored in 3/4 trials)

## Evaluation Details

- **Model:** claude-haiku-4-5 via Claude Code CLI (`claude-code@2.1.73`)
- **Trials:** 3 runs (trial_4 through trial_6)
- **Timeout:** 1200 seconds per task
- **Concurrency:** 20
- **Grader:** deterministic F1 over key-value pairs + LLM judge fallback (Claude Haiku 4.5)
- **Dataset:** [DeepSynthesisTeam/deepsynth-bench](https://huggingface.co/datasets/DeepSynthesisTeam/deepsynth-bench) (40 dev questions)

## Parity Comparison

See the [Harbor adapter README](https://github.com/harbor-framework/harbor/pull/1112) for the full parity table. Summary:

| Side | Mean F1 | Agent Version |
|------|---------|---------------|
| Original (this repo) | 0.093 ± 0.010 | claude-code@2.1.73 |
| Harbor | 0.078 ± 0.006 | claude-code@2.1.73 |

Error bars overlap — Harbor introduces no significant accuracy distortion.

## Author

[Chao Beyond Zhou](mailto:thinkstepbystep@gmail.com)
