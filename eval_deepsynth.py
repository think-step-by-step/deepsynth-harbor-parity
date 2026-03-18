#!/usr/bin/env python3
"""Standalone batch F1 evaluator for DeepSynth predictions.

Scoring functions (flatten_json, normalize_value, compute_f1) are copied
VERBATIM from template/tests/test_outputs.py to guarantee identical scoring
between Harbor-side and original-side evaluation.

Usage:
    python eval_deepsynth.py \
        --predictions predictions.json \
        --dataset-json dev.json \
        --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring functions — VERBATIM copy from template/tests/test_outputs.py
# (lines 65-118). DO NOT modify these without updating the template too.
# ---------------------------------------------------------------------------


def flatten_json(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict: {"a": {"b": 1}} -> {"a.b": "1"}."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key))
        else:
            items[new_key] = normalize_value(v)
    return items


def normalize_value(v) -> str:
    """Canonical form: floats stripped of trailing zeros, strings lowered and stripped."""
    if isinstance(v, float):
        # Format without unnecessary trailing zeros
        formatted = f"{v:.10f}".rstrip("0").rstrip(".")
        return formatted
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if v is None:
        return "null"
    # String: lowercase and strip whitespace
    return str(v).lower().strip()


def compute_f1(gold: dict, model: dict) -> float:
    """Precision/recall/F1 over exact key-value matches."""
    gold_flat = flatten_json(gold)
    model_flat = flatten_json(model)

    if not gold_flat and not model_flat:
        return 1.0
    if not gold_flat or not model_flat:
        return 0.0

    # Normalize keys for comparison (lowercase, strip)
    gold_pairs = {k.lower().strip(): v for k, v in gold_flat.items()}
    model_pairs = {k.lower().strip(): v for k, v in model_flat.items()}

    # Count exact key-value matches
    matches = 0
    for k, v in model_pairs.items():
        if k in gold_pairs and gold_pairs[k] == v:
            matches += 1

    precision = matches / len(model_pairs) if model_pairs else 0.0
    recall = matches / len(gold_pairs) if gold_pairs else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# End verbatim copy
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tier 2: LLM judge — ported from harbor's test_outputs.py
# ---------------------------------------------------------------------------


def llm_judge(
    predicted_str: str,
    expected_str: str,
    question: str,
    qnum: str = "",
    judge_log: list | None = None,
) -> float | None:
    """Call Claude Haiku to judge semantic equivalence. Returns 1.0, 0.0, or None on failure.

    When *judge_log* is provided, every call (success or failure) is appended
    as a dict so the caller can persist it for debugging.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        entry = {"qnum": qnum, "status": "skipped", "reason": "no ANTHROPIC_API_KEY"}
        print(f"  [judge] Q{qnum}: SKIPPED — no ANTHROPIC_API_KEY", file=sys.stderr)
        if judge_log is not None:
            judge_log.append(entry)
        return None

    try:
        import anthropic
    except ImportError:
        entry = {"qnum": qnum, "status": "skipped", "reason": "anthropic package not installed"}
        print(f"  [judge] Q{qnum}: SKIPPED — anthropic package not installed", file=sys.stderr)
        if judge_log is not None:
            judge_log.append(entry)
        return None

    prompt = (
        "You are an expert data-extraction evaluator.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer (gold): {expected_str}\n\n"
        f"Model predicted answer: {predicted_str}\n\n"
        "Are the model's answers semantically equivalent to the expected answers? "
        "Minor formatting differences (whitespace, key ordering, quote style) should be ignored. "
        "Respond with ONLY the letter A if correct, or B if incorrect."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip().upper()
        score = 1.0 if answer == "A" else 0.0
        entry = {
            "qnum": qnum,
            "status": "ok",
            "verdict": answer,
            "score": score,
            "prompt": prompt,
            "predicted": predicted_str,
            "expected": expected_str,
        }
        print(f"  [judge] Q{qnum}: verdict={answer} score={score}", file=sys.stderr)
        if judge_log is not None:
            judge_log.append(entry)
        return score
    except Exception as e:
        entry = {
            "qnum": qnum,
            "status": "error",
            "reason": str(e),
            "predicted": predicted_str,
            "expected": expected_str,
        }
        print(f"  [judge] Q{qnum}: ERROR — {e}", file=sys.stderr)
        if judge_log is not None:
            judge_log.append(entry)
        return None


def _fix_json(s: str) -> str:
    """Attempt to fix common JSON issues."""
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    open_braces = s.count("{") - s.count("}")
    if open_braces > 0:
        s += "}" * open_braces
    s = re.sub(
        r'("(?:[^"\\]|\\.)*")\s*,\s*(?=\d)',
        r"\1: ",
        s,
    )
    return s


def parse_json_string(s: str) -> dict:
    """Parse JSON from various formats."""
    s = s.strip()
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", s, re.DOTALL)
    if md_match:
        s = md_match.group(1).strip()
    ans_match = re.match(r"<Answer>\s*:?\s*(.*)", s, re.DOTALL)
    if ans_match:
        s = ans_match.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    fixed = _fix_json(s)
    return json.loads(fixed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSynth predictions (batch F1)"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help='Predictions JSON: {"001": {...}, "002": {...}}',
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="DeepSynth dev.json (list of records with 'Question Number', "
        "'Questions', 'Answers')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write eval_results.json (optional)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        help="Disable Tier 2 LLM judge; use deterministic F1 only",
    )
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))

    if isinstance(dataset, dict) and "data" in dataset:
        dataset = dataset["data"]

    # Build gold answers map: question_number -> answer
    gold_map: dict[str, dict] = {}
    question_map: dict[str, str] = {}
    for item in dataset:
        qnum = str(item["Question Number"]).strip()
        answer_raw = item["Answers"]
        if isinstance(answer_raw, str):
            answer_raw = parse_json_string(answer_raw)
        gold_map[qnum] = answer_raw
        question_map[qnum] = item["Questions"]

    use_judge = not args.no_llm_judge
    results = []
    f1_scores = []
    parse_errors = 0
    judge_upgrades = 0
    judge_log: list[dict] = []

    for qnum in sorted(gold_map.keys(), key=lambda x: x.zfill(10)):
        gold = gold_map[qnum]
        pred_raw = predictions.get(qnum)

        if pred_raw is None:
            print(f"Q{qnum}: MISSING prediction — score=0.0")
            results.append({"question_number": qnum, "f1": 0.0, "score": 0.0, "status": "missing"})
            f1_scores.append(0.0)
            continue

        # Parse prediction
        if isinstance(pred_raw, str):
            try:
                pred = parse_json_string(pred_raw)
            except (json.JSONDecodeError, ValueError):
                print(f"Q{qnum}: PARSE ERROR — score=0.0")
                results.append(
                    {"question_number": qnum, "f1": 0.0, "score": 0.0, "status": "parse_error"}
                )
                f1_scores.append(0.0)
                parse_errors += 1
                continue
        else:
            pred = pred_raw

        f1 = compute_f1(gold, pred)
        score = f1
        judged = False

        # Tier 2: LLM judge fallback when F1 < 1.0
        if use_judge and f1 < 1.0:
            pred_str = json.dumps(pred, sort_keys=True)
            gold_str = json.dumps(gold, sort_keys=True)
            judge_result = llm_judge(pred_str, gold_str, question_map[qnum], qnum=qnum, judge_log=judge_log)
            if judge_result is not None and judge_result == 1.0:
                score = 1.0
                judge_upgrades += 1
                judged = True

        f1_scores.append(score)
        status = "ok_judge_upgrade" if judged else "ok"
        results.append({"question_number": qnum, "f1": round(f1, 4), "score": round(score, 4), "status": status})
        if judged:
            print(f"Q{qnum}: F1={f1:.4f} -> judge upgraded to 1.0")
        else:
            print(f"Q{qnum}: F1={f1:.4f}")

    # Aggregate
    n = len(f1_scores)
    mean_score = sum(f1_scores) / n if n > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"Tasks evaluated:  {n}")
    print(f"Parse errors:     {parse_errors}")
    print(f"Judge upgrades:   {judge_upgrades}")
    print(f"Mean score:       {mean_score:.4f}")

    summary = {
        "n_tasks": n,
        "parse_errors": parse_errors,
        "judge_upgrades": judge_upgrades,
        "llm_judge_enabled": use_judge,
        "mean_score": round(mean_score, 4),
        "per_task": results,
    }

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "eval_results.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

        if judge_log:
            judge_path = args.output_dir / "judge_log.json"
            judge_path.write_text(json.dumps(judge_log, indent=2), encoding="utf-8")
            print(f"Wrote {judge_path} ({len(judge_log)} entries)")
        elif use_judge:
            print("WARNING: LLM judge was enabled but judge_log is empty — "
                  "check ANTHROPIC_API_KEY and `pip install anthropic`")

    sys.exit(0)


if __name__ == "__main__":
    main()
