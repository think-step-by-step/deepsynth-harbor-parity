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
    if isinstance(v, int):
        return str(v)
    if isinstance(v, bool):
        return str(v).lower()
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

    results = []
    f1_scores = []
    parse_errors = 0

    for qnum in sorted(gold_map.keys(), key=lambda x: x.zfill(10)):
        gold = gold_map[qnum]
        pred_raw = predictions.get(qnum)

        if pred_raw is None:
            print(f"Q{qnum}: MISSING prediction — F1=0.0")
            results.append({"question_number": qnum, "f1": 0.0, "status": "missing"})
            f1_scores.append(0.0)
            continue

        # Parse prediction
        if isinstance(pred_raw, str):
            try:
                pred = parse_json_string(pred_raw)
            except (json.JSONDecodeError, ValueError):
                print(f"Q{qnum}: PARSE ERROR — F1=0.0")
                results.append(
                    {"question_number": qnum, "f1": 0.0, "status": "parse_error"}
                )
                f1_scores.append(0.0)
                parse_errors += 1
                continue
        else:
            pred = pred_raw

        f1 = compute_f1(gold, pred)
        f1_scores.append(f1)
        results.append({"question_number": qnum, "f1": round(f1, 4), "status": "ok"})
        print(f"Q{qnum}: F1={f1:.4f}")

    # Aggregate
    n = len(f1_scores)
    mean_f1 = sum(f1_scores) / n if n > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"Tasks evaluated: {n}")
    print(f"Parse errors:    {parse_errors}")
    print(f"Mean F1:         {mean_f1:.4f}")

    summary = {
        "n_tasks": n,
        "parse_errors": parse_errors,
        "mean_f1": round(mean_f1, 4),
        "per_task": results,
    }

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "eval_results.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
