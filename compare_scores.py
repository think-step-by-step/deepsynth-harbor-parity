#!/usr/bin/env python3
"""Compare DeepSynth parity scores between original-side and Harbor-side trials.

Takes multiple eval_results.json files from each side, computes mean +/- std,
and prints a comparison table with pass/fail determination.

Usage:
    python compare_scores.py \
        --original-results trial1/eval_results.json trial2/eval_results.json trial3/eval_results.json \
        --harbor-results trial1/eval_results.json trial2/eval_results.json trial3/eval_results.json \
        --output-dir comparison/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_eval_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DeepSynth parity scores (original vs Harbor)"
    )
    parser.add_argument(
        "--original-results",
        type=Path,
        nargs="+",
        required=True,
        help="eval_results.json files from original-side trials",
    )
    parser.add_argument(
        "--harbor-results",
        type=Path,
        nargs="+",
        required=True,
        help="eval_results.json files from Harbor-side trials",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Maximum allowed difference in mean F1 for pass (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write comparison.json (optional)",
    )
    args = parser.parse_args()

    # Load original-side results
    original_f1s = []
    for path in args.original_results:
        results = _load_eval_results(path)
        original_f1s.append(results["mean_f1"])
    original_mean = _mean(original_f1s)
    original_std = _std(original_f1s)

    # Load Harbor-side results
    harbor_f1s = []
    for path in args.harbor_results:
        results = _load_eval_results(path)
        harbor_f1s.append(results["mean_f1"])
    harbor_mean = _mean(harbor_f1s)
    harbor_std = _std(harbor_f1s)

    # Compare
    diff = abs(original_mean - harbor_mean)
    passed = diff <= args.threshold

    # Print comparison table
    print()
    print("=" * 65)
    print("DeepSynth Parity Comparison")
    print("=" * 65)
    print(f"{'':>20} {'Original':>18} {'Harbor':>18}")
    print(f"{'-' * 20} {'-' * 18} {'-' * 18}")
    print(f"{'Mean F1':>20} {original_mean:>14.4f}     {harbor_mean:>14.4f}")
    print(f"{'Std F1':>20} {original_std:>14.4f}     {harbor_std:>14.4f}")
    print(f"{'Trials':>20} {len(original_f1s):>14d}     {len(harbor_f1s):>14d}")
    print(f"{'-' * 20} {'-' * 18} {'-' * 18}")
    print(f"{'Original trials':>20} {[round(x, 4) for x in original_f1s]}")
    print(f"{'Harbor trials':>20} {[round(x, 4) for x in harbor_f1s]}")
    print()
    print(f"Absolute difference: {diff:.4f}")
    print(f"Threshold:           {args.threshold:.4f}")
    print(f"Result:              {'PASS' if passed else 'FAIL'}")
    print("=" * 65)

    comparison = {
        "metric": "Mean F1",
        "original": f"{original_mean:.4f} +/- {original_std:.4f}",
        "harbor": f"{harbor_mean:.4f} +/- {harbor_std:.4f}",
        "original_trials": [round(x, 4) for x in original_f1s],
        "harbor_trials": [round(x, 4) for x in harbor_f1s],
        "abs_difference": round(diff, 4),
        "threshold": args.threshold,
        "passed": passed,
    }

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "comparison.json"
        out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
