#!/usr/bin/env python3
"""Export Harbor job predictions to DeepSynth original benchmark format.

Iterates trial directories in a Harbor job, reads /app/answer.json from
collected artifacts, and outputs predictions.json in the original format
({"001": {...}, "002": {...}}).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_task_toml(path: Path) -> dict:
    """Minimal TOML parser for extracting key-value pairs from task.toml."""
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("[") or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        data[key] = value
    return data


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from text that may contain markdown fences or extra output."""
    text = text.strip()

    # Strip markdown code fences
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # Find first { to last matching }
    brace_start = text.find("{")
    if brace_start == -1:
        return text

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]

    return text[brace_start:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Harbor DeepSynth predictions to original format"
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        required=True,
        help="Harbor job directory (jobs/<job_id>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write predictions.json",
    )
    args = parser.parse_args()

    predictions: dict[str, object] = {}
    found = 0
    missing = []

    for task_dir in sorted(args.jobs_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        config_path = task_dir / "config.json"
        if not config_path.exists():
            continue

        # Get source_id from task.toml
        config = _load_json(config_path)
        task_path = config.get("task", {}).get("path")
        if not task_path:
            continue

        task_toml = Path(task_path) / "task.toml"
        if not task_toml.exists():
            continue

        task_meta = _parse_task_toml(task_toml)
        source_id = task_meta.get("source_id")
        if not source_id:
            # Fall back to task directory name (question number)
            source_id = Path(task_path).name

        # Try to read answer.json from collected artifacts
        answer_path = task_dir / "artifacts" / "answer.json"
        if not answer_path.exists():
            # Fallback: check verifier directory
            answer_path = task_dir / "verifier" / "answer.json"
        if not answer_path.exists():
            # Fallback: try to parse from test stdout
            test_stdout = task_dir / "verifier" / "test-stdout.txt"
            if test_stdout.exists():
                stdout_text = test_stdout.read_text(encoding="utf-8")
                # Look for "Model answer:" line in test output
                match = re.search(r"Model answer:\s*(.+)", stdout_text, re.DOTALL)
                if match:
                    json_str = _extract_json_from_text(match.group(1))
                    try:
                        answer = json.loads(json_str)
                        predictions[str(source_id)] = answer
                        found += 1
                        continue
                    except json.JSONDecodeError:
                        pass
            missing.append(source_id)
            continue

        raw = answer_path.read_text(encoding="utf-8").strip()
        json_str = _extract_json_from_text(raw)
        try:
            answer = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse JSON for source_id={source_id}")
            answer = raw

        predictions[str(source_id)] = answer
        found += 1

    # Sort by question number for consistent output
    ordered: OrderedDict[str, object] = OrderedDict()
    for key in sorted(predictions.keys(), key=lambda x: x.zfill(10)):
        ordered[key] = predictions[key]

    if missing:
        print(f"Missing predictions for source_ids: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "predictions.json"
    out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({found} predictions, {len(missing)} missing)")


if __name__ == "__main__":
    main()
