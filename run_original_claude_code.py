#!/usr/bin/env python3
"""Run Claude Code CLI directly on DeepSynth questions (original-side parity runner).

Runs each question outside Harbor/Docker, collects JSON answers into
predictions.json in the original benchmark format ({"001": {...}, "002": {...}}).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from text."""
    stripped = text.strip()
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()
    return stripped


def _extract_json(text: str) -> str:
    """Extract JSON dict from agent output, handling fences and extra text."""
    stripped = _strip_fences(text)

    # Try to find a JSON object in the text
    brace_start = stripped.find("{")
    if brace_start == -1:
        return stripped

    # Find matching closing brace
    depth = 0
    for i in range(brace_start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                return stripped[brace_start : i + 1]

    # If no matching brace found, return from first brace onwards
    return stripped[brace_start:]


def _run_claude(prompt: str, model: str, claude_bin: str) -> tuple[str, dict]:
    """Run claude CLI and return (raw_output, trace_dict)."""
    cmd = [
        claude_bin,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "text",
        "--dangerously-skip-permissions",
    ]
    env = os.environ.copy()

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (rc={proc.returncode}): {proc.stderr or proc.stdout}"
        )

    raw_output = proc.stdout.strip()
    json_str = _extract_json(raw_output)

    trace = {
        "return_code": proc.returncode,
        "stdout_chars": len(proc.stdout),
        "stderr_chars": len(proc.stderr),
        "raw_output": raw_output[:2000],
        "extracted_json": json_str[:2000],
    }

    return json_str, trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Claude Code CLI on DeepSynth questions (original-side)"
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="DeepSynth dev.json (list of records with 'Question Number', "
        "'Questions', 'Answers')",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        required=True,
        help="Prompt template file (must contain {question} placeholder)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--claude-bin", type=str, default="claude", help="Claude CLI binary path"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to write JSONL logs (default: <output-dir>/claude_log.jsonl)",
    )
    args = parser.parse_args()

    template = args.prompt_template.read_text(encoding="utf-8")
    data = json.loads(args.dataset_json.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    predictions: OrderedDict[str, object] = OrderedDict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "claude_log.jsonl")

    for idx, item in enumerate(data):
        qnum = str(item.get("Question Number", idx + 1)).strip()
        question = item["Questions"]
        print(f"Processing {idx + 1}/{len(data)} (Q{qnum})")

        prompt = template.format(question=question)

        start = time.perf_counter()
        json_str, trace = _run_claude(prompt, args.model, args.claude_bin)
        elapsed = time.perf_counter() - start

        # Try to parse the JSON answer
        try:
            answer = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"  WARNING: Could not parse JSON for Q{qnum}, storing raw string")
            answer = json_str

        predictions[qnum] = answer
        print(f"  Q{qnum} done ({elapsed:.1f}s)")

        log_entry = {
            "index": idx,
            "question_number": qnum,
            "model": args.model,
            "elapsed_sec": round(elapsed, 3),
            "prompt_chars": len(prompt),
            "trace": trace,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")

    out_path = args.output_dir / "predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(predictions)} predictions)")


if __name__ == "__main__":
    main()
