#!/usr/bin/env python3
"""Run Claude Code CLI directly on DeepSynth questions (original-side parity runner).

Runs each question outside Harbor/Docker, collects JSON answers into
predictions.json in the original benchmark format ({"001": {...}, "002": {...}}).

Supports concurrent execution via --concurrency N (default: 1 = sequential).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
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


async def _run_claude(
    prompt: str, model: str, claude_bin: str, timeout_sec: float | None = None,
) -> tuple[str, dict]:
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
    # Only strip the nesting-detection env var; keep everything else
    # (META_CLAUDE_API_KEY_HELPER_*, OTEL_*, etc. are needed by the Meta build)
    env.pop("CLAUDECODE", None)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(
            f"claude CLI timed out after {timeout_sec:.0f}s"
        )

    stdout = stdout_bytes.decode()
    stderr = stderr_bytes.decode()

    raw_output = stdout.strip()

    if proc.returncode != 0:
        if raw_output:
            # Hook failures may cause rc=1 but stdout still has the model's response
            pass
        else:
            raise RuntimeError(
                f"claude CLI failed (rc={proc.returncode}): {stderr}"
            )
    json_str = _extract_json(raw_output)

    trace = {
        "return_code": proc.returncode,
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
        "raw_output": raw_output,
        "extracted_json": json_str,
    }

    return json_str, trace


async def _process_question(
    idx: int,
    total: int,
    item: dict,
    template: str,
    model: str,
    claude_bin: str,
    log_path: Path,
    semaphore: asyncio.Semaphore,
    log_lock: asyncio.Lock,
    timeout_sec: float | None = None,
) -> tuple[str, object]:
    """Process a single question, respecting the concurrency semaphore."""
    qnum = str(item.get("Question Number", idx + 1)).strip()
    question = item["Questions"]

    async with semaphore:
        print(f"[started]  {idx + 1}/{total} Q{qnum}")
        prompt = template.format(question=question)

        start = time.perf_counter()
        try:
            json_str, trace = await _run_claude(
                prompt, model, claude_bin, timeout_sec=timeout_sec,
            )
        except TimeoutError:
            elapsed = time.perf_counter() - start
            print(f"[timeout]  {idx + 1}/{total} Q{qnum} ({elapsed:.1f}s)")
            trace = {"error": "timeout", "timeout_sec": timeout_sec}
            json_str = None
        except RuntimeError as e:
            elapsed = time.perf_counter() - start
            print(f"[error]    {idx + 1}/{total} Q{qnum} ({elapsed:.1f}s): {e}")
            trace = {"error": str(e)}
            json_str = None
        else:
            elapsed = time.perf_counter() - start

        if json_str is not None:
            try:
                answer = json.loads(json_str)
            except json.JSONDecodeError:
                print(
                    f"  WARNING: Could not parse JSON for Q{qnum}, storing raw string"
                )
                answer = json_str
            print(f"[done]     {idx + 1}/{total} Q{qnum} ({elapsed:.1f}s)")
        else:
            answer = None

        log_entry = {
            "index": idx,
            "question_number": qnum,
            "model": model,
            "elapsed_sec": round(elapsed, 3),
            "prompt_chars": len(prompt),
            "trace": trace,
        }
        async with log_lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry) + "\n")

    return qnum, answer


async def _async_main(args: argparse.Namespace) -> None:
    template = args.prompt_template.read_text(encoding="utf-8")
    data = json.loads(args.dataset_json.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "claude_log.jsonl")

    semaphore = asyncio.Semaphore(args.concurrency)
    log_lock = asyncio.Lock()
    total = len(data)

    timeout = args.timeout if args.timeout > 0 else None
    timeout_str = f"{timeout:.0f}s" if timeout else "none"
    print(f"Running {total} questions with concurrency={args.concurrency}, timeout={timeout_str}")

    tasks = [
        _process_question(
            idx, total, item, template, args.model, args.claude_bin,
            log_path, semaphore, log_lock, timeout_sec=timeout,
        )
        for idx, item in enumerate(data)
    ]

    results = await asyncio.gather(*tasks)

    # Sort by question number to maintain deterministic output order
    predictions = {qnum: answer for qnum, answer in sorted(results, key=lambda r: r[0])}

    out_path = args.output_dir / "predictions.json"
    out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(predictions)} predictions)")


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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent Claude CLI invocations (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200,
        help="Timeout in seconds per question (default: 1200 = 20min, 0 = no timeout)",
    )
    args = parser.parse_args()

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
