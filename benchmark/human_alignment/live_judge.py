#!/usr/bin/env python3
"""Live LLM pairwise judge for human-alignment annotations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.data_generation.llm_utils import call_llm_json
from benchmark.human_alignment.common import METRIC_BY_CODE, METRIC_CODES, read_json, write_json
from benchmark.human_alignment.summarize_annotations import summarize_annotations

DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
SYSTEM_PROMPT = """You are an expert blind evaluator for TutorBench.

You compare System A and System B for the same student profile, task, source excerpts,
dialog, and practice questions. For each metric, choose exactly one of:

- A: System A is better.
- B: System B is better.
- tie: the two systems are comparable or evidence is insufficient.

Use the metric definitions exactly. Do not infer backend identity. Return only valid JSON.
"""


def _load_package(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows[str(item["pair_id"])] = item
    return rows


def _load_key(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    items = data.get("items", data if isinstance(data, list) else [])
    return {str(item["pair_id"]): item for item in items}


def _load_annotated_pair_ids(path: Path) -> list[str]:
    import csv

    ids: set[str] = set()
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    pair_id = str(row.get("pair_id", "")).strip()
                    if pair_id:
                        ids.add(pair_id)
        return sorted(ids)

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pair_id = str(row.get("pair_id", "")).strip()
            if pair_id:
                ids.add(pair_id)
    return sorted(ids)


def _short(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _format_dialog(dialog: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for idx, msg in enumerate(dialog, start=1):
        role = str(msg.get("role", "")).upper()
        content = _short(msg.get("content", ""), max_chars)
        blocks.append(f"{idx}. {role}\n{content}")
    return "\n\n".join(blocks) or "(none)"


def _format_questions(questions: list[Any], max_chars: int) -> str:
    blocks = []
    for idx, question in enumerate(questions, start=1):
        blocks.append(f"Q{idx}. {_short(question, max_chars)}")
    return "\n\n".join(blocks) or "(none)"


def _format_gaps(gaps: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    for gap in gaps:
        source = "\n".join(
            f"Page {page}: {_short(text, max_chars // 2)}"
            for page, text in (gap.get("source_excerpts") or {}).items()
        )
        blocks.append(
            "\n".join(
                [
                    f"Gap: {gap.get('gap_id', '')} {gap.get('target_concept', '')}",
                    f"Type: {gap.get('gap_type', '')}",
                    f"Description: {_short(gap.get('description', ''), max_chars)}",
                    f"Correct understanding: {_short(gap.get('correct_understanding', ''), max_chars)}",
                    f"Source excerpts:\n{source or '(none)'}",
                ]
            )
        )
    return "\n\n".join(blocks) or "(none)"


def _judge_prompt(item: dict[str, Any]) -> str:
    profile = item.get("profile", {}) or {}
    task = item.get("task", {}) or {}
    system_a = item.get("system_a", {}) or {}
    system_b = item.get("system_b", {}) or {}
    metrics = "\n".join(
        f"- {code}: {METRIC_BY_CODE[code]['label']}"
        for code in METRIC_CODES
    )
    return f"""Compare System A and System B for this pair.

Return JSON with exactly this shape:
{{
  "preferences": {{
    "SF": "A|B|tie",
    "PER": "A|B|tie",
    "APP": "A|B|tie",
    "VID": "A|B|tie",
    "LD": "A|B|tie",
    "FIT": "A|B|tie",
    "GND": "A|B|tie",
    "DIV": "A|B|tie",
    "ANS": "A|B|tie",
    "CC": "A|B|tie"
  }},
  "rationale": {{
    "SF": "brief reason",
    "PER": "brief reason",
    "APP": "brief reason",
    "VID": "brief reason",
    "LD": "brief reason",
    "FIT": "brief reason",
    "GND": "brief reason",
    "DIV": "brief reason",
    "ANS": "brief reason",
    "CC": "brief reason"
  }}
}}

Metric definitions:
{metrics}

Shared profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Gaps and source excerpts:
{_format_gaps(item.get("gaps", []) or [], 1200)}

SYSTEM A DIALOG:
{_format_dialog(system_a.get("dialog", []) or [], 1800)}

SYSTEM A PRACTICE QUESTIONS:
{_format_questions(system_a.get("practice_questions", []) or [], 1600)}

SYSTEM B DIALOG:
{_format_dialog(system_b.get("dialog", []) or [], 1800)}

SYSTEM B PRACTICE QUESTIONS:
{_format_questions(system_b.get("practice_questions", []) or [], 1600)}
"""


def _side_to_backend_pref(side_pref: str, key: dict[str, Any]) -> str | None:
    value = str(side_pref or "").strip().lower()
    if value == "tie":
        return "tie"
    if value == "a":
        backend = key.get("system_a_backend")
    elif value == "b":
        backend = key.get("system_b_backend")
    else:
        return None
    if backend == key.get("target_backend"):
        return "target"
    if backend == key.get("baseline_backend"):
        return "baseline"
    return None


async def _judge_one(
    *,
    pair_id: str,
    item: dict[str, Any],
    key: dict[str, Any],
    model: str,
    binding: str | None,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        response = await call_llm_json(
            user_prompt=_judge_prompt(item),
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            **{k: v for k, v in {"binding": binding, "base_url": base_url, "api_key": api_key}.items() if v},
        )
    preferences = response.get("preferences", {}) or {}
    backend_preferences: dict[str, str] = {}
    for code in METRIC_CODES:
        backend = _side_to_backend_pref(str(preferences.get(code, "")), key)
        if backend in {"target", "baseline", "tie"}:
            backend_preferences[code] = backend
    return {
        "pair_id": pair_id,
        "model": model,
        "binding": binding,
        "side_preferences": {code: str(preferences.get(code, "")) for code in METRIC_CODES},
        "backend_preferences": backend_preferences,
        "rationale": response.get("rationale", {}),
        "raw_response": response,
    }


async def run_live_judge(
    *,
    annotations_path: Path,
    key_path: Path,
    package_path: Path,
    output_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    binding: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    limit_pairs: int = 0,
) -> dict[str, Any]:
    key_by_pair = _load_key(key_path)
    package_by_pair = _load_package(package_path)
    pair_ids = [
        pair_id
        for pair_id in _load_annotated_pair_ids(annotations_path)
        if pair_id in key_by_pair and pair_id in package_by_pair
    ]
    if limit_pairs > 0:
        pair_ids = pair_ids[:limit_pairs]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        _judge_one(
            pair_id=pair_id,
            item=package_by_pair[pair_id],
            key=key_by_pair[pair_id],
            model=model,
            binding=binding,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            semaphore=semaphore,
        )
        for pair_id in pair_ids
    ]
    items = await asyncio.gather(*tasks)
    result = {
        "step": "human_alignment_live_llm_judge",
        "annotations_path": str(annotations_path),
        "annotation_key_path": str(key_path),
        "annotation_package_path": str(package_path),
        "model": model,
        "binding": binding or "",
        "base_url": base_url or "",
        "num_pairs_judged": len(items),
        "items": items,
    }
    write_json(output_path, result)
    return result


def live_preferences_by_pair(judge_result: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        str(item["pair_id"]): dict(item.get("backend_preferences", {}) or {})
        for item in judge_result.get("items", [])
    }


def summarize_with_live_judge(
    *,
    annotations_path: Path,
    key_path: Path,
    package_path: Path,
    summary_output_path: Path,
    judge_output_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    binding: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    concurrency: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    limit_pairs: int = 0,
) -> dict[str, Any]:
    judge_result = asyncio.run(
        run_live_judge(
            annotations_path=annotations_path,
            key_path=key_path,
            package_path=package_path,
            output_path=judge_output_path,
            model=model,
            binding=binding,
            base_url=base_url,
            api_key=api_key,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=max_tokens,
            limit_pairs=limit_pairs,
        )
    )
    return summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=summary_output_path,
        live_llm_preferences=live_preferences_by_pair(judge_result),
        judge_metadata={
            "judge_output_path": str(judge_output_path),
            "model": model,
            "binding": binding or "",
            "base_url": base_url or "",
            "num_pairs_judged": judge_result.get("num_pairs_judged", 0),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Claude pairwise judge for human alignment")
    parser.add_argument("--annotations", required=True, help="Completed annotation CSV/JSONL")
    parser.add_argument("--key", required=True, help="annotation_key.json")
    parser.add_argument("--package", default="", help="annotation_package.jsonl (default: next to key)")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model")
    parser.add_argument("--binding", default="", help="Override LLM provider binding; default uses existing LLM config")
    parser.add_argument("--base-url", default="", help="Override judge API base URL; default uses existing LLM config")
    parser.add_argument("--api-key", default=None, help="Judge API key (default: provider env var)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent judge calls")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max tokens per judge response")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Debug: judge only first N annotated pairs")
    parser.add_argument("--judge-output", default="", help="Live judge JSON output")
    parser.add_argument("--summary-output", default="", help="Summary JSON output")
    args = parser.parse_args()

    key_path = Path(args.key)
    package_path = Path(args.package) if args.package else key_path.parent / "annotation_package.jsonl"
    judge_output_path = Path(args.judge_output) if args.judge_output else key_path.parent / "live_llm_judgments.json"
    summary_output_path = Path(args.summary_output) if args.summary_output else key_path.parent / "human_alignment_summary.json"
    summarize_with_live_judge(
        annotations_path=Path(args.annotations),
        key_path=key_path,
        package_path=package_path,
        summary_output_path=summary_output_path,
        judge_output_path=judge_output_path,
        model=args.model,
        binding=args.binding or None,
        base_url=args.base_url or None,
        api_key=args.api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        limit_pairs=args.limit_pairs,
    )
    print(f"Live judge: {judge_output_path}")
    print(f"Summary: {summary_output_path}")
    print(f"Markdown: {summary_output_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
