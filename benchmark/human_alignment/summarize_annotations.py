#!/usr/bin/env python3
"""Summarize pairwise human preferences against Step 3 LLM-as-judge preferences."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.human_alignment.common import (
    METRIC_BY_CODE,
    METRIC_CODES,
    normalize_preference,
    read_json,
    write_json,
)


def _load_key(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    items = data.get("items", data if isinstance(data, list) else [])
    return {row["pair_id"]: row for row in items}


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_from_summary(metrics: dict[str, Any], metric_key: str) -> float | None:
    sf = metrics.get("source_faithfulness", {}) or {}
    tq = metrics.get("teaching_quality", {}) or {}
    pq = metrics.get("practice_questions", {}) or {}
    if "summary" in pq and isinstance(pq["summary"], dict):
        pq = pq["summary"]

    values = {
        "source_faithfulness": sf.get("avg_score_overall", sf.get("avg_score")),
        "personalization": tq.get(
            "avg_personalization_overall",
            (tq.get("personalization", {}) or {}).get("avg"),
        ),
        "applicability": tq.get(
            "avg_applicability_overall",
            (tq.get("applicability", {}) or {}).get("avg"),
        ),
        "vividness": tq.get(
            "avg_vividness_overall",
            (tq.get("vividness", {}) or {}).get("avg"),
        ),
        "logical_depth": tq.get(
            "avg_logical_depth_overall",
            (tq.get("logical_depth", {}) or {}).get("avg"),
        ),
        "pq_fitness": pq.get("avg_fitness"),
        "pq_groundedness": pq.get("avg_groundedness"),
        "pq_diversity": pq.get("avg_diversity"),
        "pq_answer_quality": pq.get("avg_answer_quality"),
        "pq_cross_concept": pq.get("avg_cross_concept"),
    }
    value = values.get(metric_key)
    return float(value) if isinstance(value, (int, float)) else None


def _load_session_metrics(eval_path: Path, entry_id: str, session_index: int) -> dict[str, Any]:
    if not eval_path.exists():
        return {}
    eval_data = read_json(eval_path)
    if isinstance(eval_data.get("sessions"), list):
        session = None
        for candidate in eval_data["sessions"]:
            if entry_id and candidate.get("entry_id") == entry_id:
                session = candidate
                break
        if session is None and 1 <= session_index <= len(eval_data["sessions"]):
            session = eval_data["sessions"][session_index - 1]
        return session.get("metrics", {}) if session else {}
    return eval_data.get("metrics", {})


def _scores_for_side(key: dict[str, Any], side: str) -> dict[str, float | None]:
    eval_path = Path(key[f"system_{side.lower()}_evaluation_path"])
    metrics = _load_session_metrics(
        eval_path=eval_path,
        entry_id=str(key.get("entry_id", "")),
        session_index=int(key.get("session_index") or 1),
    )
    return {
        code: _metric_from_summary(metrics, METRIC_BY_CODE[code]["key"])
        for code in METRIC_CODES
    }


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    counts = Counter(values)
    top_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return "tie"
    return winners[0]


def _side_to_backend_pref(side_pref: str | None, key: dict[str, Any]) -> str | None:
    if side_pref is None:
        return None
    if side_pref == "tie":
        return "tie"
    backend = key["system_a_backend"] if side_pref == "A" else key["system_b_backend"]
    if backend == key["target_backend"]:
        return "target"
    if backend == key["baseline_backend"]:
        return "baseline"
    return backend


def _llm_side_preference(score_a: float | None, score_b: float | None, threshold: float) -> str | None:
    if score_a is None or score_b is None:
        return None
    delta = score_a - score_b
    if abs(delta) <= threshold:
        return "tie"
    return "A" if delta > 0 else "B"


def _cohen_kappa(xs: list[str], ys: list[str], labels: list[str]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    n = len(xs)
    observed = sum(1 for x, y in zip(xs, ys, strict=True) if x == y) / n
    x_counts = Counter(xs)
    y_counts = Counter(ys)
    expected = sum((x_counts[label] / n) * (y_counts[label] / n) for label in labels)
    if math.isclose(1.0 - expected, 0.0):
        return None
    return round((observed - expected) / (1.0 - expected), 4)


def _rate(values: list[str], label: str) -> float | None:
    return round(sum(1 for v in values if v == label) / len(values), 4) if values else None


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pairwise Human Alignment Summary",
        "",
        f"- Generated at: {summary['timestamp']}",
        f"- Annotation rows: {summary['num_annotation_rows']}",
        f"- Pairs with human labels: {summary['num_pairs_with_human_labels']}",
        f"- Raters: {summary['num_raters']}",
        f"- Tie threshold: {summary['tie_threshold']}",
        "",
        "| Metric | N | Human prefers target | LLM prefers target | Agreement | Kappa | Human tie | LLM tie |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for code in METRIC_CODES:
        metric = summary["metrics"].get(code, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    code,
                    str(metric.get("n", 0)),
                    _fmt_pct(metric.get("human_target_preference_rate")),
                    _fmt_pct(metric.get("llm_target_preference_rate")),
                    _fmt_pct(metric.get("agreement_rate")),
                    _fmt(metric.get("cohen_kappa")),
                    _fmt_pct(metric.get("human_tie_rate")),
                    _fmt_pct(metric.get("llm_tie_rate")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return "-"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{100 * float(value):.2f}%"
    return "-"


def summarize_annotations(
    *,
    annotations_path: Path,
    key_path: Path,
    output_path: Path,
    tie_threshold: float = 0.25,
    live_llm_preferences: dict[str, dict[str, str]] | None = None,
    judge_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key_by_pair = _load_key(key_path)
    rows = _load_annotations(annotations_path)
    raters = set()
    human_votes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    rater_votes: dict[str, dict[str, dict[str, str]]] = {
        code: defaultdict(dict) for code in METRIC_CODES
    }

    for row in rows:
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id not in key_by_pair:
            continue
        rater_id = str(row.get("rater_id", "")).strip()
        if rater_id:
            raters.add(rater_id)
        for code in METRIC_CODES:
            pref = normalize_preference(row.get(code))
            if pref is None:
                continue
            human_votes[pair_id][code].append(pref)
            if rater_id:
                rater_votes[code][rater_id][pair_id] = _side_to_backend_pref(pref, key_by_pair[pair_id]) or pref

    pair_records = []
    for pair_id in sorted(human_votes.keys()):
        key = key_by_pair[pair_id]
        scores_a = _scores_for_side(key, "A")
        scores_b = _scores_for_side(key, "B")
        metric_records = {}
        for code in METRIC_CODES:
            human_side = _majority(human_votes[pair_id].get(code, []))
            human_backend = _side_to_backend_pref(human_side, key)
            live_backend = (live_llm_preferences or {}).get(pair_id, {}).get(code)
            if live_backend in {"target", "baseline", "tie"}:
                llm_side = None
                llm_backend = live_backend
            else:
                llm_side = _llm_side_preference(scores_a.get(code), scores_b.get(code), tie_threshold)
                llm_backend = _side_to_backend_pref(llm_side, key)
            metric_records[code] = {
                "human_side_preference": human_side,
                "human_backend_preference": human_backend,
                "llm_side_preference": llm_side,
                "llm_backend_preference": llm_backend,
                "score_a": scores_a.get(code),
                "score_b": scores_b.get(code),
            }
        pair_records.append(
            {
                "pair_id": pair_id,
                "kb_name": key.get("kb_name"),
                "profile_id": key.get("profile_id"),
                "entry_id": key.get("entry_id"),
                "session_index": key.get("session_index"),
                "system_a_backend": key.get("system_a_backend"),
                "system_b_backend": key.get("system_b_backend"),
                "target_backend": key.get("target_backend"),
                "baseline_backend": key.get("baseline_backend"),
                "metrics": metric_records,
            }
        )

    metric_summary = {}
    labels = ["target", "baseline", "tie"]
    for code in METRIC_CODES:
        human_labels = []
        llm_labels = []
        for rec in pair_records:
            metric = rec["metrics"][code]
            human = metric.get("human_backend_preference")
            llm = metric.get("llm_backend_preference")
            if human in labels and llm in labels:
                human_labels.append(human)
                llm_labels.append(llm)
        agreement = (
            round(
                sum(
                    1
                    for human_label, llm_label in zip(human_labels, llm_labels, strict=True)
                    if human_label == llm_label
                )
                / len(human_labels),
                4,
            )
            if human_labels
            else None
        )
        metric_summary[code] = {
            "label": METRIC_BY_CODE[code]["label"],
            "metric_key": METRIC_BY_CODE[code]["key"],
            "n": len(human_labels),
            "human_target_preference_rate": _rate(human_labels, "target"),
            "human_baseline_preference_rate": _rate(human_labels, "baseline"),
            "human_tie_rate": _rate(human_labels, "tie"),
            "llm_target_preference_rate": _rate(llm_labels, "target"),
            "llm_baseline_preference_rate": _rate(llm_labels, "baseline"),
            "llm_tie_rate": _rate(llm_labels, "tie"),
            "agreement_rate": agreement,
            "cohen_kappa": _cohen_kappa(human_labels, llm_labels, labels),
            "counts": {
                "human": dict(Counter(human_labels)),
                "llm": dict(Counter(llm_labels)),
            },
        }

    inter_rater = {}
    for code, by_rater in rater_votes.items():
        pair_stats = []
        rater_ids = sorted(by_rater.keys())
        for i, rater_a in enumerate(rater_ids):
            for rater_b in rater_ids[i + 1 :]:
                shared = sorted(set(by_rater[rater_a]) & set(by_rater[rater_b]))
                xs = [by_rater[rater_a][pair_id] for pair_id in shared]
                ys = [by_rater[rater_b][pair_id] for pair_id in shared]
                pair_stats.append(
                    {
                        "rater_a": rater_a,
                        "rater_b": rater_b,
                        "n": len(shared),
                        "agreement_rate": (
                            round(sum(1 for x, y in zip(xs, ys, strict=True) if x == y) / len(xs), 4)
                            if xs
                            else None
                        ),
                        "cohen_kappa": _cohen_kappa(xs, ys, labels),
                    }
                )
        valid_agreements = [
            float(p["agreement_rate"])
            for p in pair_stats
            if isinstance(p.get("agreement_rate"), (int, float))
        ]
        valid_kappas = [
            float(p["cohen_kappa"])
            for p in pair_stats
            if isinstance(p.get("cohen_kappa"), (int, float))
        ]
        inter_rater[code] = {
            "mean_pairwise_agreement": round(sum(valid_agreements) / len(valid_agreements), 4) if valid_agreements else None,
            "mean_pairwise_kappa": round(sum(valid_kappas) / len(valid_kappas), 4) if valid_kappas else None,
            "pairs": pair_stats,
        }

    summary = {
        "step": "human_alignment_summarize_pairwise_annotations",
        "timestamp": datetime.now().isoformat(),
        "annotations_path": str(annotations_path),
        "annotation_key_path": str(key_path),
        "tie_threshold": tie_threshold,
        "num_annotation_rows": len(rows),
        "num_pairs_with_human_labels": len(pair_records),
        "num_raters": len(raters),
        "llm_preference_source": "live_judge" if live_llm_preferences is not None else "step3_eval",
        "judge_metadata": judge_metadata or {},
        "metrics": metric_summary,
        "inter_rater": inter_rater,
        "pairs": pair_records,
    }
    write_json(output_path, summary)
    md_path = output_path.with_suffix(".md")
    md_path.write_text(_build_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pairwise human-vs-LLM alignment")
    parser.add_argument("--annotations", required=True, help="Completed CSV or JSONL annotations")
    parser.add_argument("--key", required=True, help="Private annotation_key.json from export")
    parser.add_argument("--tie-threshold", type=float, default=0.25, help="LLM score delta treated as tie")
    parser.add_argument(
        "--output",
        default="",
        help="Output summary JSON (default: sibling human_alignment_summary.json)",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    key_path = Path(args.key)
    output_path = Path(args.output) if args.output else key_path.parent / "human_alignment_summary.json"
    summary = summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=output_path,
        tie_threshold=args.tie_threshold,
    )
    print(f"Summary: {output_path}")
    print(f"Markdown: {output_path.with_suffix('.md')}")
    print(
        f"Rows: {summary['num_annotation_rows']} | "
        f"Pairs: {summary['num_pairs_with_human_labels']} | "
        f"Raters: {summary['num_raters']}"
    )


if __name__ == "__main__":
    main()
