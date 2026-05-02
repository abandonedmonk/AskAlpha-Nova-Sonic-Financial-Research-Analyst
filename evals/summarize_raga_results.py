"""
Summarize RAG evaluation JSON reports.

Usage:
  python -m evals.summarize_raga_results evals/raga_results_20260321_093927.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = [
    "answer_relevance",
    "faithfulness",
    "context_relevance",
    "correctness",
]


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("results", [])

    metric_values: dict[str, list[float]] = {m: [] for m in METRICS}
    by_category: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m in METRICS}
    )

    missing_context = 0
    no_passages = 0
    hard_fail_correctness = 0

    for row in rows:
        scores = row.get("scores", {})
        category = row.get("category", "Unknown")

        if int(row.get("retrieved_passages", 0)) == 0:
            no_passages += 1

        if float(scores.get("context_relevance", 0.0)) == 0.0:
            missing_context += 1

        if float(scores.get("correctness", 0.0)) == 0.0:
            hard_fail_correctness += 1

        for metric in METRICS:
            value = float(scores.get(metric, 0.0))
            metric_values[metric].append(value)
            by_category[category][metric].append(value)

    overall = {m: _safe_mean(metric_values[m]) for m in METRICS}
    overall["overall"] = _safe_mean(list(overall.values()))

    category_summary: dict[str, dict[str, float]] = {}
    for category, values in by_category.items():
        cat_metrics = {m: _safe_mean(values[m]) for m in METRICS}
        cat_metrics["overall"] = _safe_mean(list(cat_metrics.values()))
        category_summary[category] = cat_metrics

    total = len(rows)
    quality = {
        "total_questions": total,
        "no_passage_count": no_passages,
        "no_passage_pct": (no_passages / total * 100.0) if total else 0.0,
        "zero_context_relevance_count": missing_context,
        "zero_context_relevance_pct": (
            (missing_context / total * 100.0) if total else 0.0
        ),
        "zero_correctness_count": hard_fail_correctness,
        "zero_correctness_pct": (
            (hard_fail_correctness / total * 100.0) if total else 0.0
        ),
    }

    return {
        "meta": report.get("meta", {}),
        "overall": overall,
        "by_category": category_summary,
        "quality_flags": quality,
    }


def print_summary(summary: dict[str, Any]) -> None:
    meta = summary["meta"]
    overall = summary["overall"]
    quality = summary["quality_flags"]

    print("RAG Eval Summary")
    print("=" * 72)
    print(f"Dataset: {meta.get('dataset', 'unknown')}")
    print(f"Company: {meta.get('company', 'unknown')}")
    print(f"Model:   {meta.get('groq_model', 'unknown')}")
    print(f"Total:   {quality['total_questions']}")
    print()

    print("Overall Metrics")
    for metric in METRICS + ["overall"]:
        print(f"  {metric:20s} {overall[metric]:.4f}")
    print()

    print("By Category")
    for category, metrics in sorted(summary["by_category"].items()):
        print(f"  - {category}")
        for metric in METRICS + ["overall"]:
            print(f"      {metric:18s} {metrics[metric]:.4f}")
    print()

    print("Quality Flags")
    print(
        "  no_passage_count: "
        f"{quality['no_passage_count']} ({quality['no_passage_pct']:.1f}%)"
    )
    print(
        "  zero_context_relevance_count: "
        f"{quality['zero_context_relevance_count']} "
        f"({quality['zero_context_relevance_pct']:.1f}%)"
    )
    print(
        "  zero_correctness_count: "
        f"{quality['zero_correctness_count']} ({quality['zero_correctness_pct']:.1f}%)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RAG evaluation output JSON")
    parser.add_argument("report", type=Path, help="Path to raga_results_*.json")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save computed summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.report.exists():
        raise FileNotFoundError(f"Report not found: {args.report}")

    report = _load(args.report)
    summary = summarize_report(report)
    print_summary(summary)

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"Saved summary JSON: {args.save_json}")


if __name__ == "__main__":
    main()
