from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = (
    PROJECT_ROOT / "evaluation" / "predictions.jsonl"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT / "evaluation" / "recall_analysis.json"
)

OUTPUT_ERROR_PATH = (
    PROJECT_ROOT / "evaluation" / "error_analysis.jsonl"
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from services.absa.data.loader import load_jsonl, save_jsonl

def normalize_aspect(aspect: str) -> str:
    return " ".join(
        str(aspect)
        .strip()
        .lower()
        .split()
    )

def extract_aspects(
        record: dict[str, Any],
        field: str,
) -> set[str]:
    
    container = record.get(field) or {}

    if not isinstance(container, dict):
        return set()
    
    aspects = container.get("aspects") or []

    if not isinstance(aspects, list):
        return set()
    
    aspects = container.get("aspects") or []

    if not isinstance(aspects, list):
        return set()
    
    result = set()

    for item in aspects:
        if aspects is None:
            continue
        
        aspect = item.get("aspect")

        if aspect is None:
            continue

        result.add(
            normalize_aspect(aspect)
        )

    return result


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def analyze_recall(
        predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_jsonl(predictions_path)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    aspect_gold_count = Counter()
    aspect_pred_count = Counter()

    aspect_tp = Counter()
    aspect_fn = Counter()
    aspect_fp = Counter()

    review_errors = []

    # Recall grouped by number of gold aspects
    count_bucket_stats = defaultdict(
        lambda: {
            "reviews": 0,
            "gold_aspects": 0,
            "matched_aspects": 0,
            "missed_aspects": 0
        }
    )

    # Review level analysis

    for record in records:
        review_id = record.get("review_id")

        gold = extract_aspects(
            record,
            "reference"
        )

        predicted = extract_aspects(
            record,
            "prediction"
        )

        matched = gold & predicted
        missing = gold - predicted
        extra = predicted - gold

        tp = len(matched)
        fn = len(missing)
        fp = len(extra)

        total_tp += tp
        total_fn += fn
        total_fp += fp

        # Aspect-level counters
        for aspect in gold:
            aspect_gold_count[aspect] += 1

        for aspect in predicted:
            aspect_pred_count[aspect] += 1

        for aspect in matched:
            aspect_tp[aspect] += 1

        for aspect in missing:
            aspect_fn[aspect] += 1

        for aspect in extra:
            aspect_fp[aspect] += 1

        # Review-level count analysis
        gold_count = len(gold)

        count_bucket_stats[gold_count][
            "reviews"
        ] += 1

        count_bucket_stats[gold_count][
            "gold_aspects"
        ] += len(gold)

        count_bucket_stats[gold_count][
            "matched_aspects"
        ] += len(matched)

        count_bucket_stats[gold_count][
            "missed_aspects"
        ] += len(missing)

        # Keep reviews that actually have errors
        if missing or extra:

            review_errors.append(
                {
                    "review_id": review_id,
                    "gold_aspects": sorted(gold),
                    "predicted_aspects": sorted(predicted),
                    "matched_aspects": sorted(matched),
                    "missing_aspects": sorted(missing),
                    "extra_aspects": sorted(extra),
                    "gold_count": len(gold),
                    "predicted_count": len(predicted),
                    "matched_count": len(matched),
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                }
            )
        # -----------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Per-aspect metrics
    # -----------------------------------------------------

    all_aspects = (
        set(aspect_gold_count)
        | set(aspect_pred_count)
    )

    per_aspect = []

    for aspect in all_aspects:

        tp = aspect_tp[aspect]
        fp = aspect_fp[aspect]
        fn = aspect_fn[aspect]

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_aspect.append(
            {
                "aspect": aspect,
                "gold_count": aspect_gold_count[aspect],
                "predicted_count": aspect_pred_count[aspect],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Most frequently occurring aspects first
    per_aspect.sort(
        key=lambda x: (
            -x["gold_count"],
            -x["fn"],
        )
    )

    # -----------------------------------------------------
    # Worst recall aspects
    # -----------------------------------------------------

    worst_recall = sorted(
        per_aspect,
        key=lambda x: (
            x["recall"],
            -x["gold_count"],
        ),
    )

    # Only aspects that actually occur in gold
    worst_recall = [
        item
        for item in worst_recall
        if item["gold_count"] > 0
    ]

    # -----------------------------------------------------
    # Most missed aspects
    # -----------------------------------------------------

    most_missed = sorted(
        [
            {
                "aspect": aspect,
                "gold_count": aspect_gold_count[aspect],
                "missed_count": aspect_fn[aspect],
                "recall": (
                    aspect_tp[aspect]
                    / (
                        aspect_tp[aspect]
                        + aspect_fn[aspect]
                    )
                    if (
                        aspect_tp[aspect]
                        + aspect_fn[aspect]
                    ) > 0
                    else 0.0
                ),
            }
            for aspect in aspect_gold_count
        ],
        key=lambda x: (
            -x["missed_count"],
            -x["gold_count"],
        ),
    )

    # -----------------------------------------------------
    # Recall by review aspect count
    # -----------------------------------------------------

    recall_by_review_size = {}

    for count, stats in sorted(
        count_bucket_stats.items()
    ):

        gold_aspects = stats["gold_aspects"]
        matched_aspects = stats["matched_aspects"]

        recall = (
            matched_aspects / gold_aspects
            if gold_aspects > 0
            else 0.0
        )

        recall_by_review_size[str(count)] = {
            "reviews": stats["reviews"],
            "gold_aspects": gold_aspects,
            "matched_aspects": matched_aspects,
            "missed_aspects": stats["missed_aspects"],
            "recall": recall,
        }

    # -----------------------------------------------------
    # Worst reviews
    # -----------------------------------------------------

    worst_reviews = sorted(
        review_errors,
        key=lambda x: (
            -x["missing_count"],
            -x["gold_count"],
        ),
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    report = {
        "summary": {
            "test_reviews": len(records),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },

        "most_missed_aspects": most_missed,

        "worst_recall_aspects": worst_recall,

        "per_aspect": per_aspect,

        "recall_by_review_aspect_count": (
            recall_by_review_size
        ),

        "error_review_count": len(review_errors),
    }

    return report, worst_reviews


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    print(
        f"Loading predictions from:\n"
        f"{PREDICTIONS_PATH}"
    )

    report, review_errors = analyze_recall(
        PREDICTIONS_PATH
    )

    OUTPUT_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_jsonl(
        review_errors,
        OUTPUT_ERROR_PATH,
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    summary = report["summary"]

    print("\n" + "=" * 60)
    print("ASPECT RECALL ANALYSIS")
    print("=" * 60)

    print(
        f"Test reviews : {summary['test_reviews']}"
    )

    print(
        f"TP           : {summary['true_positive']}"
    )

    print(
        f"FP           : {summary['false_positive']}"
    )

    print(
        f"FN           : {summary['false_negative']}"
    )

    print(
        f"Precision    : {summary['precision']:.4f}"
    )

    print(
        f"Recall       : {summary['recall']:.4f}"
    )

    print(
        f"F1           : {summary['f1']:.4f}"
    )

    # -----------------------------------------------------
    # Most missed
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MOST MISSED ASPECTS")
    print("=" * 60)

    for item in report["most_missed_aspects"][:20]:

        print(
            f"{item['aspect']:<30} "
            f"gold={item['gold_count']:<4} "
            f"missed={item['missed_count']:<4} "
            f"recall={item['recall']:.3f}"
        )

    # -----------------------------------------------------
    # Worst recall
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("LOWEST RECALL ASPECTS")
    print("=" * 60)

    for item in report["worst_recall_aspects"][:20]:

        print(
            f"{item['aspect']:<30} "
            f"gold={item['gold_count']:<4} "
            f"TP={item['tp']:<4} "
            f"FN={item['fn']:<4} "
            f"recall={item['recall']:.3f}"
        )

    # -----------------------------------------------------
    # Review complexity
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("RECALL BY NUMBER OF GOLD ASPECTS")
    print("=" * 60)

    for (
        count,
        stats,
    ) in report[
        "recall_by_review_aspect_count"
    ].items():

        print(
            f"{count:>2} aspects/review | "
            f"reviews={stats['reviews']:<3} | "
            f"gold={stats['gold_aspects']:<4} | "
            f"missed={stats['missed_aspects']:<4} | "
            f"recall={stats['recall']:.3f}"
        )

    # -----------------------------------------------------
    # Worst reviews
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("REVIEWS WITH MOST MISSED ASPECTS")
    print("=" * 60)

    for review in review_errors[:20]:

        print(
            f"\nReview ID: {review['review_id']}"
        )

        print(
            f"Gold      : "
            f"{review['gold_aspects']}"
        )

        print(
            f"Predicted : "
            f"{review['predicted_aspects']}"
        )

        print(
            f"MISSING   : "
            f"{review['missing_aspects']}"
        )

        print(
            f"EXTRA     : "
            f"{review['extra_aspects']}"
        )

    print("\n" + "=" * 60)

    print(
        f"Saved report:\n"
        f"{OUTPUT_JSON_PATH}"
    )

    print(
        f"Saved review errors:\n"
        f"{OUTPUT_ERROR_PATH}"
    )


if __name__ == "__main__":
    main()

