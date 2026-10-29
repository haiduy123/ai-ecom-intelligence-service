from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.absa.data.loader import save_jsonl, load_jsonl

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits_data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "sft"
)

SYSTEM_PROMPT = """
You are an expert Aspect-Based Sentiment Analysis system for Amazon product reviews.

Given an Amazon product review:
1. Extract all relevant product aspects.
2. Map each aspect to a clean, concise canonical aspect name (avoid overly fragmented sub-aspects; group closely related features under standard, general names).
3. Determine the sentiment strictly from: positive, negative, neutral, mixed.
4. Extract the exact short phrase from the review as evidence.
5. Return valid JSON only.

The output must contain an "aspects" array.

Each object in "aspects" must contain:
- aspect (string: clean canonical name)
- sentiment (string: positive, negative, neutral, or mixed)
- evidence (string: exact short substring copied from the review)

Evidence must be an exact span copied from the review.
""".strip()


def build_output(record: dict[str, any]) -> str:
    aspects = []

    for annotation in record.get("aspects" , [],):
        aspects.append({
            "aspect": annotation["aspect"],
            "sentiment": annotation["sentiment"],
            "evidence": annotation["evidence"]
        })
    
    output = {
        "aspects": aspects
    }

    return json.dumps(output, ensure_ascii=False)

def format_record(record: dict[str, any]) -> dict[str, Any]:
    comment = record.get("comment", "")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Analyze this product review:\n\n"
                f"{comment}"
            )},
            # {"role": "assistant", "content": build_output(record)}
        ],
        "review_id": record.get("review_id"),
    }

def convert_split(split_name: str) -> None:
    input_path = (
        INPUT_DIR
        / f"{split_name}.jsonl"
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}.jsonl"
    )

    records = load_jsonl(input_path)

    formatted = [
        format_record(record)
        for record in records
    ]

    save_jsonl(formatted, output_path)

    print(
        f"{split_name}: "
        f"{len(formatted)} records"
    )

def main() -> None:
    for split_name in ["train", "validation", "test"]:
        convert_split(split_name)

if __name__ == "__main__":
    main()
