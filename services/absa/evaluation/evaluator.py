from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import sys

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from .metrics import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]
 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.absa.data.loader import load_jsonl
from services.absa.data.loader import save_jsonl
from services.absa.data.sft_formatter import format_record

# ---------------------------------------------------------------------------
# Model output parsing
# ---------------------------------------------------------------------------
import json
import re
from typing import Any

def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()

    # 1. Thử parse trực tiếp (dành cho JSON1)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Làm sạch các ký tự Markdown ```json và ``` nếu có
    cleaned_text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        result = json.loads(cleaned_text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Tìm khối JSON từ dấu '{' ĐẦU TIÊN đến dấu '}' CUỐI CÙNG (Dùng Greedy pattern r"\{.*\}" thay vì ".*?")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None

def load_model(
        model_name: str,
        adapter_path: str | None = None
):
    
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(
            model,
            adapter_path
        )
    else:
        print("Running with Base Model only (No LoRA adapter).")

    model.eval()

    return model, tokenizer

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_prediction(
    model,
    tokenizer,
    review: dict[str, Any],
    max_new_token: int = 1000
) -> tuple[dict[str, Any] | None, str]:

    messages = format_record(review)

    prompt = tokenizer.apply_chat_template(
        messages['messages'],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_token,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    try:
        parsed = extract_json(generated_text)
    except Exception as e:
        print(e)

    
    return parsed, generated_text

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
        model_name: str,
        adapter_path: str,
        predictions_path: str,
        metrics_path: str,
        test_path: str
):
    test_path = Path(test_path)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)

    print(
        f"Loading test data from {test_path}..."
    )

    test_records = load_jsonl(test_path)

    print(
        f"Test data loaded: {len(test_records)} records."
    )

    model, tokenizer = load_model(
        model_name,
        adapter_path
    )

    prediction_records = []

    metric_predictions = []
    metric_references = []

    failed_parses = 0

    for index, review in enumerate(test_records, start=1):
        print(
            f"\rEvaluating "
            f"{index}/{len(test_records)}",
            end="",
            flush=True,
        )

        prediction, raw_output = (
            generate_prediction(
                model=model,
                tokenizer=tokenizer,
                review=review
            )
        )
        
        parses_failed = (
            prediction is None
        )

        if parses_failed:
            failed_parses += 1

            prediction = {
                "aspects": []
            }

        if not isinstance(
            prediction.get("aspects"),
            list
        ):
            prediction["aspects"] = []

        reference = {
            "aspects": review.get(
                "aspects",
                [],
            )
        }

        prediction_records.append(
            {
                "review_id": review.get(
                    "review_id"
                ),
                "prediction": prediction,
                "reference": reference,
                "raw_output": raw_output,
                "parse_failed": parses_failed,
            }
        )

        metric_predictions.append(prediction)

        metric_references.append(reference)
    print()

    # -----------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------

    save_jsonl(
        prediction_records,
        predictions_path
    )

    # -----------------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------------

    metrics = calculate_metrics(
        predictions=metric_predictions,
        references=metric_references,
        threshold=0.75
    )

    metrics["test_examples"] = len(test_records)

    metrics["failed_json_parses"] = failed_parses

    metrics["json_parse_rate"] = (
        1.0
        - (
            failed_parses
            / len(test_records)
        )
        if test_records
        else 0.0
    )
    # ---------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(f"{'Aspect Extraction':<30} | {'Strict':<15} | {'Embedding (Semantic)':<15}")
    print("-" * 70)
    print(f"{'Precision':<30} | {metrics['aspect_precision']:<15.4f} | {metrics.get('emb_aspect_precision', 0):<15.4f}")
    print(f"{'Recall':<30} | {metrics['aspect_recall']:<15.4f} | {metrics.get('emb_aspect_recall', 0):<15.4f}")
    print(f"{'F1':<30} | {metrics['aspect_f1']:<15.4f} | {metrics.get('emb_aspect_f1', 0):<15.4f}")
    print(f"{'TP / FP / FN':<30} | {metrics['aspect_tp']}/{metrics['aspect_fp']}/{metrics['aspect_fn']:<9} | {metrics.get('emb_aspect_tp')}/{metrics.get('emb_aspect_fp')}/{metrics.get('emb_aspect_fn')}")

    print()
    print(f"{'Sentiment (Matched Aspects)':<30} | {'Strict Matches':<15} | {'Semantic Matches':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<30} | {metrics['sentiment_accuracy']:<15.4f} | {metrics.get('emb_sentiment_accuracy', 0):<15.4f}")

    print()
    print(f"{'End-to-End ABSA':<30} | {'Strict':<15} | {'Embedding (Semantic)':<15}")
    print("-" * 70)
    print(f"{'Precision':<30} | {metrics['end_to_end_precision']:<15.4f} | {metrics.get('emb_end_to_end_precision', 0):<15.4f}")
    print(f"{'Recall':<30} | {metrics['end_to_end_recall']:<15.4f} | {metrics.get('emb_end_to_end_recall', 0):<15.4f}")
    print(f"{'F1':<30} | {metrics['end_to_end_f1']:<15.4f} | {metrics.get('emb_end_to_end_f1', 0):<15.4f}")

    print()
    print("Generation Quality")
    print("-" * 70)
    print(f"JSON parse rate: {metrics['json_parse_rate']:.4f}")
    print(f"Failed parses  : {metrics['failed_json_parses']}")

    print()
    print("=" * 70)
    print(f"Predictions saved to:\n  {predictions_path}")
    print(f"Metrics saved to:\n  {metrics_path}")

    return metrics