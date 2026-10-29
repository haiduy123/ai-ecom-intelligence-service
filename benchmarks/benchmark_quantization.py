import time
import gc
import json
import os

os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/huggingface/hub")

import re
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, modeling_utils
from peft import PeftModel
from dotenv import load_dotenv

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

from services.absa.evaluation.metrics import calculate_metrics
from services.absa.evaluation.evaluator import extract_json, generate_prediction
from services.absa.data.loader import load_jsonl

MODEL_NAME = os.getenv("BASE_MODEL")
LORA_ADAPTER_PATH = os.getenv("ADAPTER_PATH").strip()
TEST_DATA_PATH = PROJECT_ROOT / "data/splits_data/test.jsonl" 
OUTPUT_DIR = PROJECT_ROOT / "quantization_benchmark"

def get_vram_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def run_evaluation_loop(model, tokenizer, test_records):
    metric_predictions = []
    metric_references = []
    failed_parses = 0

    for index, review in enumerate(test_records, start=1):
        prediction, _ = generate_prediction(model, tokenizer, review)
        
        parses_failed = (prediction is None)
        if parses_failed:
            failed_parses += 1
            prediction = {"aspects": []}

        if not isinstance(prediction.get("aspects"), list):
            prediction["aspects"] = []

        reference = {"aspects": review.get("aspects", [])}

        metric_predictions.append(prediction)
        metric_references.append(reference)

    metrics = calculate_metrics(
        predictions=metric_predictions,
        references=metric_references,
        threshold=0.75
    )

    metrics["test_examples"] = len(test_records)
    metrics["failed_json_parses"] = failed_parses
    metrics["json_parse_rate"] = (1.0 - (failed_parses / len(test_records))) if test_records else 0.0

    return metrics

def run_quantization_benchmark():
    print("=" * 60)
    print("Running Quantization Trade-off Benchmark (VRAM, Size vs Accuracy)...")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this benchmark on a machine with a GPU.")
        return

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Không tìm thấy file test tại {TEST_DATA_PATH}")
        return

    test_records = load_jsonl(TEST_DATA_PATH)
    print(f"Đã tải {len(test_records)} bản ghi từ tập test.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    configs = {
        "FP16": None,
        "INT8": BitsAndBytesConfig(load_in_8bit=True),
        "INT4": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    }

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_name": MODEL_NAME,
        "adapter_path": LORA_ADAPTER_PATH,
        "configurations": []
    }

    for name, q_config in configs.items():
        print(f"\n" + "-"*50)
        print(f"Benchmarking Quantization Config: {name}")
        print("-" * 50)
        
        gc.collect()
        torch.cuda.empty_cache()
        baseline_vram = get_vram_usage()

        try:
            if q_config is None:
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to("cuda")
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, 
                    quantization_config=q_config, 
                    device_map="auto",
                    trust_remote_code=True
                )

            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            model.eval()
        except Exception as e:
            print(f"Failed to load model with {name} quantization: {e}")
            continue

        peak_vram = get_vram_usage() - baseline_vram
        param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        print(f"[*] Running evaluation Accuracy/ABSA metrics for config {name}...")
        eval_metrics = run_evaluation_loop(model, tokenizer, test_records)

        config_entry = {
            "config_name": name,
            "weights_size_mb": round(param_size_mb, 2),
            "actual_vram_usage_mb": round(peak_vram, 2),
            "metrics": eval_metrics
        }

        benchmark_summary["configurations"].append(config_entry)
        
        print(f"[{name}] Result:")
        print(f"  - Weights size       : {param_size_mb:.2f} MB")
        print(f"  - Actual VRAM usage  : {peak_vram:.2f} MB")
        print(f"  - Aspect Recall (Strict) : {eval_metrics.get('aspect_recall', 0):.4f}")
        print(f"  - Aspect Precision (Strict) : {eval_metrics.get('aspect_precision', 0):.4f}")
        print(f"  - Aspect Recall (Embedding) : {eval_metrics.get('emb_aspect_recall', 0):.4f}")
        print(f"  - Aspect Precision (Embedding) : {eval_metrics.get('emb_aspect_precision', 0):.4f}")
        print(f"  - End-to-End F1      : {eval_metrics.get('end_to_end_f1', 0):.4f}")
        print(f"  - JSON Parse Rate    : {eval_metrics.get('json_parse_rate', 0):.4f}")

        del model 
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "quantization_tradeoff_full_benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Benchmark completed. Saved at:\n  {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    run_quantization_benchmark()