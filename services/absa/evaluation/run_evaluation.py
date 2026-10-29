from pathlib import Path
from dotenv import load_dotenv
import os

from .evaluator import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / '.env')

MODEL_NAME = os.getenv("BASE_MODEL")
ADAPTER_PATH = os.getenv("ADAPTER_PATH")

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits_data"
    / "test.jsonl"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "predictions_base_model.jsonl"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "metrics_base_model.json"
)


if __name__ == "__main__":
    evaluate(
        model_name=MODEL_NAME,
        adapter_path=None,
        test_path=str(TEST_PATH),
        predictions_path=str(PREDICTIONS_PATH),
        metrics_path=str(METRICS_PATH),
    )