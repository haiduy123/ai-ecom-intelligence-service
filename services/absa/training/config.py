from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / '.env')

@dataclass
class TrainingConfig:
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_name: str = os.getenv("BASE_MODEL")

    output_dir: Path = (
        PROJECT_ROOT
        / "models"
        / "absa-qwen-lora"
    )

    checkpoint_dir: Path = (
        PROJECT_ROOT
        / "models"
        / "absa-qwen-lora"
        / "checkpoints"
    )

    final_dir: Path = (
        PROJECT_ROOT
        / "models"
        / "absa-qwen-lora"
        / "final"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_file: Path = (
        PROJECT_ROOT
        / "data"
        / "sft"
        / "train.jsonl"
    )

    validation_file: Path = (
        PROJECT_ROOT
        / "data"
        / "sft"
        / "validation.jsonl"
    )

    test_file: Path = (
        PROJECT_ROOT
        / "data"
        / "sft"
        / "test.jsonl"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    num_train_epochs: int = 3

    learning_rate: float = 5e-5

    per_device_train_batch_size: int = 1

    per_device_eval_batch_size: int = 1

    gradient_accumulation_steps: int = 16

    warmup_steps: float = 2

    weight_decay: float = 0.01

    max_seq_length: int = 1024

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps: int = 10

    eval_steps: int = 10

    save_steps: int = 10

    save_total_limit: int = 3

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed: int = 42

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_r: int = 32

    lora_alpha: int = 64

    lora_dropout: float = 0.05

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    use_4bit: bool = True