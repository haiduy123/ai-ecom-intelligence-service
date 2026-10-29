from __future__ import annotations

import torch

from peft import LoraConfig

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

import sys

# ============================================================
# PROJECT ROOT
# ============================================================

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from services.absa.training.config import (
    TrainingConfig
)

def create_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else
            torch.float16
        ),
        bnb_4bit_use_double_quant = True
    )

def load_model(model_name: str, use_4bit: bool = True):
    quantization_config = None

    if use_4bit:
        quantization_config = (create_quantization_config())

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else
            torch.float16
        )
    )

    model.config.use_cache = False

    return model

def create_lora_config():
    config = TrainingConfig()
    return LoraConfig(
        r = config.lora_r,
        lora_alpha = config.lora_alpha,
        lora_dropout = config.lora_dropout,
        task_type = "CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
