import os
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig, modeling_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent 
load_dotenv(PROJECT_ROOT / '.env')

def load_model():
    base = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter = os.getenv("ADAPTER_PATH", "/home/2800/.ssh/Amazon Reviews/models/absa-qwen-lora/final").strip()
    quant4 = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    
    kwargs = {"device_map": os.getenv("DEVICE_MAP", "auto"),
              "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
              "trust_remote_code": True}
    
    if quant4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        
    if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
        modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model