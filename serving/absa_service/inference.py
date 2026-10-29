import threading
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from vllm.lora.request import LoRARequest
from pathlib import Path
import sys
from observability import trace_llm_request, log_llm_generation
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
load_dotenv(PROJECT_ROOT / ".env")

from services.absa.data.sft_formatter import format_record

_TOKENIZER = None
_LLM_ENGINE = None
_LORA_REQUEST = None
_ASYNC_LLM_ENGINE = None

base_model = os.getenv(
    "BASE_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)

adapter_path = os.getenv(
    "ADAPTER_PATH",
    "/home/2800/.ssh/Amazon Reviews/models/absa-qwen-lora/final",
).strip()

_model_load_lock = threading.Lock()
_inference_lock = threading.Lock()
    
def get_async_loaded_model():
    """
    Initializes tokenizer and vLLM AsyncLLMEngine once for true async continuous batching.
    """
    global _TOKENIZER, _ASYNC_LLM_ENGINE, _LORA_REQUEST

    if _ASYNC_LLM_ENGINE is not None:
        return _TOKENIZER, _ASYNC_LLM_ENGINE, _LORA_REQUEST
    
    with _model_load_lock:
        if _ASYNC_LLM_ENGINE is None:
            base_model = os.getenv(
                "BASE_MODEL",
                "Qwen/Qwen2.5-1.5B-Instruct",
            )

            adapter_path = os.getenv(
                "ADAPTER_PATH",
                "/home/2800/.ssh/Amazon Reviews/models/absa-qwen-lora/final",
            ).strip()

            if not adapter_path:
                raise ValueError("ADAPTER_PATH is not configured")

            adapter_dir = Path(adapter_path).expanduser()

            if not adapter_dir.is_dir():
                raise FileNotFoundError(
                    f"Could not find LoRA: {adapter_dir}"
                )

            if not (adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"adapter_config.json is not in {adapter_dir}"
                )

            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )

            print(f"Initializing vLLM Async Engine and model: {base_model}")

            engine_args = AsyncEngineArgs(
                model=base_model,
                tokenizer=base_model,
                enable_lora=True,
                max_lora_rank=32,
                max_loras=1,
                tensor_parallel_size=1,
                gpu_memory_utilization=float(
                    os.getenv("GPU_MEMORY_UTILIZATION", "0.8")
                ),
                max_model_len=512,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True,
            )

            _ASYNC_LLM_ENGINE = AsyncLLMEngine.from_engine_args(engine_args)

            _LORA_REQUEST = LoRARequest(
                "absa_lora",
                1,
                str(adapter_dir.resolve()),
            )

            _TOKENIZER = tokenizer

            print("Initialized vLLM Async Engine and LoRA successfully!")

    return _TOKENIZER, _ASYNC_LLM_ENGINE, _LORA_REQUEST

def get_loaded_model():
    """
    Khởi tạo tokenizer và vLLM engine một lần.
    Không tải model bằng Transformers/PEFT trước.
    """
    global _TOKENIZER, _LLM_ENGINE, _LORA_REQUEST

    if _LLM_ENGINE is not None:
        return _TOKENIZER, _LLM_ENGINE
    
    with _model_load_lock:
        if _LLM_ENGINE is None:
            if not adapter_path:
                raise ValueError("ADAPTER_PATH is not config")

            adapter_dir = Path(adapter_path).expanduser()

            if not adapter_dir.is_dir():
                raise FileNotFoundError(
                    f"Could not find LoRA: {adapter_dir}"
                )

            if not (adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"adapter_config.json is not in {adapter_dir}"
                )

            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )

            print(f"Initializing vLLM and model: {base_model}")

            llm = LLM(
                model=base_model,
                tokenizer=base_model,
                enable_lora=True,
                max_lora_rank=32,
                max_loras=1,
                tensor_parallel_size=1,
                gpu_memory_utilization=float(
                    os.getenv("GPU_MEMORY_UTILIZATION", "0.7")
                ),
                max_model_len=512,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True,
            )

            lora_request = LoRARequest(
                "absa_lora",
                1,
                str(adapter_dir.resolve()),
            )

            _TOKENIZER = tokenizer
            _LLM_ENGINE = llm
            _LORA_REQUEST = lora_request

            print("Initialized vLLM and LoRA successfully!")

    return _TOKENIZER, _LLM_ENGINE


def build_prompt(request: dict) -> str:
    return format_record(request)


def parse_json(text: str) -> dict:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)

    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, (dict, list)):
            return obj

    raise ValueError("Model output did not contain valid JSON")

async def predict_async(request: dict, request_id: str, max_new_tokens: int = 512) -> dict:
    tokenizer, engine, lora_request = get_async_loaded_model()
    raw = ""
    start_time = time.time()
    total_tokens = 0
    try:
        message = build_prompt(request)

        prompt = tokenizer.apply_chat_template(
            message["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        trace = trace_llm_request(
            model_name="absa_prediction_async",
            prompt=prompt,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        # True async generation stream call for iteration-level scheduling
        results_generator = engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        raw = final_output.outputs[0].text.strip()
        latency = time.time() - start_time
        
        output_tokens = len(final_output.outputs[0].token_ids)
        input_tokens = len(final_output.prompt_token_ids)
        total_tokens = input_tokens + output_tokens

        parsed = parse_json(raw)

        if isinstance(parsed, list):
            aspects = parsed
        elif isinstance(parsed, dict):
            aspects = parsed.get("aspects", [])
        else:
            aspects = []

        if not isinstance(aspects, list):
            raise ValueError("Aspects field is not a list")

        cleaned = []

        for item in aspects:
            if isinstance(item, dict):
                name = str(item.get("aspect", "")).strip()
                sentiment = str(
                    item.get("sentiment", "")
                ).strip().lower()

                if sentiment in {
                    "positive", "negative", "neutral", "mixed"
                }:
                    evidence = str(
                        item.get("evidence", "")
                    ).strip()

                    cleaned.append({
                        "aspect": name,
                        "sentiment": sentiment,
                        "evidence": evidence,
                    })

        log_llm_generation(trace, raw, total_tokens, latency, status="SUCCESS")

        return {
            "aspects": cleaned,
            "parse_success": True,
            "raw_output": raw,
            "error": None,
        }

    except Exception as exc:
        latency = time.time() - start_time
        err_message = str(exc) or f"{type(exc).__name__}: Unknown reason"
    
        log_llm_generation(
            trace, 
            raw if raw else str(exc), 
            total_tokens, 
            latency=round(latency, 4),
            status="FAILED",
            error_msg=err_message
        )
        return {
            "aspects": [],
            "parse_success": False,
            "raw_output": raw,
            "error": str(exc),
        }
    
def predict(request: dict, max_new_tokens: int = 512) -> dict:
    global _LORA_REQUEST

    tokenizer, llm = get_loaded_model()
    raw = ""
    start_time = time.time()
    total_tokens = 0
    try:
        message = build_prompt(request)

        prompt = tokenizer.apply_chat_template(
            message["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        trace = trace_llm_request(
            model_name="absa_prediction",
            prompt=prompt,
        )
        with _inference_lock:
            outputs = llm.generate(
                prompts=[prompt],
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                ),
                lora_request=_LORA_REQUEST,
                use_tqdm=False,
            )

        raw = outputs[0].outputs[0].text.strip()

        latency = time.time() - start_time
        
        output_tokens = len(outputs[0].outputs[0].token_ids)
        input_tokens = len(outputs[0].prompt_token_ids)
        total_tokens = input_tokens + output_tokens

        parsed = parse_json(raw)

        if isinstance(parsed, list):
            aspects = parsed
        elif isinstance(parsed, dict):
            aspects = parsed.get("aspects", [])
        else:
            aspects = []

        if not isinstance(aspects, list):
            raise ValueError("Aspects field is not a list")

        cleaned = []

        for item in aspects:
            if not isinstance(item, dict):
                continue

            name = str(item.get("aspect", "")).strip()
            sentiment = str(
                item.get("sentiment", "")
            ).strip().lower()

            if sentiment in {
                "positive", "negative", "neutral", "mixed"
            }:
                evidence = str(
                    item.get("evidence", "")
                ).strip()

                cleaned.append({
                    "aspect": name,
                    "sentiment": sentiment,
                    "evidence": evidence,
                })
        log_llm_generation(trace, raw, total_tokens, latency, status="SUCCESS")

        return {
            "aspects": cleaned,
            "parse_success": True,
            "raw_output": raw,
            "error": None,
        }

    except (Exception) as exc:
        latency = time.time() - start_time
        err_message = str(exc) or f"{type(exc).__name__}: Unkonwn reason"
    
        log_llm_generation(
            trace, 
            raw if raw else str(exc), 
            total_tokens, 
            latency=round(latency, 4),
            status="FAILED",
            error_msg=err_message
        )
        return {
            "aspects": [],
            "parse_success": False,
            "raw_output": raw,
            "error": str(exc),
        }
    
def predict_batch(requests: list, max_new_tokens: int = 512) -> list:
    global _LORA_REQUEST

    tokenizer, llm = get_loaded_model()

    prompts = []
    task_metadata = []

    for req in requests:
        try:
            message = build_prompt(req)
            prompt = tokenizer.apply_chat_template(
                message["messages"],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
            task_metadata.append({"success": True, "req": req})
        except Exception as e:
            task_metadata.append({"success": False, "req": req, "error": str(e)})
    
    valid_prompts = [p for p in prompts if p]

    if not valid_prompts:
        return [{"request": m["req"], "result": None, "error": m.get("error", "Invalid prompt")} for m in task_metadata]
    
    try:
        outputs = llm.generate(
            prompts=valid_prompts,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens
            ),
            lora_request = _LORA_REQUEST,
            use_tqdm=False
        )

        results = []
        out_idx = 0

        for meta in task_metadata:
            if not meta["success"]:
                results.append({"request": meta["req"], "result": None, "error": meta["error"]})
                continue
                
            output = outputs[out_idx]
            out_idx += 1
            raw = output.outputs[0].text.strip()
            
            try:
                parsed = parse_json(raw)
                aspects = parsed.get("aspects", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
                
                cleaned = []
                for item in aspects:
                    if isinstance(item, dict):
                        name = str(item.get("aspect", "")).strip()
                        sentiment = str(item.get("sentiment", "")).strip().lower()
                        if sentiment in {"positive", "negative", "neutral", "mixed"}:
                            cleaned.append({
                                "aspect": name,
                                "sentiment": sentiment,
                                "evidence": str(item.get("evidence", "")).strip(),
                            })
                
                results.append({
                    "request": meta["req"],
                    "result": {"aspects": cleaned, "parse_success": True, "raw_output": raw, "error": None},
                    "error": None
                })
            except (ValueError, json.JSONDecodeError) as exc:
                results.append({
                    "request": meta["req"],
                    "result": {"aspects": [], "parse_success": False, "raw_output": raw, "error": str(exc)},
                    "error": str(exc)
                })
        return results

    except Exception as e:
        return [{"request": m["req"], "result": None, "error": str(e)} for m in task_metadata]


