import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse, get_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def trace_llm_request(prompt: str, model_name: str):
    client = get_client()

    generation = client.start_observation(
        name="vllm_absa_generation",
        as_type="generation",
        model=model_name,
        input=prompt,
    )
    return generation


def log_llm_generation(
    generation,
    output_text: str,
    tokens_used: int,
    latency: float,
    status: str,
    error_msg="None"
):
    try:
        if generation:
            if status == "SUCCESS":
                generation.update(
                    name="vllm_generate",
                    output=output_text,
                    usage={"total_tokens": tokens_used},
                    metadata={"latency_seconds": latency, "status": "SUCCESS"}
                )
            else:
                generation.update(
                    name="vllm_generate_error",
                    output=output_text,
                    level="ERROR",
                    metadata={"latency_seconds": latency, "status": status, "error": error_msg}
                )
            generation.end()
            

    except Exception as e:
        print("Failed to log Langfuse generation: %s", e)

    finally:
        try:
            get_client().flush()
        except Exception:
            pass