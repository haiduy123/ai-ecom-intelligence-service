import os
import uuid 
import json

from pathlib import Path
from dotenv import load_dotenv
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
load_dotenv(PROJECT_ROOT / '.env', override=True)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, status, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from schemas import PredictRequest, PredictResponse
from inference import predict, predict_async
from redis_queue import redis_queue
from semantic_cache import redis_semantic_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("ABSA-Serving")

MODEL_NAME=os.getenv("MODEL_NAME", "absa_model")
API_KEY = os.getenv("API_KEY", "mysecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_loaded = True
    
    try:
        redis_queue.client.ping()
        app.state.redis_connected = True
        logger.info("Connected to Redis successfully (Producer Mode)")
    except Exception as e:
        app.state.redis_connected = False
        app.state.redis_error = str(e)
        logger.exception("Failed to connect to Redis")
    yield
    logger.info("Shutting down Producer application.")

app = FastAPI(title="Amazon Review ABSA API", version="1.0.0", lifespan=lifespan)

@app.get("/ready")
def ready():
    if not bool(getattr(app.state, "model_loaded", False)):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not bool(getattr(app.state, "redis_connected", False)):
        raise HTTPException(status_code=503, detail="Redis connection failed")
    return {
        "status": "ready", 
        "model_name": getattr(app.state, "model_name", None),
        "redis_connected": True
    }

@app.get("/health")
def health():
    loaded = bool(getattr(app.state, "model_loaded", False))
    redis_conn = bool(getattr(app.state, "redis_connected", False))

    return {
        "status": "ok" if loaded else "error",
        "model_loaded": loaded,
        "model_name": getattr(app.state, "model_name", None),
        "redis_connected": redis_conn,
        "error": getattr(app.state, "model_error", None),
        "redis_error": getattr(app.state, "redis_error", None),
    }

@app.post("/predict/vllm-async", response_model=PredictResponse)
async def predict_async_endpoint(request: PredictRequest, api_key: str = Security(verify_api_key)):
    if not bool(getattr(app.state, "model_loaded", False)):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        request_data = request.model_dump()
        request_id = request_data.get("review_id") or str(uuid.uuid4())

        # Await the async inference handler with iteration-level scheduling
        result = await predict_async(request_data, request_id=request_id)

        if result.get("error") is not None or not result.get("parse_success", True):
            raise HTTPException(
                status_code=400,
                detail=result.get("error")
            )

        return PredictResponse(
            model_name=MODEL_NAME,
            review_id=request_data.get("review_id"),
            aspects=result["aspects"],
            parse_success=result["parse_success"],
            raw_output=result["raw_output"],
            error=result["error"]
        )
    except Exception as e:
        logger.error(f"Async prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/vllm-native", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not bool(getattr(app.state, "model_loaded", False)):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt_text = request.comment
        review_id = request.review_id

        cached_raw = redis_semantic_cache.get(prompt_text, review_id)
        if cached_raw:
            cached_data = json.loads(cached_raw)
            return PredictResponse(
                model_name=MODEL_NAME + " (Cached)",
                review_id=request.review_id,
                aspects=cached_data["aspects"],
                parse_success=True,
                raw_output=cached_data.get("raw_output", "From Semantic Cache"),
                error=None
            )
        
        request = request.model_dump()

        result = predict(request)

        if result.get("error") is not None or not result.get("parse_success", True):
            raise HTTPException(
                status_code=400,
                detail=result.get("error")
            )

        if result["parse_success"]:
            cache_data = {
                "aspects": result["aspects"],
                "raw_output": result["raw_output"],
            }

            redis_semantic_cache.set(
                prompt_text,
                json.dumps(cache_data),
                review_id
            )

        return PredictResponse(
            model_name=MODEL_NAME,
            review_id=request.get("review_id"),
            aspects=result["aspects"],
            parse_success=result["parse_success"],
            raw_output=result["raw_output"],
            error=result["error"]
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/redis-queue")
def predict_async_endpoint(request: PredictRequest, api_key: str = Security(verify_api_key)):
    if not bool(getattr(app.state, "redis_connected", False)):
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    if not bool(getattr(app.state, "redis_connected", False)):
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    try:
        request_data = request.model_dump()
        
        # prompt_text = request_data.get('comment')
        task_id = request_data.get("review_id") or str(uuid.uuid4())
        request_data["task_id"] = task_id

        # cached_raw = redis_semantic_cache.get(prompt_text)
        # if cached_raw:
        #     cached_data = json.loads(cached_raw)
        #     success_result = {
        #         "aspects": cached_data["aspects"],
        #         "parse_success": True,
        #         "raw_output": cached_data.get("raw_output", "From Semantic Cache"),
        #         "error": None
        #     }
           
        #     redis_queue.set_task_status(task_id=task_id, status="SUCCESS", result=success_result)
        #     logger.info(f"[CACHE HIT] Queue task {task_id} resolved instantly from cache.")
        #     return {
        #         "status": "cached",
        #         "task_id": task_id,
        #         "message": "Request resolved instantly from Semantic Cache"
        #     }

        redis_queue.set_task_status(task_id=task_id, status="QUEUED")
        redis_queue.push_request(request_data)

        return {
            "status": "queued",
            "task_id": task_id,
            "message": "Request was added into queue"
        }
    except Exception as e:
        logger.error(f"Enqueue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
def get_task_status_endpoint(task_id: str, api_key: str = Security(verify_api_key)):
    try:
        task_info = redis_queue.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="TaskID is not exit")
        return task_info
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[ERROR] Failed to fetch task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")