import time
import logging
from redis_queue import redis_queue
from inference import get_loaded_model, predict_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABSA-Worker")

MAX_RETRIES = 3
BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05

def run_worker():
    logger.info("Starting ABSA Micro-batching Worker with Retry Mechanism...")
    get_loaded_model()
    
    while True:
        batch = []
        start_time = time.time()
        
        while len(batch) < BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT:
            task_data = redis_queue.pop_request()
            if task_data:
                task_id = task_data.get("task_id")
                redis_queue.set_task_status(task_id=task_id, status="PROCESSING")

                batch.append(task_data)
            else:
                time.sleep(0.01)
                
        if not batch:
            continue
            
        logger.info(f"[WORKER] Processing batch of {len(batch)} tasks.")
        
        batch_results = predict_batch(batch)
        
        for item in batch_results:
            req = item["request"]
            res = item["result"]
            err = item["error"]
            task_id = req.get("task_id")
            
            if err is None and res and res.get("parse_success"):
                redis_queue.set_task_status(task_id=task_id, status="SUCCESS", result=res)
            else:
                retry_count = req.get("retry_count", 0)
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    req["retry_count"] = retry_count
                    logger.warning(f"[WORKER] Task {task_id} failed. Retrying ({retry_count}/{MAX_RETRIES})...")
                    redis_queue.push_request(req)
                    redis_queue.set_task_status(task_id=task_id, status=f"RETRYING_{retry_count}")
                else:
                    logger.error(f"[WORKER] Task {task_id} permanently FAILED after {MAX_RETRIES} retries.")
                    redis_queue.set_task_status(task_id=task_id, status="FAILED", error=err or "Unknown error")

if __name__ == "__main__":
    run_worker()