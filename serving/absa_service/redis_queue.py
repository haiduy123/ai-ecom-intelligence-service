import os
import json
import redis
from logger import setup_logger

logger = setup_logger("RedisQueue")

class RedisQueueManager:
    def __init__(self, queue_name: str="absa_task_queue"):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        self.client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.queue_name = queue_name

        logger.info("Connected to Redis Queue at %s:%d", redis_host, redis_port)

    def push_request(self, data: dict):
        serialized_data = json.dumps(data)
        self.client.rpush(self.queue_name, serialized_data)

        logger.info("[QUEUE PUSH] Added request to Redis queue.")

    def pop_request(self):
        result = self.client.blpop(self.queue_name, timeout=1)

        if result:
            _, serialized_data = result
            return json.loads(serialized_data.decode('utf-8'))
        
        return None

    def get_queue_size(self) -> int:
        return self.client.llen(self.queue_name)
    

    def set_task_status(self, task_id: str, status: str, result: dict = None, error: str = None, ttl: int = 3600):
        key = f"task:{task_id}"
        payload = {
            "task_id": task_id,
            "status": status,
            "result": result,
            "error": error
        }
        self.client.setex(key, ttl, json.dumps(payload))

    def get_task_status(self, task_id: str, ttl: int = 3600):
        key = f"task:{task_id}"
        data = self.client.get(key)
        if data:
            result = json.loads(data.decode('utf-8'))
            return result

        return None
    

redis_queue = RedisQueueManager()