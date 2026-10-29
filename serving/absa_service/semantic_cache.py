import os 
import pickle
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from logger import setup_logger

logger = setup_logger("RedisSemanticCache")

class RedisSemanticCache:
    def __init__(self, threshold: float = 0.92, index_name: str = "idx:cache"):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.index_name = index_name

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        self.client = redis.Redis(host=redis_host, port=redis_port, db=0)
        logger.info("Connected to Redis at %s:%d", redis_host, redis_port)

        self._create_index_if_not_exist()

    def _create_index_if_not_exist(self):
        try:
            self.client.execute_command(f"FT.INFO {self.index_name}")
            logger.info("RediSearch index '%s' already exists.", self.index_name)
        except Exception:
            logger.info("Creating RediSearch index '%s' with HNSW vector search...", self.index_name)
            try:
                self.client.execute_command(
                    "FT.CREATE", self.index_name,
                    "ON", "HASH",
                    "PREFIX", "1", "cache:",
                    "SCHEMA",
                    "prompt", "TEXT",
                    "response", "TEXT",
                    "vector", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", "384",
                    "DISTANCE_METRIC", "COSINE"
                )
                logger.info("Successfully created RediSearch index '%s'", self.index_name)
            except Exception as e:
                logger.error("Failed to create index: %s", e)

    def get(self, prompt: str, review_id):
        query_vector = self.encoder.encode(prompt).astype(np.float32).tobytes()

        query_string = "*=>[KNN 1 @vector $vec AS score]"
        
        try:
            res = self.client.execute_command(
                "FT.SEARCH", self.index_name,
                query_string,
                "PARAMS", "2", "vec", query_vector,
                "SORTBY", "score", "ASC",
                "DIALECT", "2"
            )
            if not res or res.get(b'total_results', 0) == 0:
                logger.info("[CACHE MISS] for prompt: '%s' ------------ %s", prompt[:30], review_id)
                return None

            results = res.get(b'results', [])

            if not results:
                logger.info("[CACHE MISS] for prompt: '%s' ------------ %s", prompt[:30], review_id)
                return None

            field_dict = results[0].get(b'extra_attributes', {})

            raw_score = field_dict.get(b'score')
            raw_response = field_dict.get(b'response')

            if raw_score is None or raw_response is None:
                logger.warning("Search result missing score or response")
                return None

            distance = float(raw_score)
            similarity = 1.0 - distance

            print("similarity:", similarity)

            if similarity >= self.threshold:
                stored_response = (
                    raw_response.decode("utf-8")
                    if isinstance(raw_response, bytes)
                    else raw_response
                )

                logger.info(
                    "[CACHE HIT] Similarity: %.4f for prompt: '%s' ------------- %s",
                    similarity, prompt[:30], review_id
                )
                return stored_response

            logger.info(
                "[CACHE MISS] Similarity %.4f < threshold %.4f for prompt: '%s' ------------- %s",
                    similarity, prompt[:30], review_id
            )
            return None

        except Exception as e:
            logger.error("Semantic cache search error: %s", e)
            return None
    
    def set(self, prompt: str, response: str, review_id):
        """Lưu trữ prompt, vector dưới dạng binary và response vào Redis Hash"""
        vector = self.encoder.encode(prompt).astype(np.float32).tobytes()
        
        cache_key = f"cache:{abs(hash(prompt))}"
        
        self.client.hset(cache_key, mapping={
            'vector': vector,
            'response': response,
            'prompt': prompt
        })
        
        self.client.expire(cache_key, 86400)
        logger.info("[CACHE STORED IN REDIS] Key: %s --------- %s", cache_key, review_id)

redis_semantic_cache = RedisSemanticCache(threshold=0.65)