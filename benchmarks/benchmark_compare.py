import asyncio
import httpx
import time
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

API_URL_NATIVE = os.getenv("API_URL_NATIVE")
API_URL_REDIS = os.getenv("API_URL_REDIS")
API_KEY = os.getenv("API_KEY")

CONCURRENCY = 2

headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
payload_list = [
    {
        "review_id": "bench_test_1",
        "comment": "The screen display is exceptionally clear and gorgeous."
    },
    {
        "review_id": "bench_test_2",
        "comment": "The price is a bit high for this kind of quality."
    }
]

async def benchmark_endpoint(client, url, is_redis=False):
    start_time = time.time()
    
    async def send_one(i):
        base_payload = payload_list[i % len(payload_list)]
        
        p = base_payload.copy()
        p["review_id"] = f"rev_{i}"
        try:
            resp = await client.post(url, json=p, headers=headers, timeout=30.0)
            return resp.status_code == 200
        except Exception:
            return False

    tasks = [send_one(i) for i in range(CONCURRENCY)]
    results = await asyncio.gather(*tasks)
    
    total_duration = time.time() - start_time
    success_count = sum(1 for r in results if r)
    return total_duration, success_count

async def main():
    print(f"=== START BENCHMARK TO COMPARE (Concurrency = {CONCURRENCY}) ===\n")
    async with httpx.AsyncClient() as client:
        
        # 1. Test Native vLLM
        print("[1/2] Testing Native vLLM (/predict/vllm-native)...")
        time_native, succ_native = await benchmark_endpoint(client, API_URL_NATIVE, is_redis=False)
        print(f"-> Total time: {time_native:.2f}s | Success: {succ_native}/{CONCURRENCY}")
        print(f"-> AVG Latency API: {(time_native/CONCURRENCY)*1000:.2f} ms\n")

        # 2. Test vLLM + Redis Queue
        print("[2/2] TESTING vLLM + Redis Queue (/predict/redis-queue)...")
        time_redis, succ_redis = await benchmark_endpoint(client, API_URL_REDIS, is_redis=True)
        print(f"-> Toal time pushinh queue: {time_redis:.2f}s | Success: {succ_redis}/{CONCURRENCY}")
        print(f"-> AVG Latency API (Client response time): {(time_redis/CONCURRENCY)*1000:.2f} ms\n")

if __name__ == "__main__":
    asyncio.run(main())