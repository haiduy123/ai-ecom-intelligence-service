import time
import requests
import concurrent.futures

from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

API_URL_NATIVE = os.getenv("API_URL_NATIVE")

ROUND_1_REVIEWS = [
    {"review_id": "r1", "comment": "The screen display is exceptionally clear and gorgeous."},
    {"review_id": "r3", "comment": "Battery life drains way too fast, very disappointing."},
    {"review_id": "r5", "comment": "The price is a bit high for this kind of quality."},
    {"review_id": "r7", "comment": "Customer service was unhelpful and rude."}
]

ROUND_2_REVIEWS = [
    {"review_id": "r2", "comment": "Amazing phone, great screen quality and vivid colors!"},
    {"review_id": "r4", "comment": "The battery dies so quickly, I hate it."},  
    {"review_id": "r6", "comment": "Too expensive compared to other brands on the market."},
    {"review_id": "r8", "comment": "Support staff were very rude and gave no help."}
]

def send_request(session, payload):
    start_time = time.time()

    try:
        response = session.post(API_URL_NATIVE, json=payload, timeout=30)
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            model_name = data.get("model_name", "")

            is_cached = "(Cached)" in model_name
            return {"success": True, "latency": latency, "cached": is_cached}
        else:
            return {"success": False, "latency": latency, "cached": False}
    except Exception as e:
        print(f"Request failed: {e}")
        return {"success": False, "latency": 0, "cached": False}
    
def run_benchmark(round_name, requests_data, max_workers=1):
    print((f"\n================ [{round_name}] ================"))
    latencies = []
    cache_hits = 0
    cache_misses = 0

    session = requests.Session()

    start_total = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_request, session, req) for req in requests_data]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["success"]:
                latencies.append(res["latency"])
                if res["cached"]:
                    cache_hits += 1
                else:
                    cache_misses += 1
    
    total_time = time.time() - start_total
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    else:
        avg_latency = p95_latency = 0
    
    print(f"Total time: {total_time:.4f} second")
    print(f"Total request: {len(requests_data)}")
    print(f"Cache Hits: {cache_hits} | Cache Misses: {cache_misses}")
    print(f"(Avg Latency): {avg_latency:.2f} ms")
    print(f"Latency P95: {p95_latency:.2f} ms")
    print(f"Throughput: {len(requests_data) / total_time:.2f} req/sec")

if __name__ == "__main__":
    print("Start running Benchmark to compare performance of Semantic Cache...")
    
    print("\n[Waiting 2s before running round 2...]")
    time.sleep(2)
    
    run_benchmark("Round 2: WARM RUN (Caches Hit)", ROUND_2_REVIEWS)