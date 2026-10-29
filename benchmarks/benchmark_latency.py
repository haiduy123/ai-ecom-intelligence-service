import time
import torch
import json
import httpx
import asyncio

from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

API_URL_NATIVE = os.getenv("API_URL_NATIVE")
API_URL_REDIS = os.getenv("API_URL_REDIS")
API_URL_ASYNC = os.getenv("API_URL_ASYNC")
API_URL_STATUS = os.getenv("API_URL_STATUS")
API_KEY = os.getenv("API_KEY")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "artifacts")
TOTAL_REQUESTS = int(os.environ.get("TOTAL_REQUESTS", 100))
CONCURRENCY = int(os.environ.get("CONCURRENCY", 5))

sample_payloads = [
    {
        "review_id": f"test_{str(i).zfill(3)}",
        "comment": "Really impressed with this one. Battery life is excellent — only needed to charge it once every 3-4 days with normal use, which is way better than I expected at this price point. Heart rate monitoring, activity tracking, and the touchscreen all work smoothly. The rose gold/pink color and band look and feel more premium than the price suggests. My wife loves wearing it daily. Packaging was simple but everything needed (charging cable, instructions) was included. Would definitely recommend if you want most of the smartwatch features without the premium price tag."
    }
    for i in range(1, 101)
]

async def send_single_request(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, payload: dict, api_url: str):
    print(f"[DEBUG BENCHMARK] Sending request: {payload.get('review_id')}")
    async with semaphore:
        start_time = time.time()
        try:
            headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
            
            response = await client.post(api_url, json=payload, headers=headers, timeout=None)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"[DEBUG] Status: {response.status_code} | Body: {response.text}")
                return latency_ms, False
            
            data = response.json()
            task_id = data.get("task_id")
            
            # Nếu là endpoint Native hoặc Async trực tiếp, response trả về kết quả ngay (không có task_id từ redis)
            if not task_id or "vllm-native" in api_url or "vllm-async" in api_url:
                latency_ms = (time.time() - start_time) * 1000
                return latency_ms, True
            
            # Trường hợp Redis Queue cần polling status
            status_url = f"{API_URL_STATUS.rstrip('/')}/{task_id}"

            while True:
                try:
                    status_response = await client.get(status_url, headers=headers)
                    if status_response.status_code == 200:
                        task_info = status_response.json()
                        task_status = task_info.get("status")
                        if task_status == "SUCCESS":
                            latency_ms = (time.time() - start_time) * 1000
                            return latency_ms, True
                        elif task_status == "FAILED":
                            latency_ms = (time.time() - start_time) * 1000
                            print(f"[DEBUG] Task {task_id} failed: {task_info.get('error')}")
                            return latency_ms, False
                except (httpx.RemoteProtocolError, httpx.RequestError):
                    pass
                await asyncio.sleep(2)
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"[DEBUG Exception] Type: {type(e).__name__} | Details: {str(e)}")
            return latency_ms, False

async def run_benchmark_for_endpoint(client: httpx.AsyncClient, name: str, api_url: str):
    if not api_url:
        print(f"[{name}] API URL is not set in environment variables. Skipping...")
        return None

    print("=" * 60)
    print(f"Running API latency benchmark for: {name}")
    print(f"URL: {api_url}")
    print("=" * 60)
    print(f"Total requests: {TOTAL_REQUESTS}, Concurrency: {CONCURRENCY}")

    health_url = api_url.replace("/predict/redis-queue", "/health").replace("/predict/vllm-native", "/health").replace("/predict/vllm-async", "/health")
    try:
        health_check = await client.get(health_url, timeout=3.0)
        if health_check.status_code != 200:
            print(f"[{name}] Health check failed with status code: {health_check.status_code}")
    except Exception as e:
        print(f"[{name}] Health check failed with exception: {e}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    start_total = time.time()

    tasks = [send_single_request(client, semaphore, payload, api_url) for payload in sample_payloads[:TOTAL_REQUESTS]]
    results = await asyncio.gather(*tasks)

    total_duration = time.time() - start_total

    latencies = [res[0] for res in results if res[1]]
    success_count = sum(1 for res in results if res[1])
    fail_count = TOTAL_REQUESTS - success_count

    if not latencies:
        print(f"[{name}] No successful requests to analyze.")
        return None
    
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg_latency = sum(latencies) / len(latencies)
    throughput = TOTAL_REQUESTS / total_duration

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_name": name,
        "api_url": api_url,
        "total_requests": TOTAL_REQUESTS,
        "concurrency": CONCURRENCY,
        "success_count": success_count,
        "fail_count": fail_count,
        "total_duration_seconds": round(total_duration, 2),
        "throughput_rps": round(throughput, 2),
        "latency_metrics_ms": {
            "average": round(avg_latency, 2),
            "p50_median": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2)
        }
    }

    print(f"\n--- BENCHMARK RESULTS ({name}) ---")
    print(f"   - Throughput (RPS)    : {report['throughput_rps']} req/sec")
    print(f"   - Average Latency     : {report['latency_metrics_ms']['average']} ms")
    print(f"   - Latency P50 (Median): {report['latency_metrics_ms']['p50_median']} ms")
    print(f"   - Latency P95         : {report['latency_metrics_ms']['p95']} ms")
    print(f"   - Latency P99         : {report['latency_metrics_ms']['p99']} ms\n")

    return report

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "api_latency_comparison_report.json")
    
    # Đọc lại báo cáo cũ nếu có để bảo lưu các endpoint khác không bị xóa mất
    reports = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                reports = json.load(f)
                print(f"[INFO] Loaded existing report from {output_file}")
        except Exception as e:
            print(f"[WARNING] Could not load existing report: {e}")
            reports = {}

    async with httpx.AsyncClient() as client:
        # 1. Benchmark Native API (Bỏ comment nếu muốn test)
        if API_URL_NATIVE:
            native_report = await run_benchmark_for_endpoint(client, "NATIVE + SEMANTIC_CACHE", API_URL_NATIVE)
            if native_report:
                reports["native + semantic_cache"] = native_report

        # 2. Benchmark Async Continuous Batching API
        # if API_URL_ASYNC:
        #     async_report = await run_benchmark_for_endpoint(client, "ASYNC_VLLM", API_URL_ASYNC)
        #     if async_report:
        #         reports["async_vllm"] = async_report

        # 3. Benchmark Redis API
        # if API_URL_REDIS:
        #     redis_report = await run_benchmark_for_endpoint(client, "REDIS", API_URL_REDIS)
        #     if redis_report:
        #         reports["redis"] = redis_report

    if reports:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=4)
        print("=" * 60)
        print(f"Benchmark comparison report saved to: {output_file}")
        print(f"Stored endpoints in report: {list(reports.keys())}")
        print("=" * 60)
    else:
        print("No benchmarks were successfully executed.")

if __name__ == "__main__":
    asyncio.run(main())