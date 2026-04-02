import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        latency = time.perf_counter() - start
        parsed = json.loads(body.decode("utf-8"))
        usage = parsed.get("usage") or {}
        return {
            "ok": True,
            "latency": latency,
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        latency = time.perf_counter() - start
        return {
            "ok": False,
            "latency": latency,
            "error": str(e),
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p))))
    return sorted_values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--total", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    wall_start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                one_request,
                args.url,
                args.model,
                args.prompt,
                args.max_tokens,
                args.timeout,
            )
            for _ in range(args.total)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_time = time.perf_counter() - wall_start

    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    latencies = sorted(r["latency"] for r in oks)
    completion_tokens = sum(r["completion_tokens"] for r in oks)
    total_tokens = sum(r["total_tokens"] for r in oks)

    summary = {
        "url": args.url,
        "model": args.model,
        "prompt_len": len(args.prompt),
        "max_tokens": args.max_tokens,
        "total_requests": args.total,
        "concurrency": args.concurrency,
        "success": len(oks),
        "errors": len(errs),
        "wall_time_sec": round(wall_time, 4),
        "req_per_sec": round(len(oks) / wall_time if wall_time > 0 else 0.0, 4),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_toks_per_sec": round(completion_tokens / wall_time if wall_time > 0 else 0.0, 4),
        "lat_avg_sec": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "lat_p50_sec": round(percentile(latencies, 0.50), 4) if latencies else 0.0,
        "lat_p95_sec": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "lat_p99_sec": round(percentile(latencies, 0.99), 4) if latencies else 0.0,
    }

    print(json.dumps(summary, ensure_ascii=False))
    if errs:
        print(json.dumps({"sample_error": errs[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
