import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import statistics
import time
import urllib.request


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    dt_s = time.perf_counter() - t0
    obj = json.loads(body)
    usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    return dt_s, prompt_tokens, completion_tokens


def run_round(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    requests: int,
    timeout: int,
):
    url = f"{base_url.rstrip('/')}/v1/completions"
    latencies = []
    prompt_toks = 0
    completion_toks = 0
    errors = 0
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(one_request, url, model, prompt, max_tokens, timeout)
            for _ in range(requests)
        ]
        for f in cf.as_completed(futs):
            try:
                dt_s, pt, ct = f.result()
                latencies.append(dt_s)
                prompt_toks += pt
                completion_toks += ct
            except Exception:
                errors += 1

    wall = time.perf_counter() - start
    rps = (requests - errors) / wall if wall > 0 else 0.0
    tps = completion_toks / wall if wall > 0 else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "success": requests - errors,
        "errors": errors,
        "wall_s": round(wall, 3),
        "rps": round(rps, 3),
        "output_tps": round(tps, 3),
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "p50_s": round(p50, 3),
        "p95_s": round(p95, 3),
    }


def print_summary(results):
    print("\nsummary:")
    for r in results:
        print(
            f"c={r['concurrency']:>3} "
            f"ok={r['success']:>3}/{r['requests']} "
            f"rps={r['rps']:>7} "
            f"out_tps={r['output_tps']:>9} "
            f"p50={r['p50_s']:>6}s "
            f"p95={r['p95_s']:>6}s"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.28.158.169:8010")
    ap.add_argument("--model", default="qwen3.5")
    ap.add_argument("--prompt", default="Introduce vLLM in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", default="16,32,64")
    ap.add_argument("--requests-per-level", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--warmup-rounds", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    for i in range(args.warmup_rounds):
        print(f"warmup round {i + 1}/{args.warmup_rounds}")
        for c in levels:
            _ = run_round(
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency=c,
                requests=max(8, min(32, args.requests_per_level // 4)),
                timeout=args.timeout,
            )

    run_results = []
    for i in range(args.repeats):
        print(f"measure round {i + 1}/{args.repeats}")
        results = []
        for c in levels:
            res = run_round(
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                concurrency=c,
                requests=args.requests_per_level,
                timeout=args.timeout,
            )
            results.append(res)
            print(json.dumps({"round": i + 1, **res}, ensure_ascii=False))
        run_results.append(results)

    summary_results = []
    for idx, c in enumerate(levels):
        series = [rr[idx] for rr in run_results]
        summary_results.append({
            "concurrency": c,
            "requests": args.requests_per_level,
            "success": int(statistics.median([x["success"] for x in series])),
            "errors": int(statistics.median([x["errors"] for x in series])),
            "wall_s": round(statistics.median([x["wall_s"] for x in series]), 3),
            "rps": round(statistics.median([x["rps"] for x in series]), 3),
            "output_tps": round(statistics.median([x["output_tps"] for x in series]), 3),
            "prompt_tokens": int(statistics.median([x["prompt_tokens"] for x in series])),
            "completion_tokens": int(statistics.median([x["completion_tokens"] for x in series])),
            "p50_s": round(statistics.median([x["p50_s"] for x in series]), 3),
            "p95_s": round(statistics.median([x["p95_s"] for x in series]), 3),
        })

    print("\nsummary (median across repeats):")
    print_summary(summary_results)

    if args.output_json:
        out = {
            "label": args.label,
            "started_at": started_at,
            "base_url": args.base_url,
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "concurrency": levels,
            "requests_per_level": args.requests_per_level,
            "timeout": args.timeout,
            "warmup_rounds": args.warmup_rounds,
            "repeats": args.repeats,
            "runs": run_results,
            "summary_results": summary_results,
        }
        out_path = os.path.abspath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
