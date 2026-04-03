import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/completions")
    parser.add_argument("--model", default="qwen3.5-dense")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    print(json.dumps(body, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

