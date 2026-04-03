import json
import urllib.request
req = urllib.request.Request(
    'http://localhost:8001/v1/completions',
    data=json.dumps({'model': 'qwen3.5', 'prompt': 'Hello', 'max_tokens': 8}).encode(),
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(req, timeout=60).read().decode()
print(resp)
