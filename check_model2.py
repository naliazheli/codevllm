import json
c = json.load(open('/root/code/Qwen3.5-35B-A3B/config.json'))
# Print all keys
for k, v in sorted(c.items()):
    if not isinstance(v, dict):
        print(f"{k}: {v}")
