import json
c = json.load(open('/root/code/Qwen3.5-35B-A3B/config.json'))
keys = ['architectures','model_type','num_experts','num_experts_per_tok','hidden_size','num_hidden_layers','num_attention_heads']
for k in keys:
    print(f"{k}: {c.get(k)}")
