import json
c = json.load(open('/root/code/Qwen3.5-35B-A3B/config.json'))
print('layer_types:', c.get('layer_types'))
tc = c.get('text_config')
if isinstance(tc, dict):
    print('text_config.layer_types:', tc.get('layer_types'))
else:
    print('text_config:', tc)
print('has_linear_attention:', 'linear_attention' in (c.get('layer_types') or []))
