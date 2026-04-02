#!/bin/bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","prompt":"Hello","max_tokens":32}'
