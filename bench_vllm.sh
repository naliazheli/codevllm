#!/bin/bash
set -e
URL="$1"
for i in 1 2 3; do
  curl -s -o /dev/null -w "run${i} time_total=%{time_total}\n" \
    "$URL/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.5","prompt":"Hello","max_tokens":32}'
done
