#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Qwen2.5-VL-7B vLLM 서빙 스크립트 (GCP L4 GPU 인스턴스용)
#
# 사용법:
#   chmod +x vllm_serve.sh
#   ./vllm_serve.sh
#
# 사전 조건:
#   1. pip install -r requirements.txt
#   2. python download_models.py  (최초 1회)
# ─────────────────────────────────────────────────────────────

set -e

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PORT=8001
HOST="0.0.0.0"

echo "🚀 Starting vLLM server..."
echo "   Model: ${MODEL}"
echo "   Port:  ${PORT}"

vllm serve "${MODEL}" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port "${PORT}" \
  --host "${HOST}"
