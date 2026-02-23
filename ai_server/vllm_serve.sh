#!/bin/bash
# ─────────────────────────────────────────────────────────────
# (GCP L4 GPU 인스턴스용)
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
export VLLM_USE_V1=0

MODEL="RedHatAI/gemma-3-12b-it-quantized.w8a8"
PORT=8001
HOST="0.0.0.0"

echo "🚀 Starting vLLM server..."
echo "   Model: ${MODEL}"
echo "   Port:  ${PORT}"

vllm serve "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --port "${PORT}" \
  --host "${HOST}"
