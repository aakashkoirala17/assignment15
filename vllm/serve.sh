#!/usr/bin/env bash
# vLLM High-Performance Local Model Serving Script
# Serves Open-Source LLMs (Llama 3 8B, Mistral 7B) with PagedAttention
# ==============================================================================

set -euo pipefail

MODEL_ID=${1:-"meta-llama/Meta-Llama-3-8B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

echo "=========================================================="
echo " Starting vLLM OpenAI-Compatible Server"
echo " Model:               $MODEL_ID"
echo " Endpoint:            http://$HOST:$PORT/v1"
echo " GPU Mem Utilization: $GPU_MEMORY_UTILIZATION"
echo " Max Context Length:  $MAX_MODEL_LEN tokens"
echo " Tensor Parallelism:  $TENSOR_PARALLEL_SIZE GPU(s)"
echo "=========================================================="

# Check if CUDA is available, otherwise prompt CPU / Apple Silicon guidance
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Launching vLLM with FlashAttention & PagedAttention..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_ID" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --dtype bfloat16 \
        --trust-remote-code
else
    echo "[NOTICE] No NVIDIA CUDA device found in current shell."
    echo "[INFO] For Apple Silicon (Mac) or CPU-only development, run Ollama or llama.cpp:"
    echo "       ollama run llama3"
    echo "       export VLLM_BASE_URL='http://localhost:11434/v1'"
    echo "[INFO] Alternatively, start the Mock fallback provider in config.py."
fi
