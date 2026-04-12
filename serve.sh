#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
ADAPTER_PATH="${ADAPTER_PATH:-./finetune-output/t7-sft}"
ADAPTER_NAME="${ADAPTER_NAME:-Qwen3-14B-ft-sft}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
PORT="${PORT:-8000}"

exec uv run vllm serve "$BASE_MODEL" \
    --enable-lora \
    --tokenizer "${ADAPTER_PATH}" \
    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --port "$PORT" \
    --reasoning-parser qwen3 \
