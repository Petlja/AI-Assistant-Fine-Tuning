#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./serve.sh [model_output_dir]
# Example:
#   ./serve.sh t7-sft
# Environment overrides still work for all variables.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-./finetune-output}"
MODEL_OUTPUT_DIR="${1:-${MODEL_OUTPUT_DIR:-t7-sft}}"

# If the argument is a full/relative path, use it directly.
if [[ "$MODEL_OUTPUT_DIR" == /* || "$MODEL_OUTPUT_DIR" == ./* || "$MODEL_OUTPUT_DIR" == ../* ]]; then
    ADAPTER_PATH="${ADAPTER_PATH:-$MODEL_OUTPUT_DIR}"
else
    ADAPTER_PATH="${ADAPTER_PATH:-${FINETUNE_OUTPUT_DIR}/${MODEL_OUTPUT_DIR}}"
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "Error: adapter directory not found: $ADAPTER_PATH" >&2
    echo "Tip: run ./serve.sh <model_output_dir> (e.g. ./serve.sh t7-sft)" >&2
    exit 1
fi

BASE_MODEL_NAME="${BASE_MODEL##*/}"
ADAPTER_BASENAME="$(basename "$ADAPTER_PATH")"
ADAPTER_NAME="${ADAPTER_NAME:-${BASE_MODEL_NAME}-${ADAPTER_BASENAME}}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
PORT="${PORT:-8000}"

exec uv run vllm serve "$BASE_MODEL" \
    --enable-lora \
    --tokenizer "${ADAPTER_PATH}" \
    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --port "$PORT" \
    --reasoning-parser qwen3 \
