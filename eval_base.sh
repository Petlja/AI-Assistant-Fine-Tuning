#!/usr/bin/env bash
# Evaluate the BASE model directly via vLLM backend (no server needed).
#
# Usage:
#   ./eval_base.sh
#
# Override any config via env vars, e.g.:
#   LIMIT=200 TASKS=arc_challenge ./eval_base.sh
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval-results}"

TASKS="${TASKS:-arc_challenge,hellaswag,mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

# LIMIT controls dataset truncation.
# - unset   -> default smoke test size (500)
# - empty   -> full dataset
# - number  -> explicit cap
LIMIT="${LIMIT-500}"

GPU_MEM="${GPU_MEM:-0.85}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
# ──────────────────────────────────────────────────────────────────────────────

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

mkdir -p "$OUTPUT_DIR"

echo "==> Evaluating BASE model: $BASE_MODEL"
echo "    Tasks: $TASKS | Few-shot: $NUM_FEWSHOT | Limit: ${LIMIT:-full}"

uv run lm_eval run \
    --model vllm \
    --model_args "pretrained=${BASE_MODEL},dtype=bfloat16,gpu_memory_utilization=${GPU_MEM},tensor_parallel_size=${TENSOR_PARALLEL},max_model_len=4096,trust_remote_code=True" \
    --tasks "$TASKS" \
    --num_fewshot "$NUM_FEWSHOT" \
    --gen_kwargs temperature=0 top_p=1 \
    --output_path "${OUTPUT_DIR}/base" \
    --log_samples \
    $LIMIT_ARG

echo "==> Base model results saved to ${OUTPUT_DIR}/base/"
