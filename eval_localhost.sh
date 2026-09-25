#!/usr/bin/env bash
# Evaluate a model running on a local OpenAI-compatible server (e.g. vLLM).
# Checks that the server is reachable at localhost:PORT/v1/models before running.
#
# Usage:
#   ./eval_localhost.sh
#
# Override any config via env vars, e.g.:
#   MODEL_NAME=Qwen3-14B-ft-dpo ADAPTER_PATH=./finetune-output/t7-dpo ./eval_localhost.sh
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ADAPTER_PATH="${ADAPTER_PATH:-./finetune-output/t7-sft}"
MODEL_NAME="${MODEL_NAME:-Qwen3-14B-ft-sft}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval-results}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-ft-sft}"

TASKS="${TASKS:-arc_challenge,hellaswag,mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

# CONCURRENT: parallel requests to the server. Does not affect scores.
# MAX_LENGTH: prompt budget. lm-eval's local-completions default is 2048, which
# silently left-truncates long prompts (mmlu-prox) and changes the score.
CONCURRENT="${CONCURRENT:-16}"
MAX_LENGTH="${MAX_LENGTH:-8192}"

# LIMIT controls dataset truncation.
# - unset   -> default smoke test size (500)
# - empty   -> full dataset
# - number  -> explicit cap
LIMIT="${LIMIT-500}"
# ──────────────────────────────────────────────────────────────────────────────

# ── Check server reachability ──────────────────────────────────────────────────
MODELS_URL="http://localhost:${PORT}/v1/models"
echo "==> Checking server at ${MODELS_URL} ..."
if ! curl --silent --fail --max-time 5 "$MODELS_URL" > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Cannot reach ${MODELS_URL}"
    echo "       Make sure your vLLM server is running (e.g. ./serve.sh) and try again."
    exit 1
fi
echo "    Server is up."
# ──────────────────────────────────────────────────────────────────────────────

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

mkdir -p "$OUTPUT_DIR"

echo "==> Evaluating model via server: http://localhost:${PORT}/v1"
echo "    Model: $MODEL_NAME | Tasks: $TASKS | Few-shot: $NUM_FEWSHOT | Limit: ${LIMIT:-full}"

uv run lm_eval run \
    --model local-completions \
    --model_args "model=${MODEL_NAME},base_url=http://localhost:${PORT}/v1/completions,tokenizer=${ADAPTER_PATH},num_concurrent=${CONCURRENT},max_retries=3,max_length=${MAX_LENGTH}" \
    --tasks "$TASKS" \
    --num_fewshot "$NUM_FEWSHOT" \
    --gen_kwargs temperature=0,top_p=1 \
    --output_path "${OUTPUT_DIR}/${OUTPUT_SUBDIR}" \
    --log_samples \
    $LIMIT_ARG

echo "==> Results saved to ${OUTPUT_DIR}/${OUTPUT_SUBDIR}/"
