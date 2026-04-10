#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# smoke_test_batch.sh – batch generation (Steps 1 & 3) + sync Step 2 + finetuning
#
# Requirements:
#   OPENAI_API_KEY   – for gen-batch, batch-submit, fetch-batch
#   vLLM running     – for gen-answers (localhost:8000)
#   CUDA GPU         – for sft-finetune, dpo-finetune (optional)
#
# Configurable env vars (with defaults):
#   DATASET          – input markdown directory
#   MODEL            – teacher model (Steps 1 & 3)
#   LOCAL_MODEL      – local model served by vLLM (Step 2)
#   FINETUNE_MODEL   – base model for finetuning
#   OUT_DIR          – output directory
#   SKIP_GEN         – set to 1 to skip generation/submit/fetch
#   SKIP_FINETUNE    – set to 1 to skip GPU fine-tuning steps
#   FETCH_POLL_SEC   – seconds between fetch polls
#   FETCH_MAX_POLLS  – max fetch polling attempts
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/test/smoke/batch}"
DATASET="${DATASET:-$SCRIPT_DIR/dataset/etit1_prog/uvod}"     # 6 small .md files
MODEL="${MODEL:-gpt-4o-mini}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen3-14B}"
FINETUNE_MODEL="${FINETUNE_MODEL:-Qwen/Qwen3-14B}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FETCH_POLL_SEC="${FETCH_POLL_SEC:-30}"
FETCH_MAX_POLLS="${FETCH_MAX_POLLS:-60}"

mkdir -p "$OUT_DIR"

wait_for_batch() {
  local batch_manifest="$1"
  local batch_output="$2"
  local polls=0

  while [ "$polls" -lt "$FETCH_MAX_POLLS" ]; do
    polls=$((polls + 1))
    echo "  [batch] fetch attempt $polls/$FETCH_MAX_POLLS"

    astft fetch-batch \
      --manifest "$batch_manifest" \
      --output "$batch_output" \
      --overwrite || true

    if [ -s "$batch_output" ]; then
      echo "    ready: $(wc -l < "$batch_output") rows"
      return 0
    fi

    if [ "$polls" -lt "$FETCH_MAX_POLLS" ]; then
      sleep "$FETCH_POLL_SEC"
    fi
  done

  echo "ERROR: Timed out waiting for batch output after $FETCH_MAX_POLLS polls" >&2
  return 1
}

BATCH_TIMED_OUT=0

if [ "$SKIP_GEN" != "1" ]; then
  # ── Step 1: Question generation via Batch API ──
  echo "=== Step 1a  gen-batch --task questions ==="
  astft gen-batch \
    --task questions \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/questions.jsonl" \
    --model "$MODEL" \
    --batch-input "$OUT_DIR/questions.batch.input.jsonl" \
    --manifest "$OUT_DIR/questions.batch.manifest.json"
  echo "  -> $(wc -l < "$OUT_DIR/questions.batch.input.jsonl") batch requests"

  echo "=== Step 1b  batch-submit (questions) ==="
  astft batch-submit --manifest "$OUT_DIR/questions.batch.manifest.json"

  echo "=== Step 1c  fetch-batch (questions) ==="
  if ! wait_for_batch \
    "$OUT_DIR/questions.batch.manifest.json" \
    "$OUT_DIR/questions.jsonl"; then
    BATCH_TIMED_OUT=1
  fi

  if [ "$BATCH_TIMED_OUT" = "0" ]; then
    # ── Step 2: Local model answers (always sync) ──
    echo "=== Step 2  gen-answers ==="
    astft gen-answers \
      --questions "$OUT_DIR/questions.jsonl" \
      --output "$OUT_DIR/answers.jsonl" \
      --model "$LOCAL_MODEL" \
      --overwrite
    echo "  -> $(wc -l < "$OUT_DIR/answers.jsonl") answers"

    # ── Step 3: Evaluation via Batch API ──
    echo "=== Step 3a  gen-batch --task evaluate ==="
    astft gen-batch \
      --task evaluate \
      --answers "$OUT_DIR/answers.jsonl" \
      --output "$OUT_DIR/sft.jsonl" \
      --model "$MODEL" \
      --threshold 7 \
      --batch-input "$OUT_DIR/evaluate.batch.input.jsonl" \
      --manifest "$OUT_DIR/evaluate.batch.manifest.json"
    echo "  -> $(wc -l < "$OUT_DIR/evaluate.batch.input.jsonl") batch requests"

    echo "=== Step 3b  batch-submit (evaluate) ==="
    astft batch-submit --manifest "$OUT_DIR/evaluate.batch.manifest.json"

    echo "=== Step 3c  fetch-batch (evaluate) ==="
    if ! wait_for_batch \
      "$OUT_DIR/evaluate.batch.manifest.json" \
      "$OUT_DIR/sft.jsonl"; then
      BATCH_TIMED_OUT=1
    fi
  fi

  if [ "$BATCH_TIMED_OUT" = "0" ]; then
    echo "  -> SFT: $(wc -l < "$OUT_DIR/sft.jsonl") rows"
    [ -f "$OUT_DIR/training_data_dpo.jsonl" ] && echo "  -> DPO: $(wc -l < "$OUT_DIR/training_data_dpo.jsonl") rows"
  fi
else
  echo "=== Skipping batch generation/submit/fetch (SKIP_GEN=1) ==="
fi

if [ "$SKIP_FINETUNE" != "1" ] && [ "$BATCH_TIMED_OUT" = "0" ]; then
  SFT_DATA="$OUT_DIR/sft.jsonl"
  DPO_DATA="$OUT_DIR/training_data_dpo.jsonl"

  if [ ! -f "$SFT_DATA" ] || [ ! -s "$SFT_DATA" ]; then
    echo "  ! No SFT data found - skipping finetuning"
    exit 0
  fi

  echo "=== Step 4  sft-finetune ==="
  astft sft-finetune \
    --data "$SFT_DATA" \
    --model "$FINETUNE_MODEL" \
    --output-dir "$OUT_DIR/sft-adapter" \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --max-seq-len 512

  if [ -f "$DPO_DATA" ] && [ -s "$DPO_DATA" ]; then
    echo "=== Step 5  dpo-finetune ==="
    astft dpo-finetune \
      --adapter "$OUT_DIR/sft-adapter" \
      --training-data "$DPO_DATA" \
      --output-dir "$OUT_DIR/dpo-adapter" \
      --epochs 1 \
      --batch-size 1 \
      --grad-accum 1 \
      --max-seq-len 512
  else
    echo "  ! No DPO data - skipping dpo-finetune"
  fi
else
  echo "=== Skipping fine-tuning steps (SKIP_FINETUNE=1) ==="
fi

echo ""
if [ "$BATCH_TIMED_OUT" = "1" ]; then
  echo "=== Batch smoke test INCOMPLETE (fetch timed out) ==="
  echo "Output in: $OUT_DIR"
  echo ""
  echo "Resume commands:"
  echo "  astft fetch-batch --manifest $OUT_DIR/questions.batch.manifest.json --output $OUT_DIR/questions.jsonl --overwrite"
  echo "  astft fetch-batch --manifest $OUT_DIR/evaluate.batch.manifest.json --output $OUT_DIR/sft.jsonl --overwrite"
  echo "  Then continue: SKIP_GEN=1 bash $SCRIPT_DIR/smoke_test_batch.sh"
  exit 1
else
  echo "=== Batch smoke test complete ==="
  echo "Output in: $OUT_DIR"
fi
