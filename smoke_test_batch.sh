#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# smoke_test_batch.sh – batch generation + polling fetch + finetuning
#
# Requirements:
#   OPENAI_API_KEY   – for gen-batch, batch-submit, fetch-batch
#   CUDA GPU         – for sft-finetune, dpo-finetune (optional)
#
# Configurable env vars (with defaults):
#   DATASET          – input markdown directory
#   MODEL            – teacher model for data generation
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
FINETUNE_MODEL="${FINETUNE_MODEL:-Qwen/Qwen3-14B}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FETCH_POLL_SEC="${FETCH_POLL_SEC:-30}"
FETCH_MAX_POLLS="${FETCH_MAX_POLLS:-60}"

mkdir -p "$OUT_DIR"

wait_for_both_batches() {
  local sft_manifest="$1"
  local sft_output="$2"
  local dpo_manifest="$3"
  local dpo_output="$4"
  local polls=0

  while [ "$polls" -lt "$FETCH_MAX_POLLS" ]; do
    polls=$((polls + 1))
    echo "  [batch] fetch attempt $polls/$FETCH_MAX_POLLS"

    # Fetch both batches
    astft fetch-batch \
      --manifest "$sft_manifest" \
      --output "$sft_output" \
      --overwrite || true

    astft fetch-batch \
      --manifest "$dpo_manifest" \
      --output "$dpo_output" \
      --overwrite || true

    # Check if both outputs are ready
    sft_ready=0
    dpo_ready=0

    if [ -s "$sft_output" ]; then
      sft_ready=1
      echo "    [sft] ready: $(wc -l < "$sft_output") rows"
    fi

    if [ -s "$dpo_output" ]; then
      dpo_ready=1
      echo "    [dpo] ready: $(wc -l < "$dpo_output") rows"
    fi

    if [ "$sft_ready" -eq 1 ] && [ "$dpo_ready" -eq 1 ]; then
      echo "  [batch] both outputs ready!"
      return 0
    fi

    if [ "$polls" -lt "$FETCH_MAX_POLLS" ]; then
      sleep "$FETCH_POLL_SEC"
    fi
  done

  echo "ERROR: Timed out waiting for batch outputs after $FETCH_MAX_POLLS polls" >&2
  return 1
}

if [ "$SKIP_GEN" != "1" ]; then
  echo "=== 1/6  gen-batch --task sft ==="
  astft gen-batch \
    --task sft \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/sft_batch.jsonl" \
    --model "$MODEL" \
    --batch-input "$OUT_DIR/sft_batch.batch.input.jsonl" \
    --manifest "$OUT_DIR/sft_batch.batch.manifest.json"
  echo "  -> $(wc -l < "$OUT_DIR/sft_batch.batch.input.jsonl") batch requests"

  echo "=== 2/6  gen-batch --task dpo ==="
  astft gen-batch \
    --task dpo \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/dpo_batch.jsonl" \
    --model "$MODEL" \
    --batch-input "$OUT_DIR/dpo_batch.batch.input.jsonl" \
    --manifest "$OUT_DIR/dpo_batch.batch.manifest.json"
  echo "  -> $(wc -l < "$OUT_DIR/dpo_batch.batch.input.jsonl") batch requests"

  echo "=== 3/6  batch-submit (sft + dpo) ==="
  astft batch-submit --manifest "$OUT_DIR/sft_batch.batch.manifest.json"
  astft batch-submit --manifest "$OUT_DIR/dpo_batch.batch.manifest.json"

  echo "=== 4/6  fetch-batch loop (both) ==="
  wait_for_both_batches \
    "$OUT_DIR/sft_batch.batch.manifest.json" \
    "$OUT_DIR/sft_batch.jsonl" \
    "$OUT_DIR/dpo_batch.batch.manifest.json" \
    "$OUT_DIR/dpo_batch.jsonl"
else
  echo "=== Skipping batch generation/submit/fetch (SKIP_GEN=1) ==="
fi

if [ "$SKIP_FINETUNE" != "1" ]; then
  SFT_DATA="$OUT_DIR/sft_batch.jsonl"
  DPO_DATA="$OUT_DIR/dpo_batch.jsonl"

  if [ ! -f "$SFT_DATA" ]; then
    SFT_DATA="$SCRIPT_DIR/test/test_sync_demo.jsonl"
    echo "  (using fallback SFT data: $SFT_DATA)"
  fi

  if [ ! -f "$DPO_DATA" ]; then
    echo "  ! No DPO data found - skipping dpo-finetune"
  fi

  echo "=== 5/6  sft-finetune ==="
  astft sft-finetune \
    --data "$SFT_DATA" \
    --model "$FINETUNE_MODEL" \
    --output-dir "$OUT_DIR/sft-adapter" \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --max-seq-len 512 \
    --use-4bit

  if [ -f "$DPO_DATA" ]; then
    echo "=== 6/6  dpo-finetune ==="
    astft dpo-finetune \
      --adapter "$OUT_DIR/sft-adapter" \
      --training-data "$DPO_DATA" \
      --output-dir "$OUT_DIR/dpo-adapter" \
      --epochs 1 \
      --batch-size 1 \
      --grad-accum 1 \
      --max-seq-len 512 \
      --use-4bit
  fi
else
  echo "=== Skipping fine-tuning steps (SKIP_FINETUNE=1) ==="
fi

echo ""
echo "=== Batch smoke test complete ==="
echo "Output in: $OUT_DIR"
echo ""
echo "Resume commands if fetch loop times out:"
echo "  Poll both batches manually:"
echo "    astft fetch-batch --manifest $OUT_DIR/sft_batch.batch.manifest.json --output $OUT_DIR/sft_batch.jsonl --overwrite"
echo "    astft fetch-batch --manifest $OUT_DIR/dpo_batch.batch.manifest.json --output $OUT_DIR/dpo_batch.jsonl --overwrite"
echo "  Then continue finetune:"
echo "    SKIP_GEN=1 bash $SCRIPT_DIR/smoke_test_batch.sh"
