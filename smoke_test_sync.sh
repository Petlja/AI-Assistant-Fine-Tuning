#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# smoke_test_sync.sh – sync data generation + finetuning
#
# Requirements:
#   OPENAI_API_KEY   – for gen-sft-td, gen-dpo-td
#   CUDA GPU         – for sft-finetune, dpo-finetune (optional)
#
# Configurable env vars (with defaults):
#   DATASET          – input markdown directory
#   MODEL            – teacher model for data generation
#   FINETUNE_MODEL   – base model for finetuning
#   OUT_DIR          – output directory
#   SKIP_GEN         – set to 1 to skip generation (reuse prior output)
#   SKIP_FINETUNE    – set to 1 to skip GPU fine-tuning steps
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/test/smoke/sync}"
DATASET="${DATASET:-$SCRIPT_DIR/dataset/etit1_prog/uvod}"     # 6 small .md files
MODEL="${MODEL:-gpt-4o-mini}"
FINETUNE_MODEL="${FINETUNE_MODEL:-Qwen/Qwen3-14B}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

mkdir -p "$OUT_DIR"

if [ "$SKIP_GEN" != "1" ]; then
  echo "=== 1/4  gen-sft-td ==="
  astft gen-sft-td \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/sft.jsonl" \
    --model "$MODEL" \
    --overwrite
  echo "  -> $(wc -l < "$OUT_DIR/sft.jsonl") SFT rows"

  echo "=== 2/4  gen-dpo-td ==="
  astft gen-dpo-td \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/dpo.jsonl" \
    --model "$MODEL" \
    --overwrite
  echo "  -> $(wc -l < "$OUT_DIR/dpo.jsonl") DPO rows"
else
  echo "=== Skipping generation steps (SKIP_GEN=1) ==="
fi

if [ "$SKIP_FINETUNE" != "1" ]; then
  SFT_DATA="$OUT_DIR/sft.jsonl"
  DPO_DATA="$OUT_DIR/dpo.jsonl"

  if [ ! -f "$SFT_DATA" ]; then
    SFT_DATA="$SCRIPT_DIR/test/test_sync_demo.jsonl"
    echo "  (using fallback SFT data: $SFT_DATA)"
  fi

  if [ ! -f "$DPO_DATA" ]; then
    echo "  ! No DPO data found - skipping dpo-finetune"
  fi

  echo "=== 3/4  sft-finetune ==="
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
    echo "=== 4/4  dpo-finetune ==="
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
echo "=== Sync smoke test complete ==="
echo "Output in: $OUT_DIR"
