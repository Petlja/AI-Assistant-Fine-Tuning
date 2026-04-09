#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# smoke_test_sync.sh – 3-step sync data generation + finetuning
#
# Requirements:
#   OPENAI_API_KEY   – for gen-questions, gen-td
#   vLLM running     – for gen-answers (localhost:8000)
#   CUDA GPU         – for sft-finetune, dpo-finetune (optional)
#
# Configurable env vars (with defaults):
#   DATASET          – input markdown directory
#   MODEL            – teacher model (Steps 1 & 3)
#   LOCAL_MODEL      – local model served by vLLM (Step 2)
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
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen3-14B}"
FINETUNE_MODEL="${FINETUNE_MODEL:-Qwen/Qwen3-14B}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

mkdir -p "$OUT_DIR"

if [ "$SKIP_GEN" != "1" ]; then
  echo "=== Step 1/5  gen-questions ==="
  astft gen-questions \
    --input-dir "$DATASET" \
    --output "$OUT_DIR/questions.jsonl" \
    --model "$MODEL" \
    --overwrite
  echo "  -> $(wc -l < "$OUT_DIR/questions.jsonl") source(s) with questions"

  echo "=== Step 2/5  gen-answers ==="
  astft gen-answers \
    --questions "$OUT_DIR/questions.jsonl" \
    --output "$OUT_DIR/answers.jsonl" \
    --model "$LOCAL_MODEL" \
    --overwrite
  echo "  -> $(wc -l < "$OUT_DIR/answers.jsonl") answers"

  echo "=== Step 3/5  gen-td ==="
  astft gen-td \
    --answers "$OUT_DIR/answers.jsonl" \
    --sft-output "$OUT_DIR/sft.jsonl" \
    --dpo-output "$OUT_DIR/dpo.jsonl" \
    --model "$MODEL" \
    --threshold 7 \
    --overwrite
  echo "  -> $(wc -l < "$OUT_DIR/sft.jsonl") SFT rows, $(wc -l < "$OUT_DIR/dpo.jsonl") DPO rows"
else
  echo "=== Skipping generation steps (SKIP_GEN=1) ==="
fi

if [ "$SKIP_FINETUNE" != "1" ]; then
  SFT_DATA="$OUT_DIR/sft.jsonl"
  DPO_DATA="$OUT_DIR/dpo.jsonl"

  if [ ! -f "$SFT_DATA" ] || [ ! -s "$SFT_DATA" ]; then
    echo "  ! No SFT data found - skipping finetuning"
    exit 0
  fi

  echo "=== Step 4/5  sft-finetune ==="
  astft sft-finetune \
    --data "$SFT_DATA" \
    --model "$FINETUNE_MODEL" \
    --output-dir "$OUT_DIR/sft-adapter" \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 1 \
    --max-seq-len 512

  if [ -f "$DPO_DATA" ] && [ -s "$DPO_DATA" ]; then
    echo "=== Step 5/5  dpo-finetune ==="
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
echo "=== Sync smoke test complete ==="
echo "Output in: $OUT_DIR"
