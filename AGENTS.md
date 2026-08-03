# AI Assistant Fine-Tuning — Agent Instructions

This document helps AI coding agents understand the command flow, data structures, and processing patterns in this fine-tuning pipeline.

## Project Overview

The **AI Assistant Fine-Tuning** project generates training datasets for model fine-tuning by:
1. Extracting questions from educational markdown content
2. Generating answers using a local model (vLLM)
3. Evaluating answers and routing them to SFT (Supervised Fine-Tuning) or DPO (Direct Preference Optimization) training datasets

The pipeline supports **synchronous** (direct) and **asynchronous** (OpenAI Batch API) workflows for Steps 1 and 3.

---

## Command Flow & Data I/O

The pipeline consists of **7 CLI commands** (via `astft` entrypoint, defined in [cli.py](ai_assistant_fine_tuning/cli.py)).

### **SYNCHRONOUS PIPELINE (3-step)**

#### **Step 1: gen-questions**
- **Purpose:** Generate Q&A pairs from markdown documentation
- **Model:** Teacher model (OpenAI, default: `gpt-4o-mini`)
- **Input:**
  - `--input-dir` (default: `../data-preprocessing/dataset/`) — Directory containing `*.md` files recursively
- **Output:**
  - `questions.jsonl` — One JSONL line per source file
    ```json
    {
      "source": "relative/path/file.md",
      "system_message": "You are a helpful assistant for...",
      "questions": ["Q1", "Q2", ...]
    }
    ```
  - `questions.txt` — Human-readable summary
- **Options:** `--model`, `--output-dir`, `--append/--overwrite`
- **Key File:** [gen_questions.py](ai_assistant_fine_tuning/gen_questions.py)

#### **Step 2: gen-answers**
- **Purpose:** Answer questions using a local model (vLLM) — **usually runs once, output reused**
- **Model:** Local model served at `--base-url` (default: `http://localhost:8000/v1`)
- **Input:**
  - `--questions FILE` (default: `questions.jsonl`) — From Step 1
- **Output:**
  - `answers.jsonl` — One JSONL line per question
    ```json
    {
      "source": "relative/path/file.md",
      "system_message": "...",
      "question": "Q1",
      "answer": "Model-generated answer..."
    }
    ```
- **Options:** `--model`, `--output`, `--base-url`, `--api-key`, `--append/--overwrite`
- **Key File:** [gen_answers.py](ai_assistant_fine_tuning/gen_answers.py)

#### **Step 3: gen-td** (Generate Training Data)
- **Purpose:** Evaluate answers and route to SFT/DPO datasets
  - Score ≥ threshold → SFT (uses teacher's improved answer)
  - Score < threshold → DPO (rejected=original, chosen=improved)
- **Model:** Teacher model
- **Input:**
  - `--answers FILE` (default: `answers.jsonl`) — From Step 2
- **Output:**
  - `training_data.jsonl` — SFT examples (good answers)
    ```json
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "Teacher's improved answer"}
      ]
    }
    ```
  - `training_data_dpo.jsonl` — DPO examples (comparison pairs)
    ```json
    {
      "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "Q"}],
      "chosen": {"role": "assistant", "content": "Teacher's improved answer"},
      "rejected": {"role": "assistant", "content": "Original answer"}
    }
    ```
  - `evaluation.txt` — Statistics (scores, splits, etc.)
- **Options:** `--model`, `--threshold` (default: 7, range 1-10), `--sft-output`, `--dpo-output`, `--append/--overwrite`
- **Key File:** [gen_td.py](ai_assistant_fine_tuning/gen_td.py)

---

### **BATCH API PIPELINE (asynchronous, Steps 1 & 3)**

When processing **large datasets** (100k+ questions), use the Batch API for cost savings and async processing.

#### **Step 1b: gen-batch** (Prepare Batch Input)
- **Purpose:** Format questions/evaluation as OpenAI Batch API input
- **Input:**
  - For questions: `--input-dir` (same as Step 1)
  - For evaluation: `--answers FILE` (from Step 2, same as Step 3)
- **Output:**
  - `questions.batch.input.jsonl` or `training_data.batch.input.jsonl` — Batch API requests
  - `questions.batch.manifest.json` or `training_data.batch.manifest.json` — Metadata + processing info
    ```json
    {
      "schema_version": "1.0",
      "task": "questions|evaluate",
      "batch_id": null,
      "model": "gpt-4o-mini",
      "input_file": "questions.batch.input.jsonl",
      "created_at": "...",
      "threshold": 7,
      "status": "prepared"
    }
    ```
- **Options:** `--task` (required: `questions` or `evaluate`), `--model`, `--threshold` (for evaluate), `--output-dir`
- **Key File:** [gen_batch.py](ai_assistant_fine_tuning/gen_batch.py)

#### **Step 2b: batch-submit**
- **Purpose:** Upload and submit batch to OpenAI API
- **Input:**
  - `--manifest FILE` (required) — From Step 1b
  - Reads batch input file referenced in manifest
- **Output:**
  - Updates manifest with `batch_id`, `status: "submitted"`, server response metadata
- **Options:** `--batch-input FILE` (override default from manifest)
- **Key File:** [batch.py](ai_assistant_fine_tuning/batch.py) → `batch_submit()`

#### **Step 3b: fetch-batch**
- **Purpose:** Poll batch status and retrieve results when complete
- **Input:**
  - `--manifest FILE` (required) — From Step 2b
- **Output:**
  - Updates manifest with `status: "completed"`
  - Writes same files as sync pipeline:
    - For questions: `questions.jsonl`, `questions.txt`
    - For evaluate: `sft_training_data.jsonl`, `dpo_training_data.jsonl`, `evaluation.txt`
  - Output directory can override with `--output-dir`
- **Options:** `--manifest`, `--output-dir`, `--threshold` (override for evaluate), `--append/--overwrite`
- **Key File:** [batch.py](ai_assistant_fine_tuning/batch.py) → `fetch_batch()`

---

### **FINE-TUNING (Training)**

#### **sft-finetune**
- **Purpose:** Run LoRA SFT fine-tuning on CUDA GPU
- **Input:**
  - `--training-data FILE` (default: `training_data.jsonl`) — From gen-td Step 3
  - HF model (download on first run)
- **Output:**
  - Model checkpoint in `--output-dir` (default: `finetune-output/t7-sft/`)
  - Training logs, eval metrics
- **Key File:** [finetune_sft.py](ai_assistant_fine_tuning/finetune_sft.py)

#### **dpo-finetune**
- **Purpose:** Run LoRA DPO fine-tuning on CUDA GPU
- **Input:**
  - `--training-data FILE` (default: `training_data_dpo.jsonl`) — From gen-td Step 3
  - Base model (SFT checkpoint or pretrained)
- **Output:**
  - Model checkpoint in `--output-dir` (default: `finetune-output/t7-dpo/`)
- **Key File:** [finetune_dpo.py](ai_assistant_fine_tuning/finetune_dpo.py)

---

## Key Data Structures

### JSONL Records

**questions.jsonl** (Step 1 output):
```json
{
  "source": "str — relative path to markdown file",
  "system_message": "str — context-aware system prompt",
  "questions": ["list of question strings"]
}
```

**answers.jsonl** (Step 2 output):
```json
{
  "source": "str",
  "system_message": "str",
  "question": "str",
  "answer": "str — model-generated answer"
}
```

**training_data.jsonl** (SFT, Step 3 output):
```json
{
  "messages": [
    {"role": "system", "content": "str"},
    {"role": "user", "content": "str"},
    {"role": "assistant", "content": "str — teacher-improved answer"}
  ]
}
```

**training_data_dpo.jsonl** (DPO, Step 3 output):
```json
{
  "prompt": [
    {"role": "system", "content": "str"},
    {"role": "user", "content": "str"}
  ],
  "chosen": {"role": "assistant", "content": "str — teacher-improved"},
  "rejected": {"role": "assistant", "content": "str — original model answer"}
}
```

---

## Default Paths & Conventions

| Variable | Default | Purpose |
|----------|---------|---------|
| Input dataset | `../data-preprocessing/dataset/` | Markdown lecture files (etit1_prog, gim1_rac_ikt, etc.) |
| Questions output | `finetunedata/questions_batch/` | Step 1 output |
| Answers input | `answers.jsonl` | Step 2 input/output |
| Training data (SFT) | `training_data.jsonl` | Step 3 SFT output |
| Training data (DPO) | `training_data_dpo.jsonl` | Step 3 DPO output |
| Batch output dir | Depends on task | `questions.batch.input.jsonl`, etc. |
| Fine-tune output | `finetune-output/{t7-sft,t7-dpo,v1}/` | Checkpoints |

---

## Common Workflows

### **Workflow A: Full Synchronous Pipeline (Small Dataset)**
```bash
# Step 1: Generate questions from dataset
astft gen-questions --input-dir ./dataset/etit1_prog --output-dir finetunedata/questions_batch/ --model gpt-4o-mini

# Step 2: Answer using local model (requires vLLM server running)
astft gen-answers --questions finetunedata/questions_batch/questions.jsonl --output answers.jsonl --model Qwen/Qwen3-14B

# Step 3: Evaluate and generate SFT/DPO training data
astft gen-td --answers answers.jsonl --sft-output training_data.jsonl --dpo-output training_data_dpo.jsonl --threshold 7

# Fine-tune on generated data
astft sft-finetune --training-data training_data.jsonl
astft dpo-finetune --training-data training_data_dpo.jsonl
```

### **Workflow B: Batch API Pipeline (Large Dataset)**
```bash
# Step 1: Prepare batch for questions
astft gen-batch --task questions --input-dir ./dataset/etit1_prog --model gpt-4o-mini

# Step 2: Submit batch (async)
astft batch-submit --manifest finetunedata/questions_batch/questions.batch.manifest.json

# Step 3: Poll and fetch when ready (use in CI/cron job)
astft fetch-batch --manifest finetunedata/questions_batch/questions.batch.manifest.json

# Step 4: Once questions.jsonl is ready, run gen-answers (Step 2 of sync pipeline)
astft gen-answers --questions finetunedata/questions_batch/questions.jsonl --output answers.jsonl

# Step 5: Prepare batch for evaluation
astft gen-batch --task evaluate --answers answers.jsonl --model gpt-4o-mini

# Step 6: Submit and fetch evaluation batch
astft batch-submit --manifest finetunedata/evaluate_batch/training_data.batch.manifest.json
astft fetch-batch --manifest finetunedata/evaluate_batch/training_data.batch.manifest.json

# Now training_data.jsonl and training_data_dpo.jsonl are ready for fine-tuning
```

---

## Development Notes

### Module Organization
- **[cli.py](ai_assistant_fine_tuning/cli.py)** — Click CLI group and command registration
- **[gen_questions.py](ai_assistant_fine_tuning/gen_questions.py)** — Step 1 (sync)
- **[gen_answers.py](ai_assistant_fine_tuning/gen_answers.py)** — Step 2 (sync)
- **[gen_td.py](ai_assistant_fine_tuning/gen_td.py)** — Step 3 (sync)
- **[gen_batch.py](ai_assistant_fine_tuning/gen_batch.py)** — Batch preparation (Steps 1b/3b)
- **[batch.py](ai_assistant_fine_tuning/batch.py)** — Batch API submission/polling (Steps 2b/3b)
- **[batch_util.py](ai_assistant_fine_tuning/batch_util.py)** — Batch helpers (JSONL I/O, manifest schema)
- **[core.py](ai_assistant_fine_tuning/core.py)** — Shared logic (LLM calls, markdown collection, evaluation)
- **[finetune_sft.py](ai_assistant_fine_tuning/finetune_sft.py)** — SFT fine-tuning (Hugging Face Trainer)
- **[finetune_dpo.py](ai_assistant_fine_tuning/finetune_dpo.py)** — DPO fine-tuning (TRL)

### Configuration
- Default model: `gpt-4o-mini` (teacher), `Qwen/Qwen3-14B` (local answering)
- Evaluation threshold: 1–10 (default: 7), lower scores go to DPO
- Batch manifest schema version: 1.0 (for tracking batch state)

### Error Handling & Retry
- Uses `tenacity` for retries on transient OpenAI API failures
- Manifest updates preserve batch_id and status across invocations
- `--append` mode allows resuming interrupted runs

---

## When Adding Features

- **New command?** Add to [cli.py](ai_assistant_fine_tuning/cli.py) and define input/output clearly
- **Modify data format?** Update [batch_util.py](ai_assistant_fine_tuning/batch_util.py) schema validation and bump `MANIFEST_SCHEMA_VERSION`
- **Add LLM calls?** Keep in [core.py](ai_assistant_fine_tuning/core.py) for reuse and consistent prompting
- **Change file paths?** Update defaults in constants (e.g., `DEFAULT_QUESTIONS_OUTPUT_FILE`)
- **Test batch workflows?** Use `--batch-input` override in `batch-submit` to test with mock files
