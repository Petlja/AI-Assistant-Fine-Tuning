# AI-Assistant-Fine-Tuning


## Running Commands

You can run commands in either of these ways:

1. Installed script entrypoint:

```bash
astft <command> [options]
```

## Top-Level CLI

```bash
astft --help
astft --version
```

## Commands


### 1) gen-questions

**Step 1** — Generate questions and context-aware system messages from markdown lecture files using a teacher model.

```bash
astft gen-questions --help
```

Options:

- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`)
- `--output-dir, -o DIRECTORY` (default: `finetunedata/questions_batch/`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-questions --input-dir ./dataset/etit1_prog --output-dir finetunedata/questions_batch/ --model gpt-4o-mini
```

Output files (inside output dir):
- `questions.jsonl` — one line per source file
- `questions.txt` — human-readable summary

Example JSONL:
```json
{"source": "uvod/programski_jezici.md", "system_message": "You are a helpful assistant for a C programming course...", "questions": ["What is the difference between a compiler and an interpreter?", ...]}
```

### 2) gen-answers

**Step 2** — Answer questions using a locally-served model (vLLM).

```bash
astft gen-answers --help
```

Options:

- `--questions FILE` (default: `questions.jsonl`)
- `--output, -o FILE` (default: `answers.jsonl`)
- `--base-url TEXT` (default: `http://localhost:8000/v1`)
- `--model, -m TEXT` (default: `Qwen/Qwen3-14B`)
- `--api-key TEXT` (default: `EMPTY`)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-answers --questions questions.jsonl --model Qwen/Qwen3-14B
```

Output format (`answers.jsonl` — one line per question):

```json
{"source": "uvod/programski_jezici.md", "system_message": "...", "question": "What is the difference...", "answer": "A compiler translates..."}
```

### 3) gen-td

**Step 3** — Generate training data by evaluating answers using a teacher model.
Good answers (score ≥ threshold) become SFT training data (using teacher's
improved answer); bad answers become DPO training data (rejected = original,
chosen = teacher-improved version).

```bash
astft gen-td --help
```

Options:

- `--answers FILE` (default: `answers.jsonl`)
- `--sft-output FILE` (default: `training_data.jsonl`)
- `--dpo-output FILE` (default: `training_data_dpo.jsonl`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--threshold INT` (default: `70`, range 1-100)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-td --answers answers.jsonl --threshold 70 --model gpt-4o-mini
```

generation (Step 1) or evaluation (Step 3).

### 4) gen-batch

Prepare OpenAI Batch API input and manifest for asynchronous question generation (Step 1) or evaluation (Step 3).

```bash
astft gen-batch --help
```

Options:

- `--task [questions|evaluate]` (required)
- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`, for task=questions)
- `--answers FILE` (for task=evaluate, default: `answers.jsonl`)
- `--output-dir, -o DIRECTORY` (default depends on task)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--threshold INT` (default: `70`, stored in manifest for evaluate task)
- `--batch-input FILE` (optional batch input JSONL path)
- `--manifest FILE` (optional batch manifest path)

Example:

```bash
# Batch question generation
astft gen-batch --task questions --input-dir ./dataset/etit1_prog --output-dir finetunedata/questions_batch/ --model gpt-4o-mini
# Batch evaluation
astft gen-batch --task evaluate --answers answers.jsonl --output-dir finetunedata/evaluate_batch/ --model gpt-4o-mini
```

Output files (inside output dir):
- `questions.batch.input.jsonl` or `training_data.batch.input.jsonl`
- `questions.batch.manifest.json` or `training_data.batch.manifest.json`

### 5) batch-submit

Submit a prepared batch input file (from `gen-batch`) to OpenAI Batch API.

```bash
astft batch-submit --help
```

Options:

- `--manifest FILE` (required)
- `--batch-input FILE` (optional override)

Example:

```bash
astft batch-submit --manifest questions.batch.manifest.json
```


### 6) fetch-batch

Check batch status and fetch results if complete.

```bash
astft fetch-batch --help
```

Options:

- `--manifest FILE` (required)
- `--output-dir DIRECTORY` (optional override for output directory)
- `--threshold INT` (optional, overrides manifest value for evaluate task)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
# Fetch questions batch results
astft fetch-batch --manifest finetunedata/questions_batch/questions.batch.manifest.json --output-dir finetunedata/questions_batch/
# Fetch evaluate batch results (with threshold override)
astft fetch-batch --manifest finetunedata/evaluate_batch/training_data.batch.manifest.json --output-dir finetunedata/training_data/t5/ --threshold 5
```

Output files (inside output dir):
- For questions: `questions.jsonl`, `questions.txt`
- For evaluate: `sft_training_data.jsonl`, `dpo_training_data.jsonl`, `evaluation.txt`

### 7) sft-finetune

Run LoRA SFT fine-tuning on a CUDA GPU.

```bash
astft sft-finetune --help
```

Key options:

- `--data FILE` (default: `training_data.jsonl`)
- `--model, -m TEXT` (default: `Qwen/Qwen3-14B`)
- `--output-dir, -o DIRECTORY` (default: `./finetune-output`)
- `--epochs INT`
- `--batch-size INT`
- `--grad-accum INT`
- `--lr FLOAT`
- `--max-seq-len INT`
- `--lora-r INT`
- `--lora-alpha INT`
- `--lora-dropout FLOAT`
- `--use-4bit / --no-4bit` (default: no-4bit)

Example:

```bash
astft sft-finetune --data training_data.jsonl --model Qwen/Qwen3-14B --output-dir ./finetune-output
```

### 8) dpo-finetune

Run DPO alignment from an existing SFT adapter.

```bash
astft dpo-finetune --help
```

Required options:

- `--adapter DIRECTORY`
- `--training-data FILE`

Other options:

- `--model TEXT` (optional base model override)
- `--output-dir, -o DIRECTORY` (default: `./dpo-output`)
- `--epochs INT`
- `--batch-size INT`
- `--grad-accum INT`
- `--lr FLOAT`
- `--max-seq-len INT`
- `--beta FLOAT`
- `--use-4bit / --no-4bit` (default: no-4bit)

Example:

```bash
astft dpo-finetune --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
```

## Typical Pipelines


### Pipeline A: Synchronous 3-step generation

```bash
# Step 1: Generate questions from lectures (teacher model)
astft gen-questions --input-dir ./dataset/etit1_prog --output-dir finetunedata/questions_batch/ --model gpt-4o-mini

# Step 2: Answer questions (local vLLM model — start vLLM first with serve.sh)
astft gen-answers --questions finetunedata/questions_batch/questions.jsonl --model Qwen/Qwen3-14B --output finetunedata/answers/answers.jsonl

# Step 3: Generate training data — evaluate & route (teacher model)
astft gen-td --answers finetunedata/answers/answers.jsonl --threshold 70 --model gpt-4o-mini
```


### Pipeline B: Batch generation (Steps 1 and 3 via OpenAI Batch API)

```bash
# Step 1 via batch
astft gen-batch --task questions --input-dir ./dataset/etit1_prog --output-dir finetunedata/questions_batch/ --model gpt-4o-mini
astft batch-submit --manifest finetunedata/questions_batch/questions.batch.manifest.json
astft fetch-batch --manifest finetunedata/questions_batch/questions.batch.manifest.json --output-dir finetunedata/questions_batch/

# Step 2: local model (always sync)
astft gen-answers --questions finetunedata/questions_batch/questions.jsonl --model Qwen/Qwen3-14B --output finetunedata/answers/answers.jsonl

# Step 3 via batch
astft gen-batch --task evaluate --answers finetunedata/answers/answers.jsonl --output-dir finetunedata/evaluate_batch/ --model gpt-4o-mini
astft batch-submit --manifest finetunedata/evaluate_batch/training_data.batch.manifest.json
astft fetch-batch --manifest finetunedata/evaluate_batch/training_data.batch.manifest.json --output-dir finetunedata/training_data/t5/ --threshold 5
```

### Pipeline C: Train and align

```bash
astft sft-finetune --data training_data.jsonl --output-dir ./finetune-output
astft dpo-finetune --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
```

## Serving

`serve.sh` starts vLLM with LoRA adapter enabled.

```bash
bash serve.sh
```

Environment variables used by `serve.sh`:

- `BASE_MODEL` (default: `Qwen/Qwen3-14B`)
- `ADAPTER_PATH` (default: `./finetune-output`)
- `ADAPTER_NAME` (default: `Qwen3-14B-ft`)
- `MAX_LORA_RANK` (default: `16`)
- `PORT` (default: `8000`)
