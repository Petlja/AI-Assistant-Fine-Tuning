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

### 1) gentd

Generate chat-format SFT JSONL (`messages: system/user/assistant`) from markdown files.

```bash
astft gentd --help
```

Options:

- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`)
- `--output, -o FILE` (default: `training_data.jsonl`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--system-prompt TEXT`
- `--append / --overwrite` (default: overwrite)
- `--batch / --no-batch` (default: no-batch)
- `--batch-input FILE` (optional batch input JSONL path)
- `--manifest FILE` (optional batch manifest path)

Examples:

```bash
# Synchronous generation
astft gentd --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini --no-batch

# Batch artifact generation only (no immediate model calls)
astft gentd --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini --batch
```

### 2) gendpo

Generate DPO JSONL (`prompt/chosen/rejected`) from markdown files.

```bash
astft gendpo --help
```

Options:

- `--input-dir DIRECTORY`
- `--output, -o FILE` (default: `training_data_dpo.jsonl`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--system-prompt TEXT`
- `--append / --overwrite` (default: overwrite)
- `--batch / --no-batch` (default: no-batch)
- `--batch-input FILE`
- `--manifest FILE`

Examples:

```bash
# Synchronous generation
astft gendpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini --no-batch

# Batch artifact generation
astft gendpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini --batch
```

### 3) batch-submit

Submit a prepared batch input file (from `gentd --batch` or `gendpo --batch`) to OpenAI Batch API.

```bash
astft batch-submit --help
```

Options:

- `--manifest FILE` (required)
- `--batch-input FILE` (optional override)

Example:

```bash
astft batch-submit --manifest training_data.batch.manifest.json
```

### 4) batch-status

Check current batch status and refresh manifest state.

```bash
astft batch-status --help
```

Options:

- `--manifest FILE` (required)

Example:

```bash
astft batch-status --manifest training_data.batch.manifest.json
```

### 5) batch-fetch

Fetch completed batch output and transform into final training JSONL format.

```bash
astft batch-fetch --help
```

Options:

- `--manifest FILE` (required)
- `--output FILE` (optional override final JSONL)
- `--readable-output FILE` (optional override readable text output)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft batch-fetch --manifest training_data.batch.manifest.json --overwrite
```

### 6) finetune

Run LoRA/QLoRA SFT fine-tuning on a CUDA GPU.

```bash
astft finetune --help
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
- `--use-4bit / --no-4bit`

Example:

```bash
astft finetune --data training_data.jsonl --model Qwen/Qwen3-14B --output-dir ./finetune-output
```

### 7) dpo

Run DPO alignment from an existing SFT adapter.

```bash
astft dpo --help
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
- `--use-4bit / --no-4bit`

Example:

```bash
astft dpo --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
```

## Typical Pipelines

### Pipeline A: Synchronous generation

```bash
astft gentd --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini --no-batch
astft gendpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini --no-batch
```

### Pipeline B: Batch generation

```bash
# Create batch inputs + manifests
astft gentd --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini --batch
astft gendpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini --batch

# Submit
astft batch-submit --manifest training_data.batch.manifest.json
astft batch-submit --manifest training_data_dpo.batch.manifest.json

# Poll status
astft batch-status --manifest training_data.batch.manifest.json
astft batch-status --manifest training_data_dpo.batch.manifest.json

# Fetch final outputs when completed
astft batch-fetch --manifest training_data.batch.manifest.json --overwrite
astft batch-fetch --manifest training_data_dpo.batch.manifest.json --overwrite
```

### Pipeline C: Train and align

```bash
astft finetune --data training_data.jsonl --output-dir ./finetune-output
astft dpo --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
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
