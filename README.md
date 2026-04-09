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

### 1) gen-sft-td

Generate chat-format SFT JSONL (`messages: system/user/assistant`) from markdown files using a teacher model synchronously.

```bash
astft gen-sft-td --help
```

Options:

- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`)
- `--output, -o FILE` (default: `training_data.jsonl`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--system-prompt TEXT`
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-sft-td --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini
```

### 2) gen-dpo-td

Generate DPO JSONL (`prompt/chosen/rejected`) from markdown files using a teacher model synchronously.

```bash
astft gen-dpo-td --help
```

Options:

- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`)
- `--output, -o FILE` (default: `training_data_dpo.jsonl`)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--system-prompt TEXT`
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft gen-dpo-td --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini
```

### 3) gen-batch

Prepare OpenAI Batch API input and manifest for asynchronous SFT or DPO training data generation.

```bash
astft gen-batch --help
```

Options:

- `--task [sft|dpo]` (required)
- `--input-dir DIRECTORY` (default: `../data-preprocessing/dataset/`)
- `--output, -o FILE` (default depends on task)
- `--model, -m TEXT` (default: `gpt-5.4-mini`)
- `--system-prompt TEXT`
- `--batch-input FILE` (optional batch input JSONL path)
- `--manifest FILE` (optional batch manifest path)

Example:

```bash
astft gen-batch --task sft --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini
astft gen-batch --task dpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini
```

### 4) batch-submit

Submit a prepared batch input file (from `gen-batch`) to OpenAI Batch API.

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

### 5) fetch-batch

Check batch status and fetch results if complete. If the batch is still running, prints current status and exits. If complete, downloads outputs and transforms them into training data JSONL.

```bash
astft fetch-batch --help
```

Options:

- `--manifest FILE` (required)
- `--output FILE` (optional override final JSONL)
- `--readable-output FILE` (optional override readable text output)
- `--append / --overwrite` (default: overwrite)

Example:

```bash
astft fetch-batch --manifest training_data.batch.manifest.json --overwrite
```

### 6) sft-finetune

Run LoRA/QLoRA SFT fine-tuning on a CUDA GPU.

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
- `--use-4bit / --no-4bit`

Example:

```bash
astft sft-finetune --data training_data.jsonl --model Qwen/Qwen3-14B --output-dir ./finetune-output
```

### 7) dpo-finetune

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
- `--use-4bit / --no-4bit`

Example:

```bash
astft dpo-finetune --adapter ./finetune-output --training-data training_data_dpo.jsonl --output-dir ./dpo-output
```

## Typical Pipelines

### Pipeline A: Synchronous generation

```bash
astft gen-sft-td --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini
astft gen-dpo-td --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini
```

### Pipeline B: Batch generation

```bash
# Create batch inputs + manifests
astft gen-batch --task sft --input-dir ./dataset/os5_inf_prog --output training_data.jsonl --model gpt-4o-mini
astft gen-batch --task dpo --input-dir ./dataset/os5_inf_prog --output training_data_dpo.jsonl --model gpt-4o-mini

# Submit
astft batch-submit --manifest training_data.batch.manifest.json
astft batch-submit --manifest training_data_dpo.batch.manifest.json

# Check status & fetch results when complete
astft fetch-batch --manifest training_data.batch.manifest.json --overwrite
astft fetch-batch --manifest training_data_dpo.batch.manifest.json --overwrite
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
