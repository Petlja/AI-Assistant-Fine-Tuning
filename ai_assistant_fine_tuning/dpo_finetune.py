"""LoRA DPO alignment using TRL on a single GPU."""

from pathlib import Path

import click
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

DEFAULT_OUTPUT_DIR = "./dpo-output"
MIN_FREE_GPU_MEMORY_GIB_4BIT = 16.0


def _bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _ensure_gpu_has_headroom(use_4bit: bool) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = _bytes_to_gib(free_bytes)
    total_gib = _bytes_to_gib(total_bytes)

    click.echo(f"GPU memory available: {free_gib:.2f} GiB free / {total_gib:.2f} GiB total")

    if use_4bit and free_gib < MIN_FREE_GPU_MEMORY_GIB_4BIT:
        raise click.ClickException(
            "Not enough free GPU memory to load the 4-bit base model. "
            f"Detected only {free_gib:.2f} GiB free on the active GPU. "
            "Stop other GPU workloads such as vLLM, or retry with a smaller base model."
        )


def _resolve_base_model_name(adapter: Path, model_override: str | None) -> str:
    if model_override:
        return model_override

    try:
        peft_config = PeftConfig.from_pretrained(str(adapter))
    except Exception as exc:
        raise click.ClickException(
            "Unable to read adapter config to infer the base model. "
            "Pass --model explicitly."
        ) from exc

    base_model_name = peft_config.base_model_name_or_path
    if not base_model_name:
        raise click.ClickException(
            "Adapter config does not contain base_model_name_or_path. "
            "Pass --model explicitly."
        )

    return base_model_name


@click.command("dpo-finetune")
@click.option(
    "--adapter",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the SFT adapter directory to continue training from.",
)
@click.option(
    "--training-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to DPO training data in JSONL format.",
)
@click.option(
    "--model",
    default=None,
    help="Optional base model override. If omitted, inferred from the adapter config.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to save the DPO adapter.",
)
@click.option("--epochs", default=1, show_default=True, help="Number of training epochs.")
@click.option("--batch-size", default=1, show_default=True, help="Per-device batch size.")
@click.option("--grad-accum", default=8, show_default=True, help="Gradient accumulation steps.")
@click.option("--lr", default=5e-6, show_default=True, help="Learning rate.")
@click.option("--max-seq-len", default=2048, show_default=True, help="Maximum sequence length.")
@click.option("--beta", default=0.1, show_default=True, help="DPO beta parameter.")
@click.option("--use-4bit/--no-4bit", default=True, show_default=True, help="Use 4-bit quantization (QLoRA).")
def dpo_finetune(
    adapter: Path,
    training_data: Path,
    model: str | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    beta: float,
    use_4bit: bool,
) -> None:
    """Run LoRA DPO alignment using an existing SFT adapter."""
    click.echo(f"Adapter: {adapter}")
    click.echo(f"Training data: {training_data}")
    click.echo(f"Output dir: {output_dir}")
    click.echo(f"QLoRA 4-bit: {use_4bit}")

    if not torch.cuda.is_available():
        raise click.ClickException("DPO training requires a CUDA GPU, but none was detected.")

    _ensure_gpu_has_headroom(use_4bit)

    base_model_name = _resolve_base_model_name(adapter, model)
    click.echo(f"Base model: {base_model_name}")

    dataset = load_dataset("json", data_files=str(training_data), split="train")
    click.echo(f"Loaded {len(dataset)} DPO training examples")

    required_columns = {"prompt", "chosen", "rejected"}
    missing_columns = sorted(required_columns.difference(set(dataset.column_names)))
    if missing_columns:
        raise click.ClickException(
            "DPO training data must contain columns: prompt, chosen, rejected. "
            f"Missing: {', '.join(missing_columns)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = {"": torch.cuda.current_device()} if use_4bit else "auto"

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        train_model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=True)
    except torch.OutOfMemoryError as exc:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = _bytes_to_gib(free_bytes)
        total_gib = _bytes_to_gib(total_bytes)
        raise click.ClickException(
            "CUDA ran out of memory while loading model/adapter for DPO. "
            f"Current free memory: {free_gib:.2f} GiB / {total_gib:.2f} GiB. "
            "Free the GPU first, or retry with a smaller model."
        ) from exc

    train_model.config.use_cache = False

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        max_length=max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = DPOTrainer(
        model=train_model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    click.echo("Starting DPO training...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    click.echo(f"\nDone. DPO adapter saved to {output_dir}")
