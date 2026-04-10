"""LoRA SFT fine-tuning using TRL on a single GPU."""

from pathlib import Path

import click
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_TRAINING_DATA = "training_data.jsonl"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-14B"
DEFAULT_OUTPUT_DIR = "./finetune-output"
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


@click.command("sft-finetune")
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TRAINING_DATA,
    show_default=True,
    help="Path to chat-format JSONL training data.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_BASE_MODEL,
    show_default=True,
    help="Base model to fine-tune (HuggingFace model ID).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to save the fine-tuned adapter.",
)
@click.option("--epochs", default=3, show_default=True, help="Number of training epochs.")
@click.option("--batch-size", default=2, show_default=True, help="Per-device batch size.")
@click.option("--grad-accum", default=8, show_default=True, help="Gradient accumulation steps.")
@click.option("--lr", default=1e-4, show_default=True, help="Learning rate.")
@click.option("--max-seq-len", default=2048, show_default=True, help="Maximum sequence length.")
@click.option("--lora-r", default=16, show_default=True, help="LoRA rank.")
@click.option("--lora-alpha", default=32, show_default=True, help="LoRA alpha.")
@click.option("--lora-dropout", default=0.05, show_default=True, help="LoRA dropout.")
@click.option("--use-4bit/--no-4bit", default=False, show_default=True, help="Use 4-bit quantization (QLoRA).")
def sft_finetune(
    data: Path,
    model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    use_4bit: bool,
) -> None:
    """Run LoRA SFT fine-tuning on a base model using TRL.

    Loads chat-format JSONL training data and fine-tunes with LoRA/QLoRA
    on a single GPU (optimized for H100 95GB).
    """
    click.echo(f"Base model: {model}")
    click.echo(f"Training data: {data}")
    click.echo(f"Output dir: {output_dir}")
    click.echo(f"QLoRA 4-bit: {use_4bit}")

    if not torch.cuda.is_available():
        raise click.ClickException("Fine-tuning requires a CUDA GPU, but none was detected.")

    _ensure_gpu_has_headroom(use_4bit)

    # Load dataset
    dataset = load_dataset("json", data_files=str(data), split="train")
    click.echo(f"Loaded {len(dataset)} training examples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for QLoRA
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # This command targets a single GPU. For 4-bit loading, forcing the model
    # onto that GPU avoids `device_map="auto"` spilling modules to CPU/disk,
    # which transformers rejects for QLoRA.
    device_map = {"": torch.cuda.current_device()} if use_4bit else "auto"

    # Load base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=torch.bfloat16,
        )
    except torch.OutOfMemoryError as exc:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = _bytes_to_gib(free_bytes)
        total_gib = _bytes_to_gib(total_bytes)
        raise click.ClickException(
            "CUDA ran out of memory while loading the base model. "
            f"Current free memory: {free_gib:.2f} GiB / {total_gib:.2f} GiB. "
            "Free the GPU first, or retry with a smaller model."
        ) from exc

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
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

    # Trainer
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    click.echo("Starting training...")
    trainer.train()

    # Save adapter
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    click.echo(f"\nDone. Adapter saved to {output_dir}")
