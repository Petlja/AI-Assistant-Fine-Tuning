"""CLI for AI Assistant Fine-Tuning."""

import logging

import click
from dotenv import load_dotenv

from ai_assistant_fine_tuning.batch import batch_submit, fetch_batch
from ai_assistant_fine_tuning.dpo_finetune import dpo_finetune
from ai_assistant_fine_tuning.gen_batch import gen_batch
from ai_assistant_fine_tuning.gen_dpo_td import gen_dpo_td
from ai_assistant_fine_tuning.gen_sft_td import gen_sft_td
from ai_assistant_fine_tuning.sft_finetune import sft_finetune


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# Generate training data synchronously using a teacher model (no Batch API)
cli.add_command(gen_sft_td)
cli.add_command(gen_dpo_td)
# Generate training data and Batch API input/manifest for asynchronous processing
cli.add_command(gen_batch)
cli.add_command(batch_submit)
cli.add_command(fetch_batch)
# Finetuning commands
cli.add_command(sft_finetune)
cli.add_command(dpo_finetune)


if __name__ == "__main__":
    cli()
