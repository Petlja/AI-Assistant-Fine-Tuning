"""CLI for AI Assistant Fine-Tuning."""

import logging

import click
from dotenv import load_dotenv

from ai_assistant_fine_tuning.batch import batch_submit, fetch_batch
from ai_assistant_fine_tuning.finetune_dpo import dpo_finetune
from ai_assistant_fine_tuning.gen_td import gen_td
from ai_assistant_fine_tuning.gen_answers import gen_answers
from ai_assistant_fine_tuning.gen_batch import gen_batch
from ai_assistant_fine_tuning.gen_questions import gen_questions
from ai_assistant_fine_tuning.finetune_sft import sft_finetune
from ai_assistant_fine_tuning.sample_eval import sample_eval


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# 3-step training data pipeline (sync)
cli.add_command(gen_questions)    # Step 1: teacher generates questions
cli.add_command(gen_answers)      # Step 2: local model answers questions
cli.add_command(gen_td)           # Step 3: teacher scores & routes to SFT/DPO
# Batch API support for Steps 1 and 3
cli.add_command(gen_batch)
cli.add_command(batch_submit)
cli.add_command(fetch_batch)
# Human evaluation sampling
cli.add_command(sample_eval)      # Sample questions for human validation
# Finetuning commands
cli.add_command(sft_finetune)
cli.add_command(dpo_finetune)


if __name__ == "__main__":
    cli()
