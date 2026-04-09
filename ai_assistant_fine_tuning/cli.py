"""CLI for AI Assistant Fine-Tuning."""

import logging

import click
from dotenv import load_dotenv

from ai_assistant_fine_tuning.batch_commands import batch_fetch, batch_status, batch_submit
from ai_assistant_fine_tuning.dpo_finetune import dpo_finetune
from ai_assistant_fine_tuning.sft_finetune import sft_finetune
from ai_assistant_fine_tuning.gen_dpo_td import gen_dpo_td
from ai_assistant_fine_tuning.gen_sft_td import gen_sft_td


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


cli.add_command(gen_sft_td)
cli.add_command(gen_dpo_td)
cli.add_command(batch_submit)
cli.add_command(batch_status)
cli.add_command(batch_fetch)
cli.add_command(sft_finetune)
cli.add_command(dpo_finetune)


if __name__ == "__main__":
    cli()
