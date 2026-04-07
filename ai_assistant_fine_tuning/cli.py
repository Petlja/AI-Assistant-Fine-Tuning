"""CLI for AI Assistant Fine-Tuning."""

import logging

import click
from dotenv import load_dotenv

from ai_assistant_fine_tuning.batch_commands import batch_fetch, batch_status, batch_submit
from ai_assistant_fine_tuning.dpo import dpo
from ai_assistant_fine_tuning.finetune import finetune
from ai_assistant_fine_tuning.gendpo import gendpo
from ai_assistant_fine_tuning.gentd import gentd


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


cli.add_command(gentd)
cli.add_command(gendpo)
cli.add_command(batch_submit)
cli.add_command(batch_status)
cli.add_command(batch_fetch)
cli.add_command(finetune)
cli.add_command(dpo)


if __name__ == "__main__":
    cli()
