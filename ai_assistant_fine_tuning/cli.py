"""CLI for AI Assistant Fine-Tuning."""

import click

from ai_assistant_fine_tuning.finetune import finetune
from ai_assistant_fine_tuning.gentd import gentd


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    pass


cli.add_command(gentd)
cli.add_command(finetune)


@cli.command()
@click.option("--name", default="World", help="Name to greet.")
def hello(name: str) -> None:
    """Say hello."""
    click.echo(f"Hello, {name}!")


if __name__ == "__main__":
    cli()
