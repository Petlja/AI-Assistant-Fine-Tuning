"""CLI for AI Assistant Fine-Tuning."""

import logging

import click
from dotenv import load_dotenv

from ai_assistant_fine_tuning.finetune_dpo import dpo_finetune
from ai_assistant_fine_tuning.finetune_sft import sft_finetune
from ai_assistant_fine_tuning.gen_answers import gen_answers
from ai_assistant_fine_tuning.gen_questions import gen_questions
from ai_assistant_fine_tuning.judge import judge
from ai_assistant_fine_tuning.make_td import make_dpo, make_sft


@click.group()
@click.version_option()
def cli() -> None:
    """AI Assistant Fine-Tuning CLI."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# Data pipeline. Steps 1-3 cost API calls; make-sft and make-dpo cost nothing and
# can be re-run in any shape over the file step 3 produced.
cli.add_command(gen_questions)    # Step 1: two passes -> one question per segment
cli.add_command(gen_answers)      # Step 2: the local model answers them
cli.add_command(judge)            # Step 3: teacher scores, then repairs or rewrites
cli.add_command(make_sft)         # Build an SFT file from the judged file
cli.add_command(make_dpo)         # Build DPO pairs from the judged file
# Training. Run one or the other -- not chained.
cli.add_command(sft_finetune)
cli.add_command(dpo_finetune)


if __name__ == "__main__":
    cli()
