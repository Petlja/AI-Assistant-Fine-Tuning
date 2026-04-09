"""Sync SFT training data generation from documentation."""

import logging
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batch_util import write_jsonl
from ai_assistant_fine_tuning.data_gen import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_SFT_OUTPUT_FILE,
    DEFAULT_SYSTEM_PROMPT,
    format_for_trl,
    generate_qa_pairs,
    write_readable_sft,
)

logger = logging.getLogger(__name__)


@click.command("gen-sft-td")
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    show_default=True,
    help="Directory containing markdown documentation files.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file path.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI teacher model to use.",
)
@click.option(
    "--system-prompt",
    default=DEFAULT_SYSTEM_PROMPT,
    show_default=True,
    help="System prompt to include in each training example.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output file instead of overwriting.",
)
def gen_sft_td(
    input_dir: Path,
    output: Path,
    model: str,
    system_prompt: str,
    append: bool,
) -> None:
    """Generate SFT training data from documentation using a teacher model.

    Recursively finds all *.md files in INPUT_DIR, sends each to the
    OpenAI teacher model to produce Q/A pairs, and writes chat-format
    JSONL suitable for SFT training with TRL.
    """
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Found {len(md_files)} markdown file(s) in {input_dir}")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Output: {output}")

    client = OpenAI()

    total_pairs = 0
    txt_output = output.with_suffix(".txt")
    if not append:
        output.write_text("", encoding="utf-8")
        txt_output.write_text("", encoding="utf-8")

    for i, md_file in enumerate(md_files, 1):
        rel = md_file.relative_to(input_dir)
        click.echo(f"[{i}/{len(md_files)}] Processing {rel} ... ", nl=False)

        content = md_file.read_text(encoding="utf-8")
        if len(content.strip()) < 50:
            click.echo("skipped (too short)")
            continue

        try:
            qa_pairs = generate_qa_pairs(client, content, model)
            records = format_for_trl(qa_pairs, system_prompt)
            write_jsonl(records, output, append=True)
            write_readable_sft(qa_pairs, txt_output, source=str(rel), append=True)
            total_pairs += len(records)
            click.echo(f"{len(records)} pairs")
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_pairs} training examples -> {output}")
