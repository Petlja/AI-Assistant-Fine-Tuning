"""Step 1: Generate questions from documentation using a teacher model."""

import logging
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batch_util import write_jsonl
from ai_assistant_fine_tuning.data_gen import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUESTIONS_OUTPUT_FILE,
    generate_questions,
    write_readable_questions,
)

logger = logging.getLogger(__name__)


@click.command("gen-questions")
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
    default=DEFAULT_QUESTIONS_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file path for generated questions.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI teacher model to use.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output file instead of overwriting.",
)
def gen_questions(
    input_dir: Path,
    output: Path,
    model: str,
    append: bool,
) -> None:
    """Generate questions + context-aware system messages from documentation.

    Recursively finds all *.md files in INPUT_DIR, sends each to the
    teacher model to produce a system message and questions, and writes
    questions JSONL (one line per source file).
    """
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Found {len(md_files)} markdown file(s) in {input_dir}")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Output: {output}")

    client = OpenAI()

    total_questions = 0
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
            result = generate_questions(client, content, model)
            record = {
                "source": str(rel),
                "system_message": result.system_message,
                "questions": result.questions,
            }
            write_jsonl([record], output, append=True)
            write_readable_questions(result, txt_output, source=str(rel), append=True)
            total_questions += len(result.questions)
            click.echo(f"{len(result.questions)} questions")
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_questions} questions -> {output}")
