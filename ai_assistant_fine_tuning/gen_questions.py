"""Step 1: Generate questions from documentation using a teacher model."""

import logging
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batch_util import write_jsonl
from ai_assistant_fine_tuning.core import (
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUESTIONS_OUTPUT_FILE,
    collect_markdown_sources,
    generate_questions,
    write_readable_questions,
)

logger = logging.getLogger(__name__)
MINIMUM_CONTENT_LENGTH = 50  # Skip files with very short content

@click.command("gen-questions")
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    show_default=True,
    help="Directory containing markdown documentation files.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("finetunedata/questions_batch"),
    show_default=True,
    help="Output directory for generated questions (questions.jsonl, questions.txt)",
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
    output_dir: Path,
    model: str,
    append: bool,
) -> None:
    """Generate questions + context-aware system messages from documentation.

    Recursively finds all *.md files in INPUT_DIR, sends each to the
    teacher model to produce a system message and questions, and writes
    questions JSONL (one line per source file).
    """
    sources = collect_markdown_sources(input_dir)
    if not sources:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Found {len(sources)} markdown file(s) in {input_dir}")

    click.echo(f"Teacher model: {model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "questions.jsonl"
    txt_output = output_dir / "questions.txt"
    click.echo(f"Output: {output_file}")

    client = OpenAI()

    total_questions = 0
    if not append:
        output_file.write_text("", encoding="utf-8")
        txt_output.write_text("", encoding="utf-8")


    for i, (rel, md_file) in enumerate(sources, 1):
        click.echo(f"[{i}/{len(sources)}] Processing {rel} ... ", nl=False)

        content = md_file.read_text(encoding="utf-8")
        if len(content.strip()) < MINIMUM_CONTENT_LENGTH:
            click.echo("skipped (too short)")
            continue

        try:
            result = generate_questions(client, content, model)
            record = {
                "source": rel,
                "system_message": result.system_message,
                "questions": result.questions,
            }
            write_jsonl([record], output_file, append=True)
            write_readable_questions(result, txt_output, source=rel, append=True)
            total_questions += len(result.questions)
            click.echo(f"{len(result.questions)} questions")
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_questions} questions -> {output_file}")
