"""Step 2: Answer generated questions using a local vLLM model."""

import logging
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batch_util import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_QUESTIONS_OUTPUT_FILE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    write_readable_answers,
)

logger = logging.getLogger(__name__)


@click.command("gen-answers")
@click.option(
    "--questions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_QUESTIONS_OUTPUT_FILE,
    show_default=True,
    help="Input JSONL file with generated questions (from gen-questions).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ANSWERS_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file path for answers.",
)
@click.option(
    "--base-url",
    default=DEFAULT_VLLM_BASE_URL,
    show_default=True,
    help="Base URL of the local OpenAI-compatible API (vLLM).",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_LOCAL_MODEL,
    show_default=True,
    help="Model name served by the local endpoint.",
)
@click.option(
    "--api-key",
    default=DEFAULT_VLLM_API_KEY,
    show_default=True,
    help="API key for the local endpoint.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output file instead of overwriting.",
)
def gen_answers(
    questions: Path,
    output: Path,
    base_url: str,
    model: str,
    api_key: str,
    append: bool,
) -> None:
    """Answer questions using a local vLLM model.

    Reads questions JSONL (from gen-questions), sends each question to the
    local model with the context-aware system message, and writes one
    answer record per question.
    """
    question_records = read_jsonl(questions)
    if not question_records:
        click.echo(f"No records found in {questions}")
        raise SystemExit(1)

    total_questions = sum(len(r.get("questions", [])) for r in question_records)
    click.echo(f"Loaded {len(question_records)} source(s), {total_questions} question(s)")
    click.echo(f"Local model: {model} @ {base_url}")
    click.echo(f"Output: {output}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    txt_output = output.with_suffix(".txt")
    if not append:
        output.write_text("", encoding="utf-8")
        txt_output.write_text("", encoding="utf-8")

    total_answers = 0
    answer_batch: list[dict[str, str]] = []

    for rec in question_records:
        source = rec.get("source", "unknown")
        system_message = rec.get("system_message", "")
        qs = rec.get("questions", [])

        click.echo(f"[{source}] answering {len(qs)} question(s)...")

        for j, q in enumerate(qs, 1):
            click.echo(f"  Q{j}/{len(qs)} ... ", nl=False)
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": q},
                    ],
                    max_completion_tokens=8000,
                    temperature=0.5,
                )
                answer_text = completion.choices[0].message.content or ""

                answer_record = {
                    "source": source,
                    "system_message": system_message,
                    "question": q,
                    "answer": answer_text,
                }
                write_jsonl([answer_record], output, append=True)
                answer_batch.append(answer_record)
                total_answers += 1
                click.echo("ok")
            except Exception as exc:
                click.echo(f"error: {exc}")
                logger.warning("Failed Q%d for %s: %s", j, source, exc)

    # Write readable output
    if answer_batch:
        write_readable_answers(answer_batch, txt_output, append=append)

    click.echo(f"\nDone. Generated {total_answers} answers -> {output}")
