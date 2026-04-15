"""Step 2: Answer generated questions using a local vLLM model."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

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
@click.option(
    "--concurrency",
    type=click.IntRange(1, 64),
    default=8,
    show_default=True,
    help="Number of parallel requests to the local model.",
)
def gen_answers(
    questions: Path,
    output: Path,
    base_url: str,
    model: str,
    api_key: str,
    append: bool,
    concurrency: int,
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
    click.echo(f"Concurrency: {concurrency}")
    click.echo(f"Output: {output}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    txt_output = output.with_suffix(".txt")
    if not append:
        output.write_text("", encoding="utf-8")
        txt_output.write_text("", encoding="utf-8")

    # Flatten all questions into work items
    work_items = []
    for rec in question_records:
        source = rec.get("source", "unknown")
        system_message = rec.get("system_message", "")
        for q in rec.get("questions", []):
            work_items.append((source, system_message, q))

    write_lock = Lock()
    total_answers = 0
    total_errors = 0

    def _answer_one(item):
        source, system_message, q = item
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": q},
            ],
            max_completion_tokens=8000,
            temperature=0.5,
        )
        return {
            "source": source,
            "system_message": system_message,
            "question": q,
            "answer": completion.choices[0].message.content or "",
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_answer_one, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                record = future.result()
                with write_lock:
                    write_jsonl([record], output, append=True)
                    write_readable_answers([record], txt_output, append=True)
                    total_answers += 1
                    if total_answers % 100 == 0:
                        click.echo(f"  Progress: {total_answers}/{len(work_items)}")
            except Exception as exc:
                total_errors += 1
                source = futures[future][0]
                logger.warning("Failed for %s: %s", source, exc)

    click.echo(f"\nDone. Generated {total_answers} answers, {total_errors} errors -> {output}")
