"""Prepare OpenAI Batch API input + manifest for questions or evaluate tasks."""

import logging
from pathlib import Path

import click

from ai_assistant_fine_tuning.batch_util import (
    MANIFEST_SCHEMA_VERSION,
    create_chat_batch_request,
    make_custom_id,
    read_jsonl,
    save_manifest,
    short_sha1,
    utc_now_iso,
    write_jsonl,
)
from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUESTIONS_OUTPUT_FILE,
    DEFAULT_SFT_OUTPUT_FILE,
    DEFAULT_THRESHOLD,
    EVALUATION_PROMPT,
    TASK_CONFIG,
    collect_markdown_sources,
)

logger = logging.getLogger(__name__)


@click.command("gen-batch")
@click.option(
    "--task",
    type=click.Choice(["questions", "evaluate"], case_sensitive=False),
    required=True,
    help="Type of batch job: 'questions' (Step 1) or 'evaluate' (Step 3).",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    show_default=True,
    help="Directory with markdown files (for task=questions).",
)
@click.option(
    "--answers",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Answers JSONL from gen-answers (required for task=evaluate).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Final output JSONL path (default depends on task).",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI teacher model to use.",
)
@click.option(
    "--threshold",
    type=click.IntRange(1, 10),
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Score threshold for SFT/DPO routing (stored in manifest, used by fetch-batch).",
)
@click.option(
    "--batch-input",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for generated Batch API input JSONL.",
)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for generated batch manifest JSON.",
)
def gen_batch(
    task: str,
    input_dir: Path,
    answers: Path | None,
    output: Path | None,
    model: str,
    threshold: int,
    batch_input: Path | None,
    manifest: Path | None,
) -> None:
    """Prepare an OpenAI Batch API input file and manifest.

    For task=questions: scans INPUT_DIR for *.md files and builds batch
    requests for question generation (Step 1).

    For task=evaluate: reads answers JSONL (from gen-answers) and builds
    batch requests for evaluation (Step 3).

    After running this, submit with 'astft batch-submit'.
    """
    if task == "questions":
        _gen_batch_questions(input_dir, output, model, batch_input, manifest)
    elif task == "evaluate":
        if answers is None:
            answers = Path(DEFAULT_ANSWERS_OUTPUT_FILE)
            if not answers.exists():
                raise click.ClickException(
                    "No --answers file provided and default answers.jsonl not found. "
                    "Run gen-answers first."
                )
        _gen_batch_evaluate(answers, output, model, threshold, batch_input, manifest)


def _gen_batch_questions(
    input_dir: Path,
    output: Path | None,
    model: str,
    batch_input: Path | None,
    manifest: Path | None,
) -> None:
    default_output = Path(DEFAULT_QUESTIONS_OUTPUT_FILE)
    output = output or default_output

    sources = collect_markdown_sources(input_dir)
    if not sources:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Task: questions")
    click.echo(f"Found {len(sources)} markdown file(s) in {input_dir}")
    click.echo(f"Teacher model: {model}")

    cfg = TASK_CONFIG["questions"]
    batch_input_path = batch_input or output.with_suffix(".batch.input.jsonl")
    manifest_path = manifest or output.with_suffix(".batch.manifest.json")

    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    batch_records: list[dict] = []
    manifest_requests: list[dict] = []

    for rel, md_file in sources:
        content = md_file.read_text(encoding="utf-8")
        if len(content.strip()) < 50:
            continue

        custom_id = make_custom_id("questions", Path(rel), content)
        messages = [
            {"role": "system", "content": cfg["batch_system_message"]},
            {"role": "user", "content": cfg["generation_prompt"].format(content=content)},
        ]
        batch_records.append(
            create_chat_batch_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                temperature=0.7,
            )
        )
        manifest_requests.append(
            {
                "custom_id": custom_id,
                "source_rel_path": rel,
                "content_sha1": short_sha1(content, length=40),
                "status": "pending",
            }
        )

    if not batch_records:
        raise click.ClickException("No eligible markdown content found for batch input.")

    write_jsonl(batch_records, batch_input_path, append=False)
    save_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task": "questions",
            "model": model,
            "input_dir": str(input_dir),
            "batch_input_file": str(batch_input_path),
            "final_output_file": str(output),
            "final_readable_file": str(output.with_suffix(".txt")),
            "created_at": utc_now_iso(),
            "batch": {
                "id": None,
                "status": "not_submitted",
                "input_file_id": None,
                "output_file_id": None,
                "error_file_id": None,
            },
            "requests": manifest_requests,
        },
        manifest_path,
    )

    click.echo(f"Batch input written: {batch_input_path}")
    click.echo(f"Manifest written: {manifest_path}")
    click.echo("Next: run 'astft batch-submit --manifest <manifest_path>'")


def _gen_batch_evaluate(
    answers_path: Path,
    output: Path | None,
    model: str,
    threshold: int,
    batch_input: Path | None,
    manifest: Path | None,
) -> None:
    default_output = Path(DEFAULT_SFT_OUTPUT_FILE)
    output = output or default_output

    answer_records = read_jsonl(answers_path)
    if not answer_records:
        raise click.ClickException(f"No records found in {answers_path}")

    click.echo(f"Task: evaluate")
    click.echo(f"Loaded {len(answer_records)} answer(s) from {answers_path}")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Threshold: {threshold}/10")

    cfg = TASK_CONFIG["evaluate"]
    batch_input_path = batch_input or output.with_suffix(".batch.input.jsonl")
    manifest_path = manifest or output.with_suffix(".batch.manifest.json")

    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    batch_records: list[dict] = []
    manifest_requests: list[dict] = []

    for idx, rec in enumerate(answer_records):
        source = rec.get("source", "unknown")
        system_message = rec.get("system_message", "")
        question = rec.get("question", "")
        answer = rec.get("answer", "")

        if not question.strip() or not answer.strip():
            continue

        content_for_id = f"{source}\n{question}\n{answer}"
        custom_id = make_custom_id("evaluate", Path(f"answer_{idx}"), content_for_id)

        user_prompt = EVALUATION_PROMPT.format(
            system_message=system_message,
            question=question,
            answer=answer,
        )
        messages = [
            {"role": "system", "content": cfg["batch_system_message"]},
            {"role": "user", "content": user_prompt},
        ]
        batch_records.append(
            create_chat_batch_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                temperature=0.3,
            )
        )
        manifest_requests.append(
            {
                "custom_id": custom_id,
                "source": source,
                "system_message": system_message,
                "question": question,
                "answer": answer,
                "content_sha1": short_sha1(content_for_id, length=40),
                "status": "pending",
            }
        )

    if not batch_records:
        raise click.ClickException("No eligible answers found for batch evaluation.")

    write_jsonl(batch_records, batch_input_path, append=False)
    save_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task": "evaluate",
            "model": model,
            "answers_file": str(answers_path),
            "threshold": threshold,
            "batch_input_file": str(batch_input_path),
            "final_sft_file": str(output),
            "final_dpo_file": str(Path(DEFAULT_DPO_OUTPUT_FILE)),
            "final_readable_file": str(output.with_name("evaluation.txt")),
            "created_at": utc_now_iso(),
            "batch": {
                "id": None,
                "status": "not_submitted",
                "input_file_id": None,
                "output_file_id": None,
                "error_file_id": None,
            },
            "requests": manifest_requests,
        },
        manifest_path,
    )

    click.echo(f"Batch input written: {batch_input_path}")
    click.echo(f"Manifest written: {manifest_path}")
    click.echo("Next: run 'astft batch-submit --manifest <manifest_path>'")
