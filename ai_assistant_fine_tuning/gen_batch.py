"""Prepare OpenAI Batch API input + manifest for questions or evaluate tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

from ai_assistant_fine_tuning.batch_util import (
    MANIFEST_SCHEMA_VERSION,
    create_chat_batch_request,
    make_custom_id,
    read_jsonl,
    save_manifest,
    short_sha1,
    utc_now_iso,
)
from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_DPO_OUTPUT_FILE,
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

DEFAULT_MAX_BATCH_SIZE = 50_000


def _numbered_path(base: Path, idx: int, total: int) -> Path:
    """Return a numbered path like foo.002.ext when total > 1, else foo.ext."""
    if total <= 1:
        return base
    width = len(str(total))
    return base.with_suffix(f".{str(idx).zfill(width)}{base.suffix}")


def _append_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON line to a file (constant-memory writer)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
@click.option(
    "--max-batch-size",
    type=click.IntRange(1),
    default=DEFAULT_MAX_BATCH_SIZE,
    show_default=True,
    help="Max requests per batch file. Larger inputs are split into numbered batches.",
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
    max_batch_size: int,
) -> None:
    """Prepare an OpenAI Batch API input file and manifest.

    For task=questions: scans INPUT_DIR for *.md files and builds batch
    requests for question generation (Step 1).

    For task=evaluate: reads answers JSONL (from gen-answers) and builds
    batch requests for evaluation (Step 3).

    After running this, submit with 'astft batch-submit'.
    """
    if task == "questions":
        _gen_batch_questions(input_dir, output, model, batch_input, manifest, max_batch_size)
    elif task == "evaluate":
        if answers is None:
            answers = Path(DEFAULT_ANSWERS_OUTPUT_FILE)
            if not answers.exists():
                raise click.ClickException(
                    "No --answers file provided and default answers.jsonl not found. "
                    "Run gen-answers first."
                )
        _gen_batch_evaluate(answers, output, model, threshold, batch_input, manifest, max_batch_size)


def _gen_batch_questions(
    input_dir: Path,
    output: Path | None,
    model: str,
    batch_input: Path | None,
    manifest: Path | None,
    max_batch_size: int,
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
    base_batch_input = batch_input or output.with_suffix(".batch.input.jsonl")
    base_manifest = manifest or output.with_suffix(".batch.manifest.json")

    base_batch_input.parent.mkdir(parents=True, exist_ok=True)
    base_manifest.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect lightweight metadata (no batch_records in RAM).
    items: list[dict] = []
    for rel, md_file in sources:
        content = md_file.read_text(encoding="utf-8")
        if len(content.strip()) < 50:
            continue
        custom_id = make_custom_id("questions", Path(rel), content)
        items.append({
            "rel": rel,
            "content": content,
            "custom_id": custom_id,
        })

    if not items:
        raise click.ClickException("No eligible markdown content found for batch input.")

    # Split into chunks and write each chunk directly to disk.
    total_chunks = (len(items) + max_batch_size - 1) // max_batch_size
    for chunk_idx in range(total_chunks):
        start = chunk_idx * max_batch_size
        chunk = items[start : start + max_batch_size]

        chunk_input = _numbered_path(base_batch_input, chunk_idx + 1, total_chunks)
        chunk_manifest = _numbered_path(base_manifest, chunk_idx + 1, total_chunks)

        # Truncate files before streaming.
        chunk_input.write_text("", encoding="utf-8")

        manifest_requests: list[dict] = []
        for item in chunk:
            messages = [
                {"role": "system", "content": cfg["batch_system_message"]},
                {"role": "user", "content": cfg["generation_prompt"].format(content=item["content"])},
            ]
            _append_jsonl_line(
                chunk_input,
                create_chat_batch_request(
                    custom_id=item["custom_id"],
                    model=model,
                    messages=messages,
                    temperature=0.7,
                ),
            )
            manifest_requests.append({
                "custom_id": item["custom_id"],
                "source_rel_path": item["rel"],
                "content_sha1": short_sha1(item["content"], length=40),
                "status": "pending",
            })

        save_manifest(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "task": "questions",
                "model": model,
                "input_dir": str(input_dir),
                "batch_input_file": str(chunk_input),
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
            chunk_manifest,
        )

        click.echo(f"Batch input written: {chunk_input} ({len(chunk)} requests)")
        click.echo(f"Manifest written: {chunk_manifest}")

    if total_chunks > 1:
        click.echo(f"\nSplit into {total_chunks} batches (max {max_batch_size} per batch).")
    click.echo("Next: run 'astft batch-submit --manifest <manifest_path>' for each manifest.")


def _gen_batch_evaluate(
    answers_path: Path,
    output: Path | None,
    model: str,
    threshold: int,
    batch_input: Path | None,
    manifest: Path | None,
    max_batch_size: int,
) -> None:
    default_output = Path(DEFAULT_SFT_OUTPUT_FILE)
    output = output or default_output

    click.echo(f"Task: evaluate")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Threshold: {threshold}/10")

    cfg = TASK_CONFIG["evaluate"]
    base_batch_input = batch_input or output.with_suffix(".batch.input.jsonl")
    base_manifest = manifest or output.with_suffix(".batch.manifest.json")

    base_batch_input.parent.mkdir(parents=True, exist_ok=True)
    base_manifest.parent.mkdir(parents=True, exist_ok=True)

    # Stream-read answers and collect lightweight metadata (no full prompt in RAM).
    items: list[dict] = []
    with open(answers_path, "r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rec = json.loads(raw_line)
            source = rec.get("source", "unknown")
            system_message = rec.get("system_message", "")
            question = rec.get("question", "")
            answer = rec.get("answer", "")
            if not question.strip() or not answer.strip():
                continue
            content_for_id = f"{source}\n{question}\n{answer}"
            custom_id = make_custom_id("evaluate", Path(f"answer_{idx}"), content_for_id)
            items.append({
                "custom_id": custom_id,
                "source": source,
                "system_message": system_message,
                "question": question,
                "answer": answer,
                "content_sha1": short_sha1(content_for_id, length=40),
            })

    if not items:
        raise click.ClickException(f"No eligible answers found in {answers_path}")

    click.echo(f"Loaded {len(items)} answer(s) from {answers_path}")

    # Split into chunks and stream each chunk to disk.
    total_chunks = (len(items) + max_batch_size - 1) // max_batch_size
    for chunk_idx in range(total_chunks):
        start = chunk_idx * max_batch_size
        chunk = items[start : start + max_batch_size]

        chunk_input = _numbered_path(base_batch_input, chunk_idx + 1, total_chunks)
        chunk_manifest = _numbered_path(base_manifest, chunk_idx + 1, total_chunks)

        # Truncate files before streaming.
        chunk_input.write_text("", encoding="utf-8")

        manifest_requests: list[dict] = []
        for item in chunk:
            user_prompt = EVALUATION_PROMPT.format(
                system_message=item["system_message"],
                question=item["question"],
                answer=item["answer"],
            )
            messages = [
                {"role": "system", "content": cfg["batch_system_message"]},
                {"role": "user", "content": user_prompt},
            ]
            _append_jsonl_line(
                chunk_input,
                create_chat_batch_request(
                    custom_id=item["custom_id"],
                    model=model,
                    messages=messages,
                    temperature=0.3,
                ),
            )
            manifest_requests.append({
                "custom_id": item["custom_id"],
                "source": item["source"],
                "system_message": item["system_message"],
                "question": item["question"],
                "answer": item["answer"],
                "content_sha1": item["content_sha1"],
                "status": "pending",
            })

        save_manifest(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "task": "evaluate",
                "model": model,
                "answers_file": str(answers_path),
                "threshold": threshold,
                "batch_input_file": str(chunk_input),
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
            chunk_manifest,
        )

        click.echo(f"Batch input written: {chunk_input} ({len(chunk)} requests)")
        click.echo(f"Manifest written: {chunk_manifest}")

    if total_chunks > 1:
        click.echo(f"\nSplit into {total_chunks} batches (max {max_batch_size} per batch).")
    click.echo("Next: run 'astft batch-submit --manifest <manifest_path>' for each manifest.")
