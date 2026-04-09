"""Prepare OpenAI Batch API input + manifest for SFT or DPO generation."""

import logging
from pathlib import Path

import click

from ai_assistant_fine_tuning.batch_util import (
    MANIFEST_SCHEMA_VERSION,
    create_chat_batch_request,
    make_custom_id,
    save_manifest,
    short_sha1,
    utc_now_iso,
    write_jsonl,
)
from ai_assistant_fine_tuning.data_gen import (
    DEFAULT_DPO_OUTPUT_FILE,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_SFT_OUTPUT_FILE,
    DEFAULT_SYSTEM_PROMPT,
    TASK_CONFIG,
)

logger = logging.getLogger(__name__)


@click.command("gen-batch")
@click.option(
    "--task",
    type=click.Choice(["sft", "dpo"], case_sensitive=False),
    required=True,
    help="Type of training data to generate: 'sft' or 'dpo'.",
)
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
    "--system-prompt",
    default=DEFAULT_SYSTEM_PROMPT,
    show_default=True,
    help="System prompt for training examples.",
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
    output: Path | None,
    model: str,
    system_prompt: str,
    batch_input: Path | None,
    manifest: Path | None,
) -> None:
    """Prepare an OpenAI Batch API input file and manifest for bulk generation.

    Scans INPUT_DIR for *.md files and builds batch requests for either
    SFT Q/A generation or DPO preference generation. After running this,
    submit the batch with 'astft batch-submit'.
    """
    default_output = Path(DEFAULT_SFT_OUTPUT_FILE if task == "sft" else DEFAULT_DPO_OUTPUT_FILE)
    output = output or default_output

    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Task: {task}")
    click.echo(f"Found {len(md_files)} markdown file(s) in {input_dir}")
    click.echo(f"Teacher model: {model}")

    cfg = TASK_CONFIG[task]
    batch_input_path = batch_input or output.with_suffix(".batch.input.jsonl")
    manifest_path = manifest or output.with_suffix(".batch.manifest.json")

    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    batch_records: list[dict] = []
    manifest_requests: list[dict] = []

    for md_file in md_files:
        rel = md_file.relative_to(input_dir)
        content = md_file.read_text(encoding="utf-8")
        if len(content.strip()) < 50:
            continue

        custom_id = make_custom_id(task, rel, content)
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
                "source_rel_path": rel.as_posix(),
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
            "task": task,
            "model": model,
            "input_dir": str(input_dir),
            "system_prompt": system_prompt,
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
