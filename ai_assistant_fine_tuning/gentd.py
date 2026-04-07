"""Generate training data from documentation using a teacher model.

Reads markdown documentation files and uses an OpenAI teacher model
to produce instructional Q/A pairs in chat-format JSONL for SFT with TRL.
"""

from pathlib import Path

import json
import logging

import click
from openai import OpenAI
from pydantic import BaseModel

from ai_assistant_fine_tuning.batching import (
    MANIFEST_SCHEMA_VERSION,
    create_chat_batch_request,
    make_custom_id,
    save_manifest,
    short_sha1,
    utc_now_iso,
    write_jsonl as write_batch_jsonl,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
DEFAULT_OUTPUT_FILE = "training_data.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for teachers. "
    "Help teachers prepare lessons clearly, accurately, and concisely "
    "based on the provided study material."
)

GENERATION_PROMPT = """\
You are an expert educator. Given the following documentation/study material, \
generate a list of high-quality instructional question-answer pairs that could \
be used to fine-tune an AI assistant that helps teachers prepare lessons.

Requirements:
- Generate between 3 and 10 Q/A pairs depending on the length and richness of the material.
- Questions should cover key concepts, definitions, explanations, examples, and practical teaching use.
- Questions should be self-contained and written from a teacher's perspective preparing a lesson.
- Answers should be detailed, accurate, and based solely on the provided material.
- Vary question types: lesson planning, conceptual explanation, comparison, classroom application.
- Use the same language as the source material.

--- MATERIAL ---
{content}
--- END MATERIAL ---
"""


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairList(BaseModel):
    pairs: list[QAPair]


def generate_qa_pairs(
    client: OpenAI,
    content: str,
    model: str,
) -> QAPairList:
    """Call the teacher model to generate Q/A pairs from content."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You generate training data.",
            },
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(content=content),
            },
        ],
        response_format=QAPairList,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def format_for_trl(
    qa_pairs: QAPairList,
    system_prompt: str,
) -> list[dict]:
    """Convert Q/A pairs to TRL SFT chat-format records.

    Each record: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair.question},
                {"role": "assistant", "content": pair.answer},
            ]
        }
        for pair in qa_pairs.pairs
    ]


def write_jsonl(records: list[dict], output_path: Path, append: bool = True) -> None:
    """Append records to a JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_readable(
    qa_pairs: QAPairList, output_path: Path, source: str, append: bool = True,
) -> None:
    """Append Q/A pairs to a human-readable text file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(f"=== Source: {source} ===\n\n")
        for j, pair in enumerate(qa_pairs.pairs, 1):
            f.write(f"Q{j}: {pair.question}\n")
            f.write(f"A{j}: {pair.answer}\n\n")
        f.write("\n")


@click.command("gentd")
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
    default=DEFAULT_OUTPUT_FILE,
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
@click.option(
    "--batch/--no-batch",
    default=False,
    show_default=True,
    help="Create OpenAI Batch API input + manifest instead of calling the model directly.",
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
def gentd(
    input_dir: Path,
    output: Path,
    model: str,
    system_prompt: str,
    append: bool,
    batch: bool,
    batch_input: Path | None,
    manifest: Path | None,
) -> None:
    """Generate training data from documentation using a teacher model.

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

    if batch:
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

            custom_id = make_custom_id("gentd", rel, content)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate training data. "
                        "Return ONLY a valid JSON object with a 'pairs' key."
                    ),
                },
                {
                    "role": "user",
                    "content": GENERATION_PROMPT.format(content=content),
                },
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

        write_batch_jsonl(batch_records, batch_input_path, append=False)
        save_manifest(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "task": "gentd",
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
        return

    client = OpenAI()

    total_pairs = 0
    txt_output = output.with_suffix(".txt")
    if not append:
        # Truncate output files
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
            write_readable(qa_pairs, txt_output, source=str(rel), append=True)
            total_pairs += len(records)
            click.echo(f"{len(records)} pairs")
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_pairs} training examples -> {output}")
