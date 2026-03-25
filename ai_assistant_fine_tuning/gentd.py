"""Generate training data from documentation using a teacher model.

Reads markdown documentation files and uses an OpenAI teacher model
to produce instructional Q/A pairs in chat-format JSONL for SFT with TRL.
"""

import json
import logging
from pathlib import Path

import click
from openai import OpenAI
from pydantic import BaseModel

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
def gentd(
    input_dir: Path,
    output: Path,
    model: str,
    system_prompt: str,
    append: bool,
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
