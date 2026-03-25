"""Generate training data from documentation using a teacher model.

Reads markdown documentation files and uses an OpenAI teacher model
to produce instructional Q/A pairs in chat-format JSONL for SFT with TRL.
"""

import json
import logging
from pathlib import Path

import click
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
DEFAULT_OUTPUT_FILE = "training_data.jsonl"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant. "
    "Answer the student's question clearly, accurately, and concisely "
    "based on the provided study material."
)

GENERATION_PROMPT = """\
You are an expert educator. Given the following documentation/study material, \
generate a list of high-quality instructional question-answer pairs that could \
be used to fine-tune an AI teaching assistant.

Requirements:
- Generate between 3 and 10 Q/A pairs depending on the length and richness of the material.
- Questions should cover key concepts, definitions, explanations, and practical understanding.
- Questions should be self-contained (a student should understand the question without seeing the source material).
- Answers should be detailed, accurate, and based solely on the provided material.
- Vary question types: factual recall, conceptual understanding, comparison, application.
- Use the same language as the source material.

Respond ONLY with a JSON array of objects, each with "question" and "answer" fields.
Do not include any other text, markdown formatting, or code fences.

--- MATERIAL ---
{content}
--- END MATERIAL ---
"""


def find_markdown_files(input_dir: Path) -> list[Path]:
    """Recursively find all *.md files in the given directory."""
    files = sorted(input_dir.rglob("*.md"))
    return files


def read_file_content(path: Path) -> str:
    """Read a markdown file, returning its text content."""
    return path.read_text(encoding="utf-8")


def generate_qa_pairs(
    client: OpenAI,
    content: str,
    model: str,
) -> list[dict]:
    """Call the teacher model to generate Q/A pairs from content."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You generate training data. Respond only with valid JSON.",
            },
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(content=content),
            },
        ],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def format_for_trl(
    qa_pairs: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert Q/A pairs to TRL SFT chat-format records.

    Each record: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    records = []
    for pair in qa_pairs:
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]},
            ]
        }
        records.append(record)
    return records


def write_jsonl(records: list[dict], output_path: Path, append: bool = True) -> None:
    """Append records to a JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    md_files = find_markdown_files(input_dir)
    if not md_files:
        click.echo(f"No *.md files found in {input_dir}")
        raise SystemExit(1)

    click.echo(f"Found {len(md_files)} markdown file(s) in {input_dir}")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Output: {output}")

    client = OpenAI()

    total_pairs = 0
    if not append:
        # Truncate output file
        output.write_text("", encoding="utf-8")

    for i, md_file in enumerate(md_files, 1):
        rel = md_file.relative_to(input_dir)
        click.echo(f"[{i}/{len(md_files)}] Processing {rel} ... ", nl=False)

        content = read_file_content(md_file)
        if len(content.strip()) < 50:
            click.echo("skipped (too short)")
            continue

        try:
            qa_pairs = generate_qa_pairs(client, content, model)
            records = format_for_trl(qa_pairs, system_prompt)
            write_jsonl(records, output, append=True)
            total_pairs += len(records)
            click.echo(f"{len(records)} pairs")
        except (json.JSONDecodeError, KeyError) as exc:
            click.echo(f"parse error: {exc}")
            logger.warning("Failed to parse response for %s: %s", rel, exc)
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_pairs} training examples -> {output}")
