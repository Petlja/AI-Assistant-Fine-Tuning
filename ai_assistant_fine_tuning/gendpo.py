"""Generate DPO preference data from documentation using a teacher model.

Reads markdown documentation files and uses an OpenAI teacher model
to produce prompt/chosen/rejected triples in JSONL for DPO training.
"""

import json
import logging
from pathlib import Path

import click
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
DEFAULT_OUTPUT_FILE = "training_data_dpo.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for teachers. "
    "Help teachers prepare lessons clearly, accurately, and concisely "
    "based on the provided study material."
)

GENERATION_PROMPT = """\
You are an expert educator and evaluator. Given the following documentation/study material, \
generate high-quality preference training samples for DPO.

Each sample must include:
- prompt: a realistic teacher request (self-contained)
- chosen: the better answer, accurate and useful
- rejected: a plausible but worse answer (less complete, less precise, or slightly misleading), \
  but not unsafe and not nonsense

Requirements:
- Generate between 3 and 10 samples depending on material richness.
- Keep both answers in the same language as the source material.
- Ensure chosen is clearly better than rejected.
- Keep prompt grounded in the provided material.

--- MATERIAL ---
{content}
--- END MATERIAL ---
"""


class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str


class PreferencePairList(BaseModel):
    pairs: list[PreferencePair]


def generate_preference_pairs(
    client: OpenAI,
    content: str,
    model: str,
) -> PreferencePairList:
    """Call the teacher model to generate DPO preference pairs from content."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You generate preference training data.",
            },
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(content=content),
            },
        ],
        response_format=PreferencePairList,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def format_for_dpo(
    preference_pairs: PreferencePairList,
    system_prompt: str,
) -> list[dict]:
    """Convert preference pairs to DPO records.

    Each record: {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    return [
        {
            "prompt": f"{system_prompt}\n\n{pair.prompt}" if system_prompt else pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        for pair in preference_pairs.pairs
    ]


def write_jsonl(records: list[dict], output_path: Path, append: bool = True) -> None:
    """Append records to a JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_readable(
    preference_pairs: PreferencePairList,
    output_path: Path,
    source: str,
    append: bool = True,
) -> None:
    """Append DPO preference samples to a human-readable text file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(f"=== Source: {source} ===\n\n")
        for j, pair in enumerate(preference_pairs.pairs, 1):
            f.write(f"Prompt {j}: {pair.prompt}\n")
            f.write(f"Chosen {j}: {pair.chosen}\n")
            f.write(f"Rejected {j}: {pair.rejected}\n\n")
        f.write("\n")


@click.command("gendpo")
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
    help="Output JSONL file path for DPO records.",
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
    help="System prompt to prefix into each generated DPO prompt.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output file instead of overwriting.",
)
def gendpo(
    input_dir: Path,
    output: Path,
    model: str,
    system_prompt: str,
    append: bool,
) -> None:
    """Generate DPO preference data from documentation using a teacher model.

    Recursively finds all *.md files in INPUT_DIR, sends each to the
    OpenAI teacher model to produce prompt/chosen/rejected triples, and
    writes JSONL suitable for DPO training with TRL.
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
            preference_pairs = generate_preference_pairs(client, content, model)
            records = format_for_dpo(preference_pairs, system_prompt)
            write_jsonl(records, output, append=True)
            write_readable(preference_pairs, txt_output, source=str(rel), append=True)
            total_pairs += len(records)
            click.echo(f"{len(records)} preference pairs")
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to process %s: %s", rel, exc)

    click.echo(f"\nDone. Generated {total_pairs} DPO training examples -> {output}")
