"""Shared domain library for training data generation.

Contains Pydantic models, generation prompts, OpenAI teacher-model callers,
format converters, and human-readable writers for both SFT and DPO pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
DEFAULT_SFT_OUTPUT_FILE = "training_data.jsonl"
DEFAULT_DPO_OUTPUT_FILE = "training_data_dpo.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for teachers. "
    "Help teachers prepare lessons clearly, accurately, and concisely "
    "based on the provided study material."
)

SFT_GENERATION_PROMPT = """\
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

DPO_GENERATION_PROMPT = """\
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

# Task-specific config used by both sync and batch paths.
TASK_CONFIG = {
    "sft": {
        "system_message": "You generate training data.",
        "batch_system_message": (
            "You generate training data. "
            "Return ONLY a valid JSON object with a 'pairs' key."
        ),
        "generation_prompt": SFT_GENERATION_PROMPT,
    },
    "dpo": {
        "system_message": "You generate preference training data.",
        "batch_system_message": (
            "You generate preference training data. "
            "Return ONLY a valid JSON object with a 'pairs' key."
        ),
        "generation_prompt": DPO_GENERATION_PROMPT,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairList(BaseModel):
    pairs: list[QAPair]


class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str


class PreferencePairList(BaseModel):
    pairs: list[PreferencePair]


# ---------------------------------------------------------------------------
# Teacher-model callers (sync)
# ---------------------------------------------------------------------------


def generate_qa_pairs(client: OpenAI, content: str, model: str) -> QAPairList:
    """Call the teacher model to generate SFT Q/A pairs from content."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": TASK_CONFIG["sft"]["system_message"]},
            {"role": "user", "content": SFT_GENERATION_PROMPT.format(content=content)},
        ],
        response_format=QAPairList,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def generate_preference_pairs(client: OpenAI, content: str, model: str) -> PreferencePairList:
    """Call the teacher model to generate DPO preference pairs from content."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": TASK_CONFIG["dpo"]["system_message"]},
            {"role": "user", "content": DPO_GENERATION_PROMPT.format(content=content)},
        ],
        response_format=PreferencePairList,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def format_for_trl(qa_pairs: QAPairList, system_prompt: str) -> list[dict]:
    """Convert Q/A pairs to TRL SFT chat-format records."""
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


def format_for_dpo(preference_pairs: PreferencePairList, system_prompt: str) -> list[dict]:
    """Convert preference pairs to DPO records."""
    return [
        {
            "prompt": f"{system_prompt}\n\n{pair.prompt}" if system_prompt else pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        for pair in preference_pairs.pairs
    ]


# ---------------------------------------------------------------------------
# Batch-output transformers (operate on raw dicts from Batch API JSON)
# ---------------------------------------------------------------------------


def transform_sft_batch(json_obj: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Transform raw batch JSON into SFT records + readable pairs."""
    pairs = json_obj.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("SFT batch output missing 'pairs' list")

    records: list[dict[str, Any]] = []
    readable: list[tuple[str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        readable.append((question, answer))

    if not records:
        raise ValueError("SFT batch output had no valid question-answer pairs")

    return records, readable


def transform_dpo_batch(json_obj: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Transform raw batch JSON into DPO records + readable triples."""
    pairs = json_obj.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("DPO batch output missing 'pairs' list")

    records: list[dict[str, Any]] = []
    readable: list[tuple[str, str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            continue

        records.append(
            {
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        readable.append((prompt, chosen, rejected))

    if not records:
        raise ValueError("DPO batch output had no valid preference pairs")

    return records, readable


# ---------------------------------------------------------------------------
# Human-readable writers
# ---------------------------------------------------------------------------


def write_readable_sft(
    qa_pairs: QAPairList, output_path: Path, source: str, append: bool = True,
) -> None:
    """Append SFT Q/A pairs to a human-readable text file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(f"=== Source: {source} ===\n\n")
        for j, pair in enumerate(qa_pairs.pairs, 1):
            f.write(f"Q{j}: {pair.question}\n")
            f.write(f"A{j}: {pair.answer}\n\n")
        f.write("\n")


def write_readable_sft_raw(items: list[dict[str, Any]], output_path: Path, append: bool) -> None:
    """Write SFT readable output from raw batch-transformed data."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            f.write(f"=== Source: {source} ===\n\n")
            for i, (question, answer) in enumerate(item["pairs"], 1):
                f.write(f"Q{i}: {question}\n")
                f.write(f"A{i}: {answer}\n\n")
            f.write("\n")


def write_readable_dpo(
    preference_pairs: PreferencePairList, output_path: Path, source: str, append: bool = True,
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


def write_readable_dpo_raw(items: list[dict[str, Any]], output_path: Path, append: bool) -> None:
    """Write DPO readable output from raw batch-transformed data."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            f.write(f"=== Source: {source} ===\n\n")
            for i, (prompt, chosen, rejected) in enumerate(item["pairs"], 1):
                f.write(f"Prompt {i}: {prompt}\n")
                f.write(f"Chosen {i}: {chosen}\n")
                f.write(f"Rejected {i}: {rejected}\n\n")
            f.write("\n")
