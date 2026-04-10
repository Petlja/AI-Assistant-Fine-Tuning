"""Shared domain library for the 3-step training data pipeline.

Step 1 – Question generation:  teacher model reads lecture markdown and produces
         a context-aware system message + list of questions.
Step 2 – Answer generation:    local vLLM model answers those questions.
Step 3 – Evaluation & routing: teacher model scores each answer 1-10,
         produces an improved version, and the caller routes results into
         SFT (score >= threshold) or DPO (score < threshold) training data.

Contains Pydantic models, generation prompts, OpenAI teacher-model callers,
format converters, batch transformers, and human-readable writers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
DEFAULT_QUESTIONS_OUTPUT_FILE = "questions.jsonl"
DEFAULT_ANSWERS_OUTPUT_FILE = "answers.jsonl"
DEFAULT_SFT_OUTPUT_FILE = "training_data.jsonl"
DEFAULT_DPO_OUTPUT_FILE = "training_data_dpo.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-14B"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"
DEFAULT_THRESHOLD = 7

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for teachers. "
    "Help teachers prepare lessons clearly, accurately, and concisely "
    "based on the provided study material."
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """\
You are an expert educator. Given the following lecture / study material, \
generate:

1. A short **system_message** (1-3 sentences) that sets the context for an AI \
assistant answering questions about this specific lecture topic.  The system \
message must mention the subject area so that every question makes sense on \
its own (e.g., if the lecture is about Python, the system message should say so, \
so a question about "printing to stdout" will be answered with `print()`, \
not `cout`).

2. A list of **questions** a teacher might ask while preparing a lesson from \
this material.

Requirements:
- Generate between 3 and 15 questions depending on the length and richness of \
the material.
- Questions must be self-contained — a reader seeing only the system message \
and the question should understand what is being asked.
- Cover key concepts, definitions, explanations, comparisons, examples, and \
practical teaching use.
- Vary question types: factual recall, conceptual explanation, comparison, \
classroom-application.
- Use the same language as the source material.

--- MATERIAL ---
{content}
--- END MATERIAL ---
"""

EVALUATION_PROMPT = """\
You are an expert educator and evaluator.

Context (system message that was given to the answering assistant):
{system_message}

Question:
{question}

Answer to evaluate:
{answer}

Score the answer on a scale from 1 to 10:
- 1-3: incorrect, irrelevant, or dangerously misleading
- 4-6: partially correct but missing important details or slightly inaccurate
- 7-8: mostly correct and useful, minor issues
- 9-10: excellent, accurate, well-structured

Then provide an **improved_answer** that would deserve a 9 or 10.

Return both the numeric score and the improved answer.
"""

# Task-specific config used by both sync and batch paths.
TASK_CONFIG = {
    "questions": {
        "system_message": "You generate questions from study material.",
        "batch_system_message": (
            "You generate questions from study material. "
            "Return ONLY a valid JSON object with 'system_message' and 'questions' keys."
        ),
        "generation_prompt": QUESTION_GENERATION_PROMPT,
    },
    "evaluate": {
        "system_message": "You evaluate and improve answers.",
        "batch_system_message": (
            "You evaluate and improve answers. "
            "Return ONLY a valid JSON object with 'score' and 'improved_answer' keys."
        ),
        "generation_prompt": EVALUATION_PROMPT,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuestionList(BaseModel):
    """Output of Step 1: context-aware system message + questions per source."""

    system_message: str
    questions: list[str]


class EvaluationResult(BaseModel):
    """Output of Step 3: teacher's score and improved answer for one Q/A."""

    score: int = Field(ge=1, le=10)
    improved_answer: str


# ---------------------------------------------------------------------------
# Teacher-model callers (sync)
# ---------------------------------------------------------------------------


def generate_questions(client: OpenAI, content: str, model: str) -> QuestionList:
    """Call the teacher model to generate questions + system message from content."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": TASK_CONFIG["questions"]["system_message"]},
            {"role": "user", "content": QUESTION_GENERATION_PROMPT.format(content=content)},
        ],
        response_format=QuestionList,
        temperature=0.7,
    )
    return response.choices[0].message.parsed


def evaluate_answer(
    client: OpenAI,
    system_message: str,
    question: str,
    answer: str,
    model: str,
) -> EvaluationResult:
    """Call the teacher model to score an answer and provide an improved version."""
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": TASK_CONFIG["evaluate"]["system_message"]},
            {
                "role": "user",
                "content": EVALUATION_PROMPT.format(
                    system_message=system_message,
                    question=question,
                    answer=answer,
                ),
            },
        ],
        response_format=EvaluationResult,
        temperature=0.3,
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Routing: evaluation result -> SFT or DPO record
# ---------------------------------------------------------------------------


def route_evaluation(
    system_message: str,
    question: str,
    answer: str,
    evaluation: EvaluationResult,
    threshold: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Route a single evaluation into an SFT record, a DPO record, or neither.

    Returns (sft_record, dpo_record).  Exactly one will be non-None.
    """
    if evaluation.score >= threshold:
        sft_record = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": evaluation.improved_answer},
            ]
        }
        return sft_record, None
    else:
        dpo_record = {
            "prompt": f"{system_message}\n\n{question}",
            "chosen": evaluation.improved_answer,
            "rejected": answer,
        }
        return None, dpo_record


# ---------------------------------------------------------------------------
# Batch-output transformers
# ---------------------------------------------------------------------------


def transform_questions_batch(json_obj: dict[str, Any]) -> tuple[str, list[str]]:
    """Transform raw batch JSON into (system_message, questions).

    Returns the system_message and list of question strings.
    """
    system_message = json_obj.get("system_message")
    if not isinstance(system_message, str) or not system_message.strip():
        raise ValueError("Questions batch output missing 'system_message'")

    questions = json_obj.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions batch output missing 'questions' list")

    valid = [q for q in questions if isinstance(q, str) and q.strip()]
    if not valid:
        raise ValueError("Questions batch output had no valid question strings")

    return system_message, valid


def transform_evaluate_batch(
    json_obj: dict[str, Any],
) -> EvaluationResult:
    """Transform raw batch JSON into an EvaluationResult."""
    score = json_obj.get("score")
    improved_answer = json_obj.get("improved_answer")

    if not isinstance(score, int) or not (1 <= score <= 10):
        raise ValueError(f"Invalid score in evaluate batch output: {score}")
    if not isinstance(improved_answer, str) or not improved_answer.strip():
        raise ValueError("Missing improved_answer in evaluate batch output")

    return EvaluationResult(score=score, improved_answer=improved_answer)


# ---------------------------------------------------------------------------
# Human-readable writers
# ---------------------------------------------------------------------------


def write_readable_questions(
    questions: QuestionList,
    output_path: Path,
    source: str,
    append: bool = True,
) -> None:
    """Write generated questions to a human-readable text file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(f"=== Source: {source} ===\n")
        f.write(f"System message: {questions.system_message}\n\n")
        for j, q in enumerate(questions.questions, 1):
            f.write(f"Q{j}: {q}\n")
        f.write("\n\n")


def write_readable_answers(
    items: list[dict[str, str]],
    output_path: Path,
    append: bool = True,
) -> None:
    """Write Q/A items to a human-readable text file.

    Each item has keys: source, system_message, question, answer.
    """
    mode = "a" if append else "w"
    current_source = None
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            if source != current_source:
                if current_source is not None:
                    f.write("\n")
                f.write(f"=== Source: {source} ===\n\n")
                current_source = source
            f.write(f"Q: {item['question']}\n")
            f.write(f"A: {item['answer']}\n\n")


def write_readable_evaluation(
    items: list[dict[str, Any]],
    output_path: Path,
    append: bool = True,
) -> None:
    """Write evaluation results to a human-readable text file.

    Each item has keys: source, question, answer, score, improved_answer, routed_to.
    """
    mode = "a" if append else "w"
    current_source = None
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            if source != current_source:
                if current_source is not None:
                    f.write("\n")
                f.write(f"=== Source: {source} ===\n\n")
                current_source = source
            f.write(f"Q: {item['question']}\n")
            f.write(f"A: {item['answer']}\n")
            f.write(f"Score: {item['score']}/10 -> {item['routed_to']}\n")
            if item["routed_to"] == "DPO":
                f.write(f"Improved: {item['improved_answer']}\n")
            f.write("\n")


def write_readable_questions_raw(
    items: list[dict[str, Any]], output_path: Path, append: bool,
) -> None:
    """Write questions from batch-transformed data to a readable text file.

    Each item has keys: source, system_message, questions.
    """
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            f.write(f"=== Source: {item['source']} ===\n")
            f.write(f"System message: {item['system_message']}\n\n")
            for j, q in enumerate(item["questions"], 1):
                f.write(f"Q{j}: {q}\n")
            f.write("\n\n")


def write_readable_evaluation_raw(
    items: list[dict[str, Any]], output_path: Path, append: bool,
) -> None:
    """Write evaluation results from batch-transformed data to a readable file.

    Each item has keys: source, question, answer, score, improved_answer, routed_to.
    """
    write_readable_evaluation(items, output_path, append=append)
