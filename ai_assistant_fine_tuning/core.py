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


from plct_ai_data_unifier.metadata_model import (
    PetljaActivityEnum,
    PlctSegmentEnum,
    UdBook,
    UdSegment,
)

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
    "You are an AI teaching assistant on the LMS platform. "
    "Help teachers prepare lessons by answering questions accurately and concisely "
    "based on the provided course material. Format your answers in Markdown. "
    "Answer in the same language as the question."
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """\
You are helping build training data for an AI teaching assistant deployed on \
the LMS platform.  Teachers interact with this assistant via chat while \
preparing lessons — they ask questions about the course material and expect \
accurate, concise, Markdown-formatted answers grounded in that material.

The assistant in production receives:
- A course summary and current lesson summary as context.
- Relevant chunks from the lesson content via RAG (retrieval-augmented generation).
- The teacher's question (which may be in Serbian Cyrillic, Serbian Latin, or English).

The assistant answers in the same language as the question and formats output \
in Markdown.  If unsure, it says so rather than guessing.

Given the following study material (a lecture or quiz from an online course), \
generate:

1. A **system_message** (2-4 sentences) that will be given to the AI assistant \
so it knows the exact course context.  The system_message MUST include:
   - The subject and the specific topic of this material.
   - That the assistant helps teachers prepare lessons from this material.
   - The language context (e.g. "Questions and answers should be in Serbian").
   This is critical for disambiguation — if the course is about C, a question \
about "printing to stdout" must be answered with `printf`, not `print` or `cout`.

2. A list of **questions** a teacher would realistically ask this AI assistant \
while preparing or reviewing a lesson based on this material.

Requirements for the questions:
- Generate between 3 and 15 questions depending on how rich the material is.
- Questions must be self-contained — someone reading only the system_message \
and the question must understand what is being asked without seeing the material.
- Focus on what a teacher actually needs:
  * Explaining concepts to students clearly.
  * Clarifying definitions and terminology.
  * Finding good examples or analogies for difficult topics.
  * Understanding how topics connect to each other.
  * Anticipating student misconceptions and common mistakes.
  * Pedagogical approaches — e.g. "How would I explain time complexity to a \
student who doesn't have a strong math background?"
- Vary question types: factual recall, conceptual explanation, comparison, \
pedagogical strategy, practical classroom use.
- Write all questions in the same language as the source material.

--- MATERIAL ---
{content}
--- END MATERIAL ---
"""

EVALUATION_PROMPT = """\
You are evaluating an answer produced by an AI teaching assistant deployed on \
the LMS platform.  In production, this assistant receives course/lesson \
summaries and RAG-retrieved content chunks as context, then answers teacher \
questions via chat in Markdown format.

The assistant should give accurate, concise, and practically useful answers \
grounded in the course material.  If the assistant is unsure, it should say so \
rather than guessing.  Answers must be in the same language as the question.

Context (system message that was given to the answering assistant):
{system_message}

Question the teacher asked:
{question}

Assistant's answer:
{answer}

Score the answer on a scale from 1 to 10:
- 1-3: incorrect, irrelevant, or could mislead a teacher/student
- 4-6: partially correct but missing important details, vague, or slightly \
inaccurate for the course context
- 7-8: mostly correct and useful, minor issues or could be more concise
- 9-10: excellent — accurate, well-structured, appropriate for the course \
context, and directly useful for lesson preparation

Key evaluation criteria:
- Is the answer faithful to the specific course subject? (e.g. C-specific if \
the course is about C, not generic programming advice)
- Would a teacher find this answer practically useful for lesson prep?
- Is it concise without sacrificing important details?
- Is it formatted cleanly in Markdown?
- Does it admit uncertainty rather than hallucinating when appropriate?
- For pedagogical questions: does it offer concrete strategies a teacher can \
actually use in a classroom?

Then provide an **improved_answer** that would score 9-10.  The improved answer \
must be written as if the AI assistant is directly responding to the teacher — \
in the same language as the question, formatted in Markdown.
"""

# Task-specific config used by both sync and batch paths.
TASK_CONFIG = {
    "questions": {
        "system_message": "You generate training data for an AI teaching assistant on the Petlja LMS platform. Given course material, produce a context-setting system message and realistic teacher questions.",
        "batch_system_message": (
            "You generate training data for an AI teaching assistant on the Petlja LMS platform. "
            "Given course material, produce a context-setting system message and realistic teacher questions. "
            "Return ONLY a valid JSON object with 'system_message' and 'questions' keys."
        ),
        "generation_prompt": QUESTION_GENERATION_PROMPT,
    },
    "evaluate": {
        "system_message": "You evaluate and improve answers from an AI teaching assistant on the Petlja LMS platform that helps teachers prepare lessons for online courses.",
        "batch_system_message": (
            "You evaluate and improve answers from an AI teaching assistant. "
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



_CONTENT_SOURCE_TYPES = {PetljaActivityEnum.READING.value, PlctSegmentEnum.ACTIVITY.value}


def _iter_content_paths(segments: list[UdSegment]) -> list[str]:
    paths: list[str] = []
    for segment in segments:
        if segment.source_type in _CONTENT_SOURCE_TYPES and segment.content_path:
            paths.append(segment.content_path)
        if segment.sub_segments:
            paths.extend(_iter_content_paths(segment.sub_segments))
    return paths


def _resolve_struct_content_path(
    input_dir: Path,
    structure_dir: Path,
    content_path: str,
) -> Path | None:
    # Some datasets store paths from dataset root, others from book root.
    normalized = content_path.replace("\\", "/")
    candidates: list[Path] = [
        input_dir / normalized,
        structure_dir / normalized,
    ]

    if normalized.startswith(f"{structure_dir.name}/"):
        candidates.append(structure_dir / normalized[len(structure_dir.name) + 1 :])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".md":
            return candidate
    return None


def collect_markdown_sources(input_dir: Path) -> list[tuple[str, Path]]:
    """Collect markdown sources, preferring structure.json when available.

    Returns a sorted list of tuples: (source_rel_path, absolute_path).
    """
    structure_files: list[Path] = []
    root_structure = input_dir / "structure.json"
    if root_structure.exists():
        structure_files.append(root_structure)
    else:
        for child in sorted(input_dir.iterdir()):
            candidate = child / "structure.json"
            if child.is_dir() and candidate.exists():
                structure_files.append(candidate)

    records: list[tuple[str, Path]] = []
    seen: set[str] = set()

    if structure_files:
        for structure_file in structure_files:
            structure_dir = structure_file.parent
            book = UdBook.model_validate_json(structure_file.read_text(encoding="utf-8"))
            for content_path in _iter_content_paths(book.segments):
                resolved = _resolve_struct_content_path(input_dir, structure_dir, content_path)
                if resolved is None:
                    continue
                rel = resolved.relative_to(input_dir).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                records.append((rel, resolved))
        return sorted(records, key=lambda item: item[0])

    # Fallback: plain recursive scan for legacy datasets.
    for md_file in sorted(input_dir.rglob("*.md")):
        rel = md_file.relative_to(input_dir).as_posix()
        records.append((rel, md_file))
    return records


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
        def normalize(s):
            return (
                s.replace("\r\n", "\n")
                .replace("\xa0", " ")
                .strip()
            )

        SEP = "\n\n"

        dpo_record = {
            "prompt": normalize(system_message) + SEP + normalize(question) + SEP,
            "chosen": normalize(evaluation.improved_answer),
            "rejected": normalize(answer),
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
