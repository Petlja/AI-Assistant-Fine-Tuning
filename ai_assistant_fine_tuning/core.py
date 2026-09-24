"""Shared domain library for the 3-step training data pipeline.

Step 1 – Question generation:  two passes over a course.  Pass 1 sends each
         lecture to a local model, which writes the lecture arc and labels every
         activity `teaching` or `practice`.  Pass 2 sends each teaching activity
         to the teacher model, which cuts it into segments and writes one
         question per segment plus the context that question is answered from.
Step 2 – Answer generation:    local vLLM model answers those questions.
Step 3 – Evaluation & routing: teacher model scores each answer 1-100,
         produces an improved version, and the caller routes results into
         SFT (score >= threshold) or DPO (score < threshold) training data.

Contains Pydantic models, generation prompts, OpenAI teacher-model callers,
format converters and human-readable writers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal, NamedTuple

from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


from plct_ai_data_unifier.metadata_model import (
    PetljaActivityEnum,
    UdBook,
    UdSegment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = "../data-preprocessing/dataset/"
# Step 1 writes one file per context variant; they differ only in what fills the
# material block of the system message.  gen-answers runs once per file.
DEFAULT_QUESTIONS_DERIVED_FILE = "questions_derived.jsonl"
DEFAULT_QUESTIONS_FULL_FILE = "questions_full.jsonl"
DEFAULT_ANSWERS_OUTPUT_FILE = "answers.jsonl"
DEFAULT_SFT_OUTPUT_FILE = "training_data.jsonl"
DEFAULT_DPO_OUTPUT_FILE = "training_data_dpo.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-14B"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"
# Pass 1 is a classification over a whole lecture and runs on the local server,
# so it is free: a bigger model than the learner costs nothing here.
DEFAULT_MAPPER_MODEL = "Qwen/Qwen3.8-27B-FP8"
# gpt-5.x lives on the Petlja Azure deployment, not on api.openai.com.  Used
# only when CHATAI_AZURE_API_KEY is set; otherwise the plain OpenAI client is.
DEFAULT_AZURE_ENDPOINT = "https://petljaopenaiservicev2.openai.azure.com/"
# The versions registered in PLCT-Server's model_conf.py (2023-03-15-preview
# and friends) predate structured outputs and fail on response_format.
DEFAULT_AZURE_API_VERSION = "2024-10-21"

# Qwen3-14B is the learner and serves 32768 tokens.  The material rides in
# system_message, so its budget is that minus what the answer needs.  Measured
# with the Qwen3 tokenizer over all 1758 activities in the dataset the largest
# is 22934 tokens, so nothing is dropped today -- this is a guard, not a filter.
LEARNER_CTX = 32768
ANSWER_BUDGET = 4096
CONTEXT_CAP = LEARNER_CTX - ANSWER_BUDGET
# Pass 1 sends a whole lecture.  23 of the 448 lectures in the dataset do not
# fit one call at 32k and the largest is 96283 tokens, so this is the budget
# that decides which lectures get sent as openings instead of in full.  Raise it
# if the local server is started with a longer --max-model-len; every lecture
# fits under 131072.
MAPPER_BUDGET = 28000
CHARS_PER_TOKEN = 2.1  # measured on this corpus; Cyrillic, not the usual ~4

# Pass 2 runs against a shared Azure deployment, so it stays deliberately mild:
# 4 in flight cuts a ~7h serial run over the whole dataset to under 2h, and a
# rejection backs off rather than hammering.  Pass 1 is our own box, bounded by
# whatever --max-num-seqs the vLLM server was started with.
DEFAULT_CONCURRENCY = 4
DEFAULT_LOCAL_CONCURRENCY = 4

# Per-request timeouts.  The SDK defaults to 600s, which is long enough for a
# stuck call to look like a working one: a lecture of gim2_rac_ikt whose
# material closed the model's JSON string early generated whitespace until it
# timed out, six times over, for 3 hours of wall clock.  A pass-1 call measures
# ~10s and a pass-2 call ~24s, so these caps are generous by an order of
# magnitude and exist only to bound that case.
DEFAULT_LOCAL_TIMEOUT = 180.0
DEFAULT_TEACHER_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_GEN_SYSTEM_MESSAGE = (
    "You analyse course material for the Petlja LMS platform and produce structured "
    "breakdowns used to build training data for an AI teaching assistant."
)

# Pass 1: one call per lecture, all its activities at once.  Both outputs need a
# lecture-wide view -- "does this activity introduce material, or exercise
# material introduced earlier" cannot be answered one activity at a time, and
# the arc is by definition about the whole lecture.
LECTURE_MAP_PROMPT = """\
Below is one lecture from an online course, made of several activities marked \
`<<< ACTIVITY n.m >>>`.

1. **arc** — what this lecture teaches and in what order it builds it up.  This
is prepended to every question an assistant will be asked about this lecture, so
it is the reader’s only view of the wider material: name every concept the
lecture introduces, say how they connect, and say what each one assumes.  Cover
the whole lecture, not just its opening — a lecture that introduces five things
needs all five here.  Write it in the script the lecture uses (Cyrillic or
Latin) and never mix the two.  Refer to nothing outside itself: no
„ова лекција“, no „материјал“, no activity numbers.  Keep it dense rather than
discursive; 8-14 sentences is the usual length, longer if the lecture earns it.

2. **activities** — for each activity, its number and one label:

- `teaching`: most of the activity explains, defines or demonstrates.  Its job is \
to introduce material.
- `practice`: most of the activity is tasks set for the reader.  Its job is to \
exercise and test material that earlier activities introduced.

Judge this by what most of the activity consists of, NOT by its title.

--- LECTURE {lecture} ---
{content}
--- END LECTURE ---
"""

# Pass 2: one call per teaching activity.  Pass 1's arc rides along as input so
# the activity can be placed in its lecture.
ACTIVITY_BREAKDOWN_PROMPT = """\
You are given one activity from an online course.  This activity's job is to \
introduce material: most of it explains, defines or demonstrates.

The whole lecture this activity belongs to covers the following, in this order.  \
It is given to the assistant alongside every question, so write as though the \
reader has it.  Use it to place this activity: what came before it, and what the \
reader already has.  Do not treat it as source material.

--- LECTURE OVERVIEW ---
{arc}
--- END LECTURE OVERVIEW ---

Your job is to cut this activity into SEGMENTS and write one question per \
segment, such that someone who read only those questions and their answers would \
get the gist of the activity.

1. **activity_context** — give the assistant its exact context for this activity.  Cover, in \
this order: the subject and who the course is for; the specific topic of this \
activity and where it sits in the course; the tools and vocabulary in play, \
naming them — the environment, the commands, the terms the material introduces \
and any it takes for granted; that the assistant helps teachers prepare lessons \
from this material; and the language and script of the material.  Be specific \
rather than generic: this is what disambiguates a question that would otherwise \
be ambiguous — if the course is about Karel, a question about moving forward \
means `napred()`, and a question about a loop means the `for` that drives the \
robot, not loops in general.  Usually 5-8 sentences.

2. **segments** — walk the activity from top to bottom, in order, and cut it into \
consecutive spans.  Skip passages that carry no subject matter.

   - A segment is one paragraph, or several consecutive paragraphs that only make \
sense together.  It is NOT one sentence.
   - Segments must not overlap, and must appear in the same order as the material.
   - Cover the activity end to end: every passage that carries subject matter \
belongs to exactly one segment, and segments run consecutively down the file.
   - A fenced block is either subject matter or machinery.  Machinery builds the \
interactive widget a student clicks on; the material is not teaching it.  The \
info word after the fence and the first line inside it tell you which: read \
them, work out what the block configures, and keep only what a student sees.
   - A plain language fence (```python, ```c, ```cs, ```sql, ```haskell, ```html) is \
a worked example or a listing.  That is content.  Do not skip it.
   - ``` karel is JavaScript that builds a robot-maze exercise.  `new World(...)`, \
`setRobotStartAvenue`, `setRobotStartStreet`, `setRobotStartDirection`, \
`putBall`, `putHoles`, `addEWWall`, `addNSWall` lay out the maze and \
`isSuccess` grades the attempt — none of that is subject matter.  But the \
`var code = [...]` array inside it is the program the student is shown: read \
those lines as if they were printed in the prose around the block.
   - Other machinery follows the same test: `::: petlja-editor` wraps starter \
files for an editor, `::: parsed-literal` is captured program output, \
`::: collapse` hides a hint or a solution, `::: video` is a video reference, \
and an `{{acsection: ...}}` marker or a comment saying the student does not see \
this code marks setup that never appears on screen.
   - Never write a segment about machinery, and never quote a widget’s \
configuration in a question or a context.
   - Write NO question about course administration: instructions to finish work at \
home, how to split tasks between class and homework, or pointers to later chapters.
   - Do not write two questions that differ only in which task they name.  If \
several passages carry the same idea, cover them in one question.

   For each segment set `segment_kind`:

   - `exposition` — the span explains, defines, demonstrates or warns.  Ask about \
the SUBJECT the span is about.  The span is merely where the answer happens to be \
written down.
   - `task` — the span sets the reader a problem to solve.  Do NOT ask what the \
solution is, and do NOT ask why an author placed it here.  Ask what a student \
practises by doing it, which idea it applies, what it is easy to get wrong.

   **The question must be self-contained.**  The answering model is given the \
lecture overview, the activity description and this question’s material — \
nothing else — so the question has to be clear against those.  Four things break that, and all four are easy to write \
without noticing:

   - **Asking what the material does with the subject** instead of asking \
about the subject.  A verb describing authorial action — introducing, \
announcing, emphasising, summarising, including — makes the question about \
how the text was written.  A teacher wants the topic itself.
   - **Pointing at a part of the document** — the introduction, the closing \
section, this activity, the lesson, the material, here.  The reader has none \
of that in front of them.
   - **Pointing at a worked example rather than describing it.**  An example \
is often the whole reason a passage matters, so keep it — but carry its \
content into the question, so the question names the situation being worked \
through.  Referring to it as the example, the table, the picture, the step, or \
a numbered screen callout assumes a page the reader cannot see.
   - **Enumerating what the document lists** — asking which tasks are given, \
which items appear, what is covered.  Ask instead what the listed thing \
teaches or what doing it produces.

   In every case the repair is the same: name the thing being asked about — \
the concept, the command, the operation, the situation — so the question \
stands on its own.

   `context` — a self-contained explanation of exactly what the question targets. It fills the material block the answering model is given, alongside the lecture overview and the activity description — nothing else reaches it.  Build it from the passage the segment covers, but do not copy that passage: fill in what the lecture leaves implicit, spell out the definitions, name the commands and say what each one does, and carry over any worked example or common mistake that bears on the question.

   Judge it this way: someone who has never seen this course, handed only the lecture overview, the activity description and this context, must be able to answer the question fully and correctly.  Do not restate the activity description — the reader already has it.  If anything needed is missing, add it.

   - 4-10 sentences.  Prose and Markdown, not a quotation and not a bullet dump.
   - Stay inside what this activity actually covers.  Do not invent syntax, results or terminology the material does not use, and do not bring in concepts from later lessons.
   - Never refer to the document: no „у овом делу", „материјал", „горе наведено".
   - For a `task` segment, explain what the task asks, which idea it exercises, and what a student commonly gets wrong — not the finished solution.
   - Write NO answer to the question.  The answer is generated downstream, from this context.

--- ACTIVITY {activity} ---
{content}
--- END ACTIVITY ---
"""

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ActivityRole(BaseModel):
    """Pass 1's verdict on one activity: does it introduce material or drill it."""

    activity: str
    kind: Literal["teaching", "practice"]


class LectureMap(BaseModel):
    """Pass 1's output for one lecture."""

    arc: str  # what the lecture builds; ships as the per-lecture synopsis
    activities: list[ActivityRole]


class Segment(BaseModel):
    """One span of an activity, with the question it is the answer to."""

    segment_kind: Literal["exposition", "task"]
    question: str
    context: str  # standalone explanation, written from the segment, not sliced


class ActivityBreakdown(BaseModel):
    """Pass 2's output for one teaching activity."""

    activity_context: str  # course + activity framing; rides in system_message
    segments: list[Segment]


class Activity(NamedTuple):
    """One non-quiz activity, numbered by position rather than by filename."""

    number: str  # "1.4" -- the 4th non-quiz activity of the 1st lecture
    title: str  # the label the course itself uses
    path: Path
    rel: str  # path relative to the dataset root; the `source` field


def _content_leaves(segments: list[UdSegment]) -> list[UdSegment]:
    """Content-bearing, non-quiz segments under these nodes, in document order.

    Quizzes are dropped here and never reach either pass: structure.json tags
    them `source_type: "quiz"`.
    """
    found: list[UdSegment] = []
    for segment in segments:
        if segment.content_path and segment.source_type != PetljaActivityEnum.QUIZ:
            found.append(segment)
        found.extend(_content_leaves(segment.sub_segments))
    return found


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


def collect_lectures(input_dir: Path) -> dict[str, dict[int, list[Activity]]]:
    """Group a dataset's activities by course and lecture, in document order.

    INPUT_DIR is either one course directory holding a structure.json, or a
    dataset root whose immediate children each hold one.  Step 1 needs lectures
    rather than loose files, so a directory with no structure.json anywhere
    yields nothing and the caller reports it.

    Lecture and activity numbers are positional counts over structure.json, not
    parsed out of filenames: only 6 of the 35 courses in the dataset have two
    numbers in every filename, and only 2 keep their lectures in `<n>_lekcija`
    directories.  So `1.4` is "the 4th non-quiz activity of the 1st lecture",
    which need not match the course's own label -- os6 calls that one 1.5,
    because its quiz took 1.4.  The label the course uses is in `title`.
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

    courses: dict[str, dict[int, list[Activity]]] = {}
    for structure_file in structure_files:
        structure_dir = structure_file.parent
        book = UdBook.model_validate_json(structure_file.read_text(encoding="utf-8"))
        lectures: dict[int, list[Activity]] = {}
        for lecture_no, node in enumerate(book.segments, 1):
            activities: list[Activity] = []
            for activity_no, segment in enumerate(_content_leaves([node]), 1):
                resolved = _resolve_struct_content_path(
                    input_dir, structure_dir, segment.content_path
                )
                if resolved is None:
                    continue
                activities.append(
                    Activity(
                        number=f"{lecture_no}.{activity_no}",
                        title=segment.title,
                        path=resolved,
                        # Always course-qualified, whether INPUT_DIR is the
                        # course or the dataset root, so JSONL from separate
                        # per-course runs fuses without `source` colliding.
                        rel=f"{structure_dir.name}/"
                        f"{resolved.relative_to(structure_dir).as_posix()}",
                    )
                )
            if activities:
                lectures[lecture_no] = activities
        if lectures:
            courses[structure_dir.name] = lectures
    return courses


# ---------------------------------------------------------------------------
# Teacher-model callers (sync)
# ---------------------------------------------------------------------------


def build_teacher_client(
    azure_endpoint: str | None = None,
    api_version: str = DEFAULT_AZURE_API_VERSION,
    timeout: float = DEFAULT_TEACHER_TIMEOUT,
) -> OpenAI | AzureOpenAI:
    """Return the client for the teacher model.

    The gpt-5.x deployments this pipeline uses live on the Petlja Azure
    endpoint, not on api.openai.com, so an AzureOpenAI client is returned
    whenever CHATAI_AZURE_API_KEY is set.  Without that key the plain OpenAI
    client is returned and OPENAI_API_KEY applies as before.
    """
    # max_retries=0: the backoff ladder in _parse_with_backoff is the only
    # retry, so the SDK's own retries do not multiply into it.
    api_key = os.environ.get("CHATAI_AZURE_API_KEY")
    if not api_key:
        return OpenAI(timeout=timeout, max_retries=0)
    endpoint = azure_endpoint or os.environ.get(
        "CHATAI_AZURE_ENDPOINT", DEFAULT_AZURE_ENDPOINT
    )
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        timeout=timeout,
        max_retries=0,
    )


def _parse_with_backoff(client: OpenAI, **kwargs) -> Any:
    """Call the teacher model, backing off when the endpoint pushes back.

    Concurrency turns a rate limit into failed activities rather than speed, so
    every retryable rejection waits instead: 4s, 8s, 16s, 32s, 64s, then gives
    up and lets the caller log the activity as failed.  Pass 2 only -- pass 1
    runs against a single-user box where a rate limit cannot happen and a
    timeout means the lecture is looping, which a retry would only repeat.
    """
    for attempt in Retrying(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        wait=wait_exponential(multiplier=4, max=64),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return client.beta.chat.completions.parse(**kwargs)


def map_lecture(
    client: OpenAI, content: str, lecture_no: int, model: str
) -> tuple[LectureMap, Any]:
    """Pass 1: write the lecture arc and label each of its activities.

    Runs greedy: a label is a classification, and sampling only adds drift -- at
    0.7 two identical runs of os1_dig_svet gave 24 and 11 teaching activities.

    Returns the map and the call's token usage.
    """
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_GEN_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": LECTURE_MAP_PROMPT.format(lecture=lecture_no, content=content),
            },
        ],
        response_format=LectureMap,
        temperature=0,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.parsed, response.usage


def break_down_activity(
    client: OpenAI, content: str, activity_no: str, arc: str, model: str
) -> tuple[ActivityBreakdown, Any]:
    """Pass 2: cut one teaching activity into segments, one question each.

    Returns the breakdown and the call's token usage.
    """
    response = _parse_with_backoff(
        client,
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_GEN_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": ACTIVITY_BREAKDOWN_PROMPT.format(
                    activity=activity_no, arc=arc, content=content
                ),
            },
        ],
        response_format=ActivityBreakdown,
        temperature=0.7,
    )
    return response.choices[0].message.parsed, response.usage


def build_system_message(arc: str, context: str, activity_context: str | None) -> str:
    """Lecture, then activity, then the material for this one question.

    All of it rides in the system message -- the one field that survives
    gen-answers, gen-td, sft-finetune and dpo-finetune untouched, so both
    context variants need zero pipeline code downstream of here.

    `activity_context` is None on the full-context variant: the whole activity
    follows two lines below it, so a written description of it would only repeat
    what the reader is about to read.
    """
    head = f"--- О ЛЕКЦИЈИ ---\n{arc}\n\n"
    if activity_context:
        head += f"--- О АКТИВНОСТИ ---\n{activity_context}\n\n"
        label = "ГРАДИВО ЗА ПИТАЊЕ"
    else:
        label = "ГРАДИВО"
    return (
        head
        + f"--- {label} ---\n{context}\n"
        + "--- КРАЈ ГРАДИВА ---"
    )


# ---------------------------------------------------------------------------
# Human-readable writers
# ---------------------------------------------------------------------------


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
