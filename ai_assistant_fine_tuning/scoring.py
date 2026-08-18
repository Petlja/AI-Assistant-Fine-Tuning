"""Pointwise answer scorer — the component that decides SFT vs DPO routing.

The teacher model reads ONE answer at a time (never a pair) and returns a score
plus an improved version.  `gen-td` routes on that score, so this module alone
determines the shape of every training dataset the pipeline produces.

**This is deliberately a LEAF module.**  It imports only `pydantic` at runtime,
and `openai` for typing.  It must never import `core` — or anything that pulls
`plct_ai_data_unifier`, `torch`, or `vllm` — because AI-Assistant-LLM-Compare
imports it straight out of a checkout of this repo to calibrate it against human
pairwise preferences, and that repo cannot install the training stack.

Keeping this module a leaf is also what lets this repo's `pyproject.toml` stay
untouched: the fine-tuning VM's environment is pinned and re-syncs on every
`uv run`, so dependency changes here are expensive.  Do not add imports.

See `PLAN.md` in AI-Assistant-LLM-Compare for the calibration workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import-light
    from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Mirrors core.DEFAULT_MODEL. Duplicated rather than imported so this module
# stays a leaf; callers normally pass an explicit model anyway.
DEFAULT_SCORER_MODEL = "gpt-5.4-mini"

# Temperature 0.3 is the long-standing value for this task. Variant "v2-temp0"
# in SCORER_VARIANTS exists to test whether run-to-run noise is costing us
# agreement with human annotators.
DEFAULT_SCORER_TEMPERATURE = 0.3

# The score scale. Moved from 1-10 to 1-100 on 2026-08-18: integer 1-10 scores
# tie constantly, which collapses derived pairwise verdicts into "Tie" and makes
# the judge/human agreement metric go flat.
DEFAULT_SCALE_MAX = 100

# Minimum score routed to SFT; below it goes to DPO. A straight translation of
# the old 7/10 and NOT yet calibrated — it gets re-anchored from a real score
# distribution once alignment work lands (PLAN.md Phase 7.2).
DEFAULT_THRESHOLD = 70

EVALUATION_SYSTEM_MESSAGE = (
    "You evaluate and improve answers from an AI teaching assistant on the "
    "Petlja LMS platform that helps teachers prepare lessons for online courses."
)

EVALUATION_BATCH_SYSTEM_MESSAGE = (
    "You evaluate and improve answers from an AI teaching assistant. "
    "Return ONLY a valid JSON object with 'score' and 'improved_answer' keys."
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

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

Score the answer on a scale from 1 to 100.  Use the full range — a generic \
paragraph and a detailed, well-structured answer with examples should NOT get \
similar scores.

Score anchors:
- 1-25: Poor — incorrect, off-topic, could mislead a teacher or student, or \
ignores the course context entirely.
- 26-50: Below average — partially correct but with significant gaps, \
inaccuracies, or generic advice that is not grounded in this course's subject.
- 51-70: Adequate — addresses the question reasonably but lacks depth, \
concrete examples, or polish; a teacher could use it but would have to fill in gaps.
- 71-85: Good — accurate, relevant and well-structured, directly usable for \
lesson preparation, with only minor issues.
- 86-100: Excellent — accurate, comprehensive, cleanly formatted, precisely \
scoped to the course subject, and immediately useful for lesson preparation.

Key evaluation criteria:
- Is the answer faithful to the specific course subject? (e.g. C-specific if \
the course is about C, not generic programming advice)
- Would a teacher find this answer practically useful for lesson prep?
- Is it concise without sacrificing important details?
- Is it formatted cleanly in Markdown?
- Does it admit uncertainty rather than hallucinating when appropriate?
- For pedagogical questions: does it offer concrete strategies a teacher can \
actually use in a classroom?

Then provide an **improved_answer** that would score 90-100.  The improved answer \
must be written as if the AI assistant is directly responding to the teacher — \
in the same language as the question, formatted in Markdown.
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Wire format the teacher model returns.

    This class's JSON schema is sent to the model as `response_format`, so it
    must contain exactly the fields the model should produce — no metadata.
    Anything the caller needs but the model should not see belongs on `Score`.
    """

    score: int = Field(ge=1, le=DEFAULT_SCALE_MAX)
    improved_answer: str


class ScorerConfig(BaseModel):
    """One scorer variant. Everything that can be tuned during calibration.

    Variants are compared against a frozen set of human pairwise verdicts; see
    PLAN.md Phase 6. Change one field per variant so a difference in agreement
    can be attributed to something.
    """

    name: str
    model: str = DEFAULT_SCORER_MODEL
    temperature: float = DEFAULT_SCORER_TEMPERATURE
    scale_max: int = DEFAULT_SCALE_MAX
    system_message: str = EVALUATION_SYSTEM_MESSAGE
    prompt_template: str = EVALUATION_PROMPT


class Score(BaseModel):
    """What the caller gets back: the model's output plus provenance.

    `scorer` and `scale_max` are recorded so a scores file can be read months
    later and still be interpretable, and `raw` is kept so a surprising score
    can be investigated without re-running the call.
    """

    score: int
    scale_max: int
    improved_answer: str
    scorer: str
    raw: str


# The registry calibration runs select from. "v1-baseline" is the behaviour
# `gen-td` has always had (at the 1-100 scale). Add variants, never edit one in
# place — an edited variant silently invalidates every result recorded against
# its name.
SCORER_VARIANTS: dict[str, ScorerConfig] = {
    "v1-baseline": ScorerConfig(name="v1-baseline"),
}

DEFAULT_SCORER = "v1-baseline"


def get_scorer(name: str = DEFAULT_SCORER, *, model: str | None = None) -> ScorerConfig:
    """Look up a variant, optionally overriding just the model.

    The model override exists because `gen-td --model` and the calibration
    harness both need to swap the teacher model without defining a new variant.
    """
    if name not in SCORER_VARIANTS:
        known = ", ".join(sorted(SCORER_VARIANTS))
        raise ValueError(f"Unknown scorer variant {name!r}. Known variants: {known}")
    config = SCORER_VARIANTS[name]
    if model is not None and model != config.model:
        config = config.model_copy(update={"model": model})
    return config


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def build_scoring_messages(
    *,
    system_message: str,
    question: str,
    answer: str,
    config: ScorerConfig,
) -> list[dict[str, str]]:
    """Render the chat messages for one scoring call.

    Split out from `score_answer` so the batch path can build identical requests
    without duplicating the prompt formatting.
    """
    return [
        {"role": "system", "content": config.system_message},
        {
            "role": "user",
            "content": config.prompt_template.format(
                system_message=system_message,
                question=question,
                answer=answer,
            ),
        },
    ]


def score_answer(
    client: OpenAI,
    *,
    system_message: str,
    question: str,
    answer: str,
    config: ScorerConfig,
) -> Score:
    """Score ONE answer and produce an improved version.

    The scorer never sees a competing answer — that is the whole point. The
    calibration harness derives a pairwise verdict by calling this once per
    side and comparing the two scores, which is exactly what `gen-td` does when
    it routes a single answer against a threshold.

    `client` is injected rather than constructed here so callers control auth,
    retries and base URL.
    """
    response = client.beta.chat.completions.parse(
        model=config.model,
        messages=build_scoring_messages(
            system_message=system_message,
            question=question,
            answer=answer,
            config=config,
        ),
        response_format=EvaluationResult,
        temperature=config.temperature,
    )
    message = response.choices[0].message
    parsed = message.parsed
    if parsed is None:
        refusal = getattr(message, "refusal", None)
        raise ValueError(f"Scorer returned no parsed result (refusal={refusal!r})")

    if not 1 <= parsed.score <= config.scale_max:
        raise ValueError(
            f"Score {parsed.score} outside 1..{config.scale_max} for scorer {config.name!r}"
        )

    return Score(
        score=parsed.score,
        scale_max=config.scale_max,
        improved_answer=parsed.improved_answer,
        scorer=config.name,
        raw=message.content or "",
    )


def transform_evaluate_batch(json_obj: dict[str, Any]) -> EvaluationResult:
    """Transform raw batch JSON into an EvaluationResult.

    The batch path asks for a plain JSON object rather than a strict schema, so
    the bounds have to be enforced here instead of by the API.
    """
    score = json_obj.get("score")
    improved_answer = json_obj.get("improved_answer")

    if not isinstance(score, int) or not (1 <= score <= DEFAULT_SCALE_MAX):
        raise ValueError(f"Invalid score in evaluate batch output: {score}")
    if not isinstance(improved_answer, str) or not improved_answer.strip():
        raise ValueError("Missing improved_answer in evaluate batch output")

    return EvaluationResult(score=score, improved_answer=improved_answer)


def evaluate_answer(
    client: OpenAI,
    system_message: str,
    question: str,
    answer: str,
    model: str,
) -> EvaluationResult:
    """Positional-argument wrapper kept for existing callers.

    Prefer `score_answer`, which returns the richer `Score`. This exists so the
    batch and sync paths could be moved onto the scorer without changing their
    call sites in the same commit.
    """
    result = score_answer(
        client,
        system_message=system_message,
        question=question,
        answer=answer,
        config=get_scorer(model=model),
    )
    return EvaluationResult(score=result.score, improved_answer=result.improved_answer)
