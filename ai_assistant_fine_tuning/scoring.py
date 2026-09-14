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

# The rubric the scorer runs, and the only one. It replaced an earlier prompt
# that described the production plumbing — the LMS, the RAG retrieval, the
# chunking — none of which the scorer can see or act on, and whose scores
# clumped hard: 42% of them were the single value 78, and a scorer that returns
# one number for everything cannot route anything, whatever the threshold.
#
# Three things this one does differently:
#
# 1. It describes only what the scorer actually has in front of it — the
#    material, the question, and the answer.
# 2. Its rules are the ones a real annotator applied, taken from written
#    feedback on two annotation sessions: the 19-case Qwen3-14B t1-vs-t2 pair
#    (2026-09-02) and the Qwen-vs-Ministral pair (2026-09-10).
# 3. It asks for the score in two steps, band then position, which is what
#    broke the clumping.
#
# Two of the rules came out of the second session and are worth calling out,
# because each corrects a failure that was measured rather than guessed:
#
# - "It does the work the request asks for" — the earlier draft could not tell
#   "grounded in the material" from "copying the material". Asked for homework
#   after two named worked examples, the answer that reproduced both of them
#   scored 78 and the one that wrote three new exercises scored 62.
# - The length counterweight inside "It brings in no more than the question
#   needs" — the other rules all leaned the same way on length, and the scorer
#   duly read them as "shorter is better": it preferred the shorter answer in
#   8 of 9 decided cases, against a human split of 3 of 7.
#
# The improved answer is asked for as what a stronger assistant would write, not
# as "an answer that would score 90-100". Scoring your own target invites the
# model to write to the rubric instead of to the teacher.
EVALUATION_PROMPT = """\
A teacher is preparing a lesson and asked an AI assistant a question about the \
material they teach.  Below you will find the course material the assistant was \
working from, the teacher's question, and the answer the assistant gave.  Your \
job is to judge that answer.

Course material the assistant was working from:
{system_message}

The teacher's question:
{question}

The answer to judge:
{answer}

What makes an answer good:

**It does the work the request asks for.**  Work the assistant hands back \
instead of doing does not count, however well presented.  Material lifted \
from the lesson above is the clearest case: when the teacher asks for \
something new — fresh exercises, a test, a plan, an explanation of their \
own — content copied from the material is the assistant handing the \
teacher back their own input.  Read what the request actually asks to be \
produced, note anything it rules out, and judge whether the answer \
produced it.  An answer that leaves the real work to the teacher has \
failed the request however sound it is.

**It stays with the course material.**  The material above is the source of \
truth.  Your own knowledge of the subject — teaching practice, introductory and \
advanced computer science — is useful and welcome where the material is silent, \
but it must never contradict or quietly replace what the material says.  An \
answer that follows the conventions of the lesson is worth more than one that is \
merely correct in general: if the lesson calls module functions directly, an \
answer that creates an object first is a worse fit even though both run.

**It answers the question that was asked.**  A teacher asked something specific. \
An answer that wanders into the subject at large — restating learning \
objectives, explaining how the task relates to the lesson, giving background \
nobody asked for — is padding, however true it is.  Judge whether the question \
was answered, then whether anything else earned its place.

**It brings in no more than the question needs.**  Reaching for extra concepts, \
options and refinements looks thorough and usually is not: it puts work on the \
teacher to find the part that answers them.  Where two approaches would both \
work, the one that stays inside what the lesson already covers is better.  \
Length itself is not the test, in either direction: a long answer whose \
parts all earn their place is better than a short one the teacher would \
have to finish, and brevity is no virtue in an answer that left the work \
undone.

**It is correct, and precise about what it claims.**  Check the substance, not \
the confidence.  Examples and exercises must actually hold up — an exercise that \
depends on something never defined is not usable.  Watch the difference between \
what a standard permits and what tools happen to allow; between what a piece of \
notation means and what it looks like it means.  Where an answer lists \
categories, the rest of it should cover the categories it listed.

**Everything it adds must earn its place.**  Explanation that tells the teacher \
something the code or the answer does not already say is valuable.  Explanation \
that restates the comments in its own example, or closes with a summary that \
merely repeats what was said, adds length and nothing else.

**A teacher could use it as it stands.**  Practical shape matters: a lesson plan \
should look like a lesson plan, with a plausible running time and the parts a \
teacher expects; notebook code should look like notebook cells.  Ask whether \
this could be taken into a classroom, not just whether it is true.

**It is written in the teacher's language and script.**  Match the question — \
including Cyrillic or Latin script — and write naturally in that language.  \
Answering in English because the subject matter is English is a real failure, \
not a stylistic quibble.  So are words borrowed from a neighbouring language \
that a native speaker would not use.

**It is clean Markdown, and honest about uncertainty.**  Saying plainly that \
something is unclear beats a confident guess.

Now score the answer from 1 to 100, in two steps.

First choose the band it belongs in:
- 86-100: Correct, grounded in the material, exactly on target, and usable as it \
stands.  Nothing in it is wasted.
- 71-85: Solid and usable, with small problems — some padding, a minor \
imprecision, a detail the teacher would have to fill in.
- 51-70: Answers the question, but thinly or without focus — it drifts, restates \
itself, reaches past what was needed, or is generic where the material is \
specific.
- 26-50: Real problems — wrong in places, ignores the conventions of the \
material, written in the wrong language, or offering examples that do not hold \
up.
- 1-25: Wrong, off topic, or liable to mislead the teacher or their students.

Then place it inside that band by how fully it satisfies the points above — the \
top of a band is an answer with one small blemish, the bottom is one that barely \
clears the band below.  Do not settle on a comfortable default number: answers \
differ, and the score should say where.

Finally, write the answer a stronger assistant would have given to this same \
question, from this same material, as **improved_answer**.  Write it as that \
assistant speaking directly to the teacher — in the language and script of the \
question, in Markdown.  Write the answer itself, not a critique of the one above.
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

    The defaults ARE the shipped scorer, so a bare `ScorerConfig(name=...)` is
    the production behaviour. Variants are compared against a frozen set of
    human pairwise verdicts; see PLAN.md Phase 6. Change one field per variant
    so a difference in agreement can be attributed to something.
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


# The registry calibration runs select from. It holds one entry: the rubric
# above, which won the calibration against its two predecessors (recorded under
# the names "v1-baseline" and "v2-natural", both removed once it was settled).
#
# The registry stays because it is the seam `plcmp calibrate --scorer` selects
# through, and because the next round of tuning wants the same discipline as the
# last one: add a variant, change ONE field in it, and never edit a variant in
# place — an edited variant silently invalidates every result already recorded
# against its name.
SCORER_VARIANTS: dict[str, ScorerConfig] = {
    "teacher-rubric": ScorerConfig(name="teacher-rubric"),
}

DEFAULT_SCORER = "teacher-rubric"


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
