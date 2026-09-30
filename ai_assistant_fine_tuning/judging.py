"""Step 3 prompts and routing: score every answer, rank it, then write to order.

Two passes, because the two cannot be one.

Pass 1 scores an answer's substance and its Serbian, separately and on their
own.  Pass 2 writes the answer that goes into the training set.  Which kind of
writing a record gets depends on where it ranks against the rest of the batch,
and a rank is not knowable while the batch is still being scored -- so the
instruction has to arrive in a second call.

Earlier versions put the boundary inside the scoring prompt, first as a number
to compare against and then as a list of failures to name.  Both made the split
swing between nothing and everything depending on which paragraph of the rubric
pushed hardest.  A rank cannot swing: the worst quarter is the worst quarter
whatever the model does with the scale, and the scale does move -- the same
answers scored 48-94 under one revision of this prompt and 43-95 under the next.

A single self-scaling instruction was tried in between ("keep what is sound,
replace what is not") and does not work: the model replaced everything, writing
72% new text whether it had just scored the answer 48 or 94.  Separate prompts
give 8% new text for repairs against 72% for rewrites.

The rewrite prompt deliberately does not receive the original answer.  It should
write the best answer the material supports rather than a reaction to a bad one,
and withholding the attempt keeps a weak answer from anchoring the good one.

Which mode a record gets is a `judge --write` decision and says nothing about
which training file it ends up in: `make-sft` and `make-dpo` both accept either
kind.  The modes differ in how far the teacher's answer travels from the model's
-- a repair moves a little, a rewrite starts over -- and that distance is what a
DPO pair is made of.  Repairs make hard pairs and rewrites make easy ones.
"""

from __future__ import annotations

import logging
import random
import string
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"

# gpt-5.6-sol and its siblings reject `temperature` outright -- "Only the
# default (1) value is supported" -- so None means omit the parameter.  Models
# that accept one take a float here.
DEFAULT_JUDGE_TEMPERATURE: float | None = None

DEFAULT_SCALE_MAX = 100

# Share of the batch, worst first by substance, that `--write split` rewrites
# rather than repairs.  It says nothing about SFT or DPO: either training file
# can be built from either kind of row.  Not in any prompt, and applied after
# scoring, so a batch that has been paid for can be re-split at a different
# share without being re-scored.
DEFAULT_REWRITE_SHARE = 0.25

SYSTEM_JUDGE = (
    "You evaluate answers from an AI teaching assistant on the Petlja LMS "
    "platform that helps teachers prepare lessons for online courses."
)
SYSTEM_WRITE = (
    "You are an AI teaching assistant on the Petlja LMS platform that helps "
    "teachers prepare lessons for online courses."
)

# What a good answer is.  Shared verbatim by all three prompts so the standard
# an answer is scored against is the standard it is then rewritten to.
_CONTEXT = """\
These questions are drawn from the lesson itself.  The teacher is not asking \
for anything invented — no new exercises, no lesson plan, no test.  They want \
what the lesson already contains explained in their own language, so they can \
teach it.  The answer is in the material, and a good answer is the one that \
finds it and states it clearly.

What makes an answer good here:

**Faithfulness to the material.**  The material is the source of truth, and \
because the question was drawn from it, the answer is in there.  Repeating what \
the lesson says is not a weakness — it is the job.  What counts is whether the \
answer says what the lesson says: the same distinctions, the same terms, the \
same conventions.  An answer that follows the lesson is worth more than one \
merely correct in general: if the lesson calls module functions directly, an \
answer that creates an object first is a worse fit even though both run.  \
Contradicting the material, or quietly replacing its account with a more \
standard one, is the most serious fault an answer can have.

**Covering the whole question.**  These questions usually ask two or three \
things at once — what something is, how it works, and why it matters, or a \
definition followed by a comparison.  Every clause needs answering.  Covering \
the first part well and letting the last one go is a real failure, and a common \
one; it is worse than answering all of them plainly.

**Correctness.**  Watch the difference between what a standard permits and what \
tools happen to allow; between what a piece of notation means and what it looks \
like it means.  Where an answer lists categories, the rest of it should cover \
the categories it listed.  Code and examples must hold up, and should be the \
lesson's own where the lesson has them.

**Staying inside the lesson.**  Asked about something the lesson mentions in \
passing, a weak assistant reaches for everything it knows: further tools, \
version histories, standards, alternatives, notes on industry practice.  All of \
it may be true and none of it was asked for.  If the lesson names three of \
something, name those three.  Depth the material does not have is not \
thoroughness.

**Shape.**  Padding, repetition and digression cost in proportion to how hard \
they make the answer to use.  A little is normal.  What is not is an answer the \
teacher has to hunt through — buried in related matters nobody asked about, or \
making the same point four times across four headings, or closing with a \
summary that repeats what was already said.

**Tasks.**  Where the question asks what a student practises or investigates in \
a task from the lesson, say what the task exercises and what students commonly \
get wrong — not the finished solution.

**Serbian.**  The right language and script to match the question, natural \
phrasing, correct grammar and case, no words borrowed from a neighbouring \
language, clean Markdown.  Answering in English because the subject matter is \
English is a real failure, not a stylistic quibble.
"""

SCORE_PROMPT = """\
A teacher is preparing a lesson and asked an AI assistant to explain something \
from the material they teach.  Below you will find that course material, the \
teacher's question, and the answer the assistant gave.  Your job is to score \
that answer.

""" + _CONTEXT + """
Course material the assistant was working from:
{system_message}

The teacher's question:
{question}

The answer to score:
{answer}

Give it two scores, from 1 to 100, and keep them apart.  An answer can be sound \
in substance and clumsy in Serbian, or fluent and wrong.  Do not let one drag \
the other.

**substance_score** — is what it says right, and is it what was asked?  \
Everything above except the Serbian.  Language does not enter this score: an \
answer whose substance is right but whose Serbian is poor scores high here.
Bands: 86-100 faithful, complete and clean; 71-85 sound with small problems; \
51-70 drifts past the material, or leaves a clause unanswered, or buries what \
was asked; 26-50 wrong in places, or contradicts the material, or misses most \
of what was asked; 1-25 wrong, off topic, or liable to mislead.

**language_score** — is it written the way a Serbian teacher would write it?  \
Substance does not enter this score: a fluent answer that happens to be wrong \
scores high here.
Bands: 86-100 reads as a native speaker wrote it; 71-85 a few slips; 51-70 \
visibly off, repeated errors or stiff translationese; 26-50 errors in most \
sentences, or the wrong script throughout; 1-25 the wrong language entirely.

Place each score inside its band by how fully it satisfies the points above.  Do \
not settle on a comfortable default number: answers differ, and the scores \
should say where.
"""

REPAIR_PROMPT = """\
A teacher asked an AI assistant to explain something from the material they \
teach.  The answer below is sound — it has been judged and it says the right \
things.  What it needs is repair, not replacement.

""" + _CONTEXT + """
Course material:
{system_message}

The teacher's question:
{question}

The answer to repair:
{answer}

Return this same answer with its faults removed.  You may **delete** and you \
may **correct**.  You may not re-word.

Delete what the lesson does not support and what the question did not ask for: \
whole sentences, bullets, headings, digressions, a closing summary that repeats \
what was already said.

Correct what is broken: grammar, case endings, typos, the wrong script, a word \
borrowed from a neighbouring language, a term the lesson uses differently, \
broken Markdown.  Fixing the script means transliterating, not rewriting.

Every sentence you keep must survive in its author's own words and in its \
original order.  Do not reach for a better way of putting something.  Do not \
merge two sentences into a cleaner one.  Do not write a new sentence to bridge \
a gap left by something you cut.  A reader who knows the original should see \
the same answer, shorter and with its mistakes gone.

This matters more than the result reading well.  An imperfect sentence in the \
author's words is worth more here than a better one in yours.

Return only the repaired answer, in Markdown.  No commentary.
"""

REWRITE_PROMPT = """\
A teacher is preparing a lesson and asks you to explain something from the \
material they teach.  Answer them.

""" + _CONTEXT + """
Course material:
{system_message}

The teacher's question:
{question}

Write the answer.  Stay inside the lesson: say what it says, cover every part \
of the question, and stop.  Write directly to the teacher, in the language and \
script of the question, in Markdown.  Return only the answer, no commentary.
"""


# Recorded into every calibration results file as provenance. Bump it whenever
# the prompts or the scale below change: agreement figures are filed under this
# name, and a run recorded against "two-pass-rank" has to mean one fixed judge
# or old numbers quietly stop being comparable to new ones.
JUDGE_NAME = "two-pass-rank"


class JudgeConfig(BaseModel):
    """The judge, and everything a calibration run is allowed to vary.

    A bare `JudgeConfig()` is production behaviour; pass `model=` to run the
    same prompts on a different teacher. If you ever need to compare two judges
    head to head, construct the second one here and give it its own `name` --
    do not edit these defaults in place, because that silently invalidates every
    result already recorded against JUDGE_NAME.
    """

    name: str = JUDGE_NAME
    model: str = DEFAULT_JUDGE_MODEL
    temperature: float | None = DEFAULT_JUDGE_TEMPERATURE
    scale_max: int = DEFAULT_SCALE_MAX
    rewrite_share: float = DEFAULT_REWRITE_SHARE
    system_judge: str = SYSTEM_JUDGE
    system_write: str = SYSTEM_WRITE
    score_prompt: str = SCORE_PROMPT
    repair_prompt: str = REPAIR_PROMPT
    rewrite_prompt: str = REWRITE_PROMPT


# A judge with a different rubric arrives as a file, never as an edit to the
# prompts above: editing them in place would change what every result filed
# under JUDGE_NAME means.  Each prompt must take exactly the placeholders its
# pass fills in -- and the rewrite prompt no {answer}, since it never sees the
# attempt.
_PROMPT_PLACEHOLDERS = {
    "score_prompt": {"system_message", "question", "answer"},
    "repair_prompt": {"system_message", "question", "answer"},
    "rewrite_prompt": {"system_message", "question"},
}
_OPTIONAL_PROMPT_FIELDS = {"system_judge", "system_write"}


def load_prompts(path: Path) -> dict[str, str]:
    """Read a judge's name and prompts from a TOML file, for `JudgeConfig(**...)`.

    Required: `name`, `score_prompt`, `repair_prompt`, `rewrite_prompt`.
    Optional: `system_judge`, `system_write`.  Everything is checked here, before
    a single call is paid for.  The score prompt still has to ask for
    `substance_score` and `language_score`: that response format is fixed.
    """
    import tomllib  # stdlib from 3.11; only this function needs it

    with open(path, "rb") as handle:
        data = tomllib.load(handle)

    required = {"name", *_PROMPT_PLACEHOLDERS}
    missing = required - set(data)
    unknown = set(data) - required - _OPTIONAL_PROMPT_FIELDS
    if missing or unknown:
        raise ValueError(
            f"{path}: missing {sorted(missing) or 'nothing'}, "
            f"unknown {sorted(unknown) or 'nothing'}"
        )
    not_text = sorted(
        k for k, v in data.items() if not isinstance(v, str) or not v.strip()
    )
    if not_text:
        raise ValueError(f"{path}: {', '.join(not_text)} must be non-empty text")
    if data["name"] == JUDGE_NAME:
        raise ValueError(
            f"{path}: name {JUDGE_NAME!r} belongs to the built-in judge; give "
            "this one its own, so its scores are never filed as the built-in's"
        )
    for field, wanted in _PROMPT_PLACEHOLDERS.items():
        try:
            found = {
                name
                for _, name, _, _ in string.Formatter().parse(data[field])
                if name is not None
            }
        except ValueError as exc:
            raise ValueError(
                f"{path}: {field}: {exc} (write a literal brace as {{{{ or }}}})"
            ) from exc
        if found != wanted:
            raise ValueError(
                f"{path}: {field} must use exactly the placeholders "
                f"{sorted(wanted)}, found {sorted(found)}"
            )
    return data


class ScoreResult(BaseModel):
    """Wire format for pass 1.

    This class's JSON schema is sent as `response_format`, so it holds exactly
    what the model should produce and nothing else.  No improved answer: pass 1
    writes no prose, which is what makes it cost ~20 output tokens instead of
    ~900.
    """

    substance_score: int = Field(ge=1, le=DEFAULT_SCALE_MAX)
    language_score: int = Field(ge=1, le=DEFAULT_SCALE_MAX)


class Score(BaseModel):
    """What a caller gets back: the model's two numbers plus provenance.

    `scorer` and `scale_max` are recorded so a scores file can be read months
    later and still be interpretable. There is no improved answer here -- pass 1
    writes no prose; see `write_answer` for that.
    """

    substance_score: int
    language_score: int
    scale_max: int
    scorer: str


def _temperature_kwargs(temperature: float | None) -> dict[str, Any]:
    return {} if temperature is None else {"temperature": temperature}


# `build_teacher_client` constructs the client with max_retries=0 on purpose --
# the SDK's own retries would multiply into whatever ladder the caller runs --
# so a caller that does not back off does not retry at all. This module is a
# leaf (pydantic and stdlib only, so the compare repo can import it without the
# training stack), which rules out tenacity, so the ladder is written by hand.
RETRY_WAITS = (4, 8, 16, 32, 64)

# Matched structurally rather than by class, to avoid importing openai here.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_RETRY_NAMES = {
    "RateLimitError", "APIConnectionError", "APITimeoutError",
    "InternalServerError", "APIStatusError",
}


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in _RETRY_STATUS:
        return True
    return type(exc).__name__ in _RETRY_NAMES


def _retry_after(exc: BaseException) -> float | None:
    """Honour the endpoint's own Retry-After if it sent one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after") or headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _with_backoff(what: str, call, *args, **kwargs):
    """Call `call`, waiting out rate limits instead of dropping the record.

    Under concurrency a rate limit is not a failure, it is a queue: the work is
    still there and the endpoint is asking for it later. Dropping it turns a
    busy deployment into silently missing training data, which is exactly what
    happened the first time this ran without a ladder -- 87% of records lost
    while the progress counter kept climbing.
    """
    last: BaseException | None = None
    for attempt, wait in enumerate((*RETRY_WAITS, None), start=1):
        try:
            return call(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if wait is None or not _is_retryable(exc):
                raise
            last = exc
            delay = _retry_after(exc) or wait
            # Jitter so eight workers rejected together do not all come back at
            # the same instant and trip the limit again.
            delay += random.uniform(0, delay * 0.25)
            logger.warning(
                "%s: %s on attempt %d, retrying in %.0fs",
                what, type(exc).__name__, attempt, delay,
            )
            time.sleep(delay)
    raise last  # unreachable; keeps type checkers happy


def score_answer(
    client: Any,
    *,
    system_message: str,
    question: str,
    answer: str,
    config: JudgeConfig | None = None,
) -> Score:
    """Pass 1: score one answer's substance and Serbian.

    Takes a config rather than loose arguments so `plcmp calibrate` runs the
    exact judge `astft gen-td` routes on, selected the same way.
    """
    config = config or JudgeConfig()
    response = _with_backoff(
        "score",
        client.beta.chat.completions.parse,
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_judge},
            {
                "role": "user",
                "content": config.score_prompt.format(
                    system_message=system_message,
                    question=question,
                    answer=answer,
                ),
            },
        ],
        response_format=ScoreResult,
        **_temperature_kwargs(config.temperature),
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(
            f"Scorer returned no parsed result "
            f"(refusal: {response.choices[0].message.refusal!r})"
        )
    if parsed.substance_score > config.scale_max or parsed.language_score > config.scale_max:
        raise ValueError(
            f"Scorer returned a score above the configured scale_max "
            f"({config.scale_max}): {parsed!r}"
        )
    return Score(
        substance_score=parsed.substance_score,
        language_score=parsed.language_score,
        scale_max=config.scale_max,
        scorer=config.name,
    )


def write_answer(
    client: Any,
    *,
    mode: str,
    system_message: str,
    question: str,
    answer: str = "",
    config: JudgeConfig | None = None,
) -> str:
    """Pass 2: write the answer this record contributes to the training set.

    `mode` is "repair" or "rewrite".  Rewrite ignores `answer` entirely — see
    the module docstring.  Returns Markdown, not JSON: the output is prose and
    round-tripping Serbian Markdown through a JSON string buys nothing.
    """
    config = config or JudgeConfig()
    if mode == "repair":
        prompt = config.repair_prompt.format(
            system_message=system_message, question=question, answer=answer,
        )
    elif mode == "rewrite":
        prompt = config.rewrite_prompt.format(
            system_message=system_message, question=question,
        )
    else:
        raise ValueError(f"mode must be 'repair' or 'rewrite', got {mode!r}")

    response = _with_backoff(
        f"write/{mode}",
        client.chat.completions.create,
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_write},
            {"role": "user", "content": prompt},
        ],
        **_temperature_kwargs(config.temperature),
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Writer returned an empty answer in {mode} mode")
    return content


def split_by_rank(
    judged: list[dict[str, Any]],
    rewrite_share: float = DEFAULT_REWRITE_SHARE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scored records into (repair, rewrite) by substance rank.

    A pure function of the scored records, so a share can be changed and the
    batch re-split without re-scoring it.  Language breaks ties, and ties at the
    cut go to repair, which keeps the rewrite group at or under the requested
    share rather than over it.
    """
    ranked = sorted(judged, key=lambda r: (r["substance_score"], r["language_score"]))
    n_rewrite = round(len(ranked) * rewrite_share)
    return ranked[n_rewrite:], ranked[:n_rewrite]


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\xa0", " ").strip()


def as_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    """The teacher's answer becomes an ordinary chat-format SFT row.

    Normalised through `_norm`, the same as `as_dpo_record`: a trailing newline
    inside an assistant turn is a token the model learns to emit, and the two
    builders disagreeing about it would put the same answer in two shapes
    depending on which training file it landed in.

    A record with a reasoning trace (`gen-answers --keep-reasoning`) puts it in
    front of the answer as `<think>\\n...\\n</think>\\n\\n`, which the Qwen3 chat
    template splits back out and renders the same way.  Not as a separate
    `reasoning_content` key: rows with and without it give the message dicts
    uneven keys, `datasets` then loads the column as Arrow JSON, and TRL's
    truncation fails on that type.
    """
    answer = _norm(record["improved_answer"])
    if has_trace(record):
        answer = f"<think>\n{_norm(record['reasoning'])}\n</think>\n\n{answer}"
    return {
        "messages": [
            {"role": "system", "content": _norm(record["system_message"])},
            {"role": "user", "content": _norm(record["question"])},
            {"role": "assistant", "content": answer},
        ]
    }


def has_trace(record: dict[str, Any]) -> bool:
    return bool(str(record.get("reasoning") or "").strip())


def as_dpo_record(record: dict[str, Any]) -> dict[str, Any]:
    """Rewritten answers become preference pairs against the original.

    Message lists, not strings: TRL reads a string `prompt` as its *standard*
    format and tokenizes it verbatim, so training would optimise a surface form
    neither `as_sft_record` nor vLLM ever produces -- both go through the
    tokenizer's chat template.  Conversational rows let TRL apply that template
    itself, which also removes the prompt/chosen token-boundary mismatches a
    raw concatenation invites.
    """
    return {
        "prompt": [
            {"role": "system", "content": _norm(record["system_message"])},
            {"role": "user", "content": _norm(record["question"])},
        ],
        "chosen": [{"role": "assistant", "content": _norm(record["improved_answer"])}],
        "rejected": [{"role": "assistant", "content": _norm(record["answer"])}],
    }
