"""Step 3, best of N: pick the local model's best thinking and answer, grade it,
improve it.

For answers from `gen-answers --samples N --keep-reasoning`.  It takes the place
of `judge` for such data and ends at the same kind of file: a written file,
self-contained, one row per kept question, that `make-sft` reads.  Every row is
a thinking row -- the model's own trace and its answer -- so it trains the
answer in the context it is used in, after real reasoning rather than an empty
think block.

For each question:

    checks   `output_checks`, code only: a degenerate pair (loops, CJK, special
             tokens, the wrong script, over the caps, off the stated minutes)
             is out before any teacher call is paid for.
    pick     one teacher call sees the lesson, the question and every pair left,
             shuffled, and picks the best, thinking and answer together.
    grade    one teacher call sees the lesson, the question and the picked pair:
               A     the teacher can use the answer as it is; spelling at most
               B+    sound thinking, a good answer that needs small improvements
               B-    sound thinking, an answer that needs real corrections
               drop  thinking that can't lead to a good answer
             B+ and B- come with what to improve; every grade with spelling fixes.
    improve  A: the spelling fixes, applied by code.  B+ and B-: one teacher call
             improves the answer as the grade says, keeping the rest; code then
             checks the length and runs the checks again.  The trace is never
             touched, so the improved answer is asked to stay the one it leads to.

The outputs are appended as each question finishes and resumed by (source,
question): an interrupted run redoes only what is missing.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel

from ai_assistant_fine_tuning import output_checks as oc
from ai_assistant_fine_tuning.core import build_teacher_client
from ai_assistant_fine_tuning.io_jsonl import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.judging import DEFAULT_JUDGE_MODEL, _temperature_kwargs, _with_backoff

logger = logging.getLogger(__name__)

# Bump when a prompt or a rule below changes, as with judging.JUDGE_NAME.
JUDGE_NAME = "bestof-v6"  # + fix_script (20:35), no prompt change

# How far an improved answer may move from the original, as a length ratio.  A
# B- may cut a good deal (unasked material) or gather answers into a key.
IMPROVE_BOUNDS = {"B+": (0.70, 1.30), "B-": (0.40, 1.40)}
NUMBER = re.compile(r"\d+")
# Calls that still fail after judging._with_backoff's ladder are asked again,
# up to ROUNDS rounds in all, after a pause.
ROUNDS, ROUND_PAUSE = 4, 60

SYSTEM = (
    "You review and improve answers that an AI assistant gave to teachers in "
    "Serbia working with Petlja's online courses."
)

# What a usable answer is, shared by the pick and the grade.  Kept general on
# purpose (the user, 2026-09-30): the teacher's request says the specifics.
GOOD_ANSWER = """\
1. It does what the teacher asked, all of it and not much more: the number of \
items, the format, the time and the level they named.
2. It is built on this lesson and it is correct: the lesson's concepts, terms \
and examples; facts, code, results and answer keys that hold up; nothing \
presented as the lesson's that isn't in it.
3. What the teacher asks to be made is new: an example is tied to the lesson's \
topic but not copied from it; test questions check the lesson without \
repeating its questions or exercises word for word; homework practises the \
lesson with tasks other than the ones done in class.
4. A lesson plan fits the time and the class: phases whose minutes add up, and \
what the teacher and the students actually do in each.
5. When the teacher asks for students who struggle, the answer gives concrete \
help: smaller steps, simpler wording, a worked example, support while they \
work; the lesson's goals stay the same, and there are no diagnoses or lectures \
about inclusion.
6. A teacher could use it in class as it is: right for the students' age and \
level, in natural Serbian in the script of the question, as long as it needs \
to be."""

_PAIR = """\
{title}
<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer>
"""

_HINT = (
    "\nCode found these words with a letter from another script or alphabet in "
    "the answer{where}: {words}\n"
)

PICK_PROMPT = """\
Below is a lesson from a Petlja course and a question a teacher asked about it.  \
An AI assistant answered the question {k} times.  Each time it first thought \
the problem through, then wrote its answer.

Which one is the best?  Judge the thinking and the answer together.  The best \
one understands what the teacher asked, thinks it through correctly, and ends \
in an answer the teacher can use.  What makes an answer usable:

""" + GOOD_ANSWER + """

The lesson:
{system_message}

The teacher's question:
{question}

{pairs}
Say in a sentence why, then give the number of the best one.
"""

GRADE_PROMPT = """\
Below is a lesson from a Petlja course, a question a teacher asked about it, \
and how an AI assistant handled it: first its thinking, then its answer.

What makes an answer usable:

""" + GOOD_ANSWER + """

Grade it:
- A: you would hand it to the teacher as it is.  Anything you would still \
change is spelling, a typo, or optional.
- B+: sound thinking and a good answer that needs a few small, local changes: \
a word or a sentence here and there, its Serbian.
- B-: sound thinking, but the answer needs corrections in a few specific \
places: a wrong fact, result or key; a short part the teacher asked for and \
didn't get; a section that isn't in the lesson or wasn't asked for, to cut.  \
The rest of the answer stays as it is.
- drop: the answer would have to be rewritten rather than corrected: a new \
structure, most of its content replaced, or a conclusion of the thinking \
reversed.  Or the thinking itself misunderstands the question, reasons its \
way to wrong content, or goes nowhere.

The answer should be in Serbian, in the script of the question, natural and \
free of Russian, Croatian or English words where Serbian has its own.

For B+ and B-, list what to improve, one item each, concretely enough to act \
on ("Question 3's key is wrong: 2**3 is 8, not 6"; "Cut the closing tips, \
nobody asked for them").  For every grade, list the spelling fixes: the wrong \
word exactly as it appears in the answer, and the right one.
{hint}
The lesson:
{system_message}

The teacher's question:
{question}

{pair}
Say in a sentence or two why, then give the grade, the improvements and the \
spelling fixes.
"""

IMPROVE_PROMPT = """\
Below is a lesson from a Petlja course, a question a teacher asked about it, \
and how an AI assistant handled it: its thinking, then its answer.  A reviewer \
found what needs improving in the answer.

Improve the answer as the reviewer says, and keep everything else as it is: \
the same approach, structure and wording wherever nothing needs to change.  \
The thinking stays as it is, so the improved answer must still be the one that \
thinking leads to.  Write in natural Serbian, in the script of the question.

What needs improving:
{improvements}

The lesson:
{system_message}

The teacher's question:
{question}

{pair}
Return only the improved answer, in Markdown.
"""


class PickResult(BaseModel):
    why: str
    best: int


class Fix(BaseModel):
    wrong: str
    right: str


class GradeResult(BaseModel):
    why: str
    grade: Literal["A", "B+", "B-", "drop"]
    improvements: list[str]
    spelling: list[Fix]


def record_key(record: dict) -> tuple[str, str]:
    return (record.get("source", ""), record.get("question", ""))


def group_pairs(records: list[dict]) -> list[dict]:
    """One question per (source, question), its samples in sample order."""
    questions: dict[tuple[str, str], dict] = {}
    for r in records:
        q = questions.setdefault(record_key(r), {
            "source": r["source"], "question": r["question"],
            "system_message": r["system_message"], "pairs": [],
        })
        q["pairs"].append({
            "sample": r.get("sample", 0),
            "answer": r.get("answer") or "",
            "reasoning": r.get("reasoning") or "",
        })
    for q in questions.values():
        q["pairs"].sort(key=lambda x: x["sample"])
    return list(questions.values())


def check_pairs(question: dict) -> None:
    """Code's findings on every pair, in place."""
    for pair in question["pairs"]:
        pair["faults"] = oc.hard_faults(question["question"], pair["answer"], pair["reasoning"])
        pair["script_words"] = oc.script_words(pair["answer"])
        pair["metrics"] = oc.metrics(pair["answer"], pair["reasoning"])


def _seed(key: tuple[str, str]) -> int:
    return int(hashlib.md5("\0".join(key).encode("utf-8")).hexdigest()[:8], 16)


def _pair_block(pair: dict, title: str) -> str:
    return _PAIR.format(title=title, reasoning=pair["reasoning"], answer=pair["answer"])


def _call(client: Any, what: str, model: str, content: str, response_format: Any = None):
    kwargs = {"model": model, **_temperature_kwargs(None), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]}
    if response_format is None:
        return _with_backoff(what, client.chat.completions.create, **kwargs)
    return _with_backoff(what, client.beta.chat.completions.parse, response_format=response_format, **kwargs)


def _usage(response: Any) -> Counter:
    u = response.usage
    return Counter({"in": u.prompt_tokens, "out": u.completion_tokens}) if u else Counter()


def pick(client: Any, question: dict, *, model: str) -> dict:
    """The pick call, over the pairs code let through.  Skipped for one pair."""
    left = [p for p in question["pairs"] if not p["faults"]]
    if len(left) == 1:
        return {"picked": left[0]["sample"], "pick_why": "the only pair left", "usage": Counter()}
    random.Random(_seed(record_key(question))).shuffle(left)
    content = PICK_PROMPT.format(
        k=len(left), system_message=question["system_message"], question=question["question"],
        pairs="\n".join(_pair_block(p, f"=== Answer {n} ===") for n, p in enumerate(left, start=1)),
    )
    usage = Counter()
    for attempt in (1, 2):
        response = _call(client, "bestof/pick", model, content, PickResult)
        usage += _usage(response)
        parsed = response.choices[0].message.parsed
        if parsed and 1 <= parsed.best <= len(left):
            return {"picked": left[parsed.best - 1]["sample"], "pick_why": parsed.why, "usage": usage}
        logger.warning("pick: malformed result on attempt %d for %s", attempt, question["source"])
    raise ValueError(f"no well-formed pick for {question['source']!r}")


def grade(client: Any, question: dict, pair: dict, *, model: str) -> tuple[GradeResult, Counter]:
    words = pair["script_words"]
    content = GRADE_PROMPT.format(
        hint=_HINT.format(where="", words=", ".join(words)) if words else "",
        system_message=question["system_message"], question=question["question"],
        pair=_pair_block(pair, "The assistant's thinking and answer:"),
    )
    response = _call(client, "bestof/grade", model, content, GradeResult)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"no grade for {question['source']!r}")
    return parsed, _usage(response)


# The grader now and then spills the JSON of its own spelling list into the
# improvements ('spelling\':[{', '"wrong":"...","right":"..."}'): not an
# instruction, and it would reach the improve call as one.
_SPILL = re.compile(r'^\W*(improvements|spelling)\W*$|"(wrong|right)"\s*:|\'?spelling\'?\s*:\s*\[', re.I)


def judge_question(client: Any, question: dict, *, model: str) -> dict:
    """Pick, then grade.  Returns the fields to merge into the judged record."""
    out = pick(client, question, model=model)
    pair = next(p for p in question["pairs"] if p["sample"] == out["picked"])
    result, usage = grade(client, question, pair, model=model)
    return {
        "picked": out["picked"], "pick_why": out["pick_why"],
        "grade": result.grade, "grade_why": result.why,
        "improvements": [i for i in result.improvements if i.strip() and not _SPILL.search(i)],
        "spelling": [f.model_dump() for f in result.spelling],
        "usage": dict(out["usage"] + usage),
    }


def apply_fixes(answer: str, fixes: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Spelling fixes, outside code only.  (text, applied, skipped).

    A fix is skipped if its `wrong` is not in the prose, if it changes a digit,
    or if it is longer than a phrase: those are not spelling fixes.
    """
    spans, last = [], 0
    for m in oc.CODE.finditer(answer):
        spans += [(False, answer[last : m.start()]), (True, m.group(0))]
        last = m.end()
    spans.append((False, answer[last:]))
    applied, skipped = [], []
    for fix in fixes:
        wrong, right = fix["wrong"], fix["right"]
        if (not wrong.strip() or wrong == right or len(wrong) > 80
                or NUMBER.findall(wrong) != NUMBER.findall(right)):
            skipped.append(fix)
            continue
        pattern = re.compile(
            (r"(?<!\w)" if wrong[0].isalnum() else "") + re.escape(wrong)
            + (r"(?!\w)" if wrong[-1].isalnum() else "")
        )
        hits = 0
        for i, (is_code, text) in enumerate(spans):
            if not is_code:
                text, n = pattern.subn(lambda _m: right, text)
                spans[i] = (False, text)
                hits += n
        (applied if hits else skipped).append(fix)
    return "".join(t for _, t in spans), applied, skipped


def improve(client: Any, question: dict, pair: dict, answer: str, improvements: list[str],
            *, model: str) -> tuple[str, Counter]:
    content = IMPROVE_PROMPT.format(
        improvements="\n".join(f"- {i}" for i in improvements),
        system_message=question["system_message"], question=question["question"],
        pair=_pair_block({**pair, "answer": answer}, "The assistant's thinking and answer:"),
    )
    response = _call(client, "bestof/improve", model, content)
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("improve returned an empty answer")
    return text, _usage(response)


def improve_faults(question: str, grade_: str, before: str, after: str) -> list[str]:
    """What the improvement left wrong or changed too much."""
    low, high = IMPROVE_BOUNDS[grade_]
    ratio = len(after) / max(len(before), 1)
    why = [] if low <= ratio <= high else [f"length-{ratio:.2f}"]
    why += oc.answer_faults(question, after)
    if oc.script_words(after):
        why.append("script")
    return why


def written_row(rec: dict, pair: dict, final: str, tier: str, model: str, **extra: Any) -> dict:
    return {
        "source": rec["source"],
        "system_message": rec["system_message"],
        "question": rec["question"],
        "answer": pair["answer"],
        # Decorative emojis stripped, as run-02 did; the trace is never touched.
        "improved_answer": oc.PICTO.sub("", final).strip(),
        "reasoning": pair["reasoning"],
        "mode": "think",
        "tier": tier,
        "sample": pair["sample"],
        "grade_why": rec.get("grade_why", ""),
        "improvements": rec.get("improvements", []),
        "judge": JUDGE_NAME,
        "model": model,
        **extra,
    }


def _read(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


@click.command("judge-bestof")
@click.option("--answers", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="gen-answers output with --samples and --keep-reasoning.")
@click.option("--judged-output", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Pick and grade: one row per question, every pair with code's findings.")
@click.option("--written-output", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="The kept rows, for make-sft.  Rows whose improvement failed its "
                   "check go to the same name with .drops.jsonl.")
@click.option("--model", "-m", default=DEFAULT_JUDGE_MODEL, show_default=True)
@click.option("--concurrency", type=click.IntRange(1, 64), default=8, show_default=True)
@click.option("--judge-only", is_flag=True, help="Pick and grade, then stop.")
@click.option("--limit", type=click.IntRange(1), default=None, help="Only the first N questions.")
def judge_bestof(
    answers: Path,
    judged_output: Path,
    written_output: Path,
    model: str,
    concurrency: int,
    judge_only: bool,
    limit: int | None,
) -> None:
    """Pick each question's best thinking and answer of N, grade it, improve it."""
    groups = group_pairs(read_jsonl(answers))
    if limit:
        groups = groups[:limit]
    client = build_teacher_client()
    click.echo(f"{len(groups)} questions, {sum(len(g['pairs']) for g in groups)} pairs; judge {model}")

    # Checks, pick, grade.
    done = {record_key(r) for r in _read(judged_output)}
    todo = [g for g in groups if record_key(g) not in done]
    calls = []
    for g in todo:
        check_pairs(g)
        g.update(judge=JUDGE_NAME, model=model)
        if any(not p["faults"] for p in g["pairs"]):
            calls.append(g)
        else:
            g.update(picked=None, grade="drop", grade_why="every pair failed the code checks")
            write_jsonl([g], judged_output, append=True)
    # A busy deployment outlasts the backoff ladder of a single call (2 min),
    # so what fails is asked again in later rounds rather than left to a rerun.
    for round_no in range(1, ROUNDS + 1):
        if round_no > 1:
            click.echo(f"  pick + grade: {len(calls)} failed, round {round_no} in {ROUND_PAUSE}s")
            time.sleep(ROUND_PAUSE)
        failures = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(judge_question, client, g, model=model): g for g in calls}
            for i, future in enumerate(as_completed(futures), start=1):
                g = futures[future]
                try:
                    g.update(future.result())
                    write_jsonl([g], judged_output, append=True)
                except Exception as exc:  # noqa: BLE001 - asked again next round
                    failures.append(g)
                    logger.warning("pick/grade failed for %s: %s", g["source"], exc)
                if i % 25 == 0:
                    click.echo(f"  pick + grade: {i}/{len(futures)}")
        calls = failures
        if not calls:
            break
    failed = len(calls)
    click.echo(f"Pick + grade: {len(todo)} new, {len(done)} from before, {failed} failed -> {judged_output}")
    if judge_only:
        return
    if failed:
        raise click.ClickException("pick/grade incomplete; run again to finish it")

    # Improve.
    keys = {record_key(g) for g in groups}
    judged = [r for r in _read(judged_output) if record_key(r) in keys]
    drops_output = written_output.with_suffix(".drops.jsonl")
    done = {record_key(r) for r in _read(written_output) + _read(drops_output)}
    stats = Counter(f"grade {r['grade']}" for r in judged)
    jobs = []
    for rec in judged:
        if rec["grade"] == "drop" or record_key(rec) in done:
            continue
        pair = next(p for p in rec["pairs"] if p["sample"] == rec["picked"])
        fixed, applied, _ = apply_fixes(pair["answer"], rec["spelling"])
        fixed = oc.fix_script(fixed)
        tier, todo_items = rec["grade"], rec["improvements"]
        if tier == "A":
            left = oc.script_words(fixed)
            if not left:
                write_jsonl([written_row(rec, pair, fixed, "A", model, spelling_applied=applied)],
                            written_output, append=True)
                continue
            # A spelling fix the grade missed: improve it as a B+.
            tier, todo_items = "B+", [f"Fix these words: {', '.join(left)}"]
            stats["A with script words left -> B+"] += 1
        jobs.append((rec, pair, fixed, tier, applied, todo_items))
    for round_no in range(1, ROUNDS + 1):
        if round_no > 1:
            click.echo(f"  improve: {len(jobs)} failed, round {round_no} in {ROUND_PAUSE}s")
            time.sleep(ROUND_PAUSE)
        failures = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(improve, client, job[0], job[1], job[2], job[5], model=model): job
                       for job in jobs}
            for i, future in enumerate(as_completed(futures), start=1):
                rec, pair, fixed, tier, applied, _ = job = futures[future]
                try:
                    text, usage = future.result()
                except Exception as exc:  # noqa: BLE001 - asked again next round
                    failures.append(job)
                    logger.warning("improve failed for %s: %s", rec["source"], exc)
                    continue
                text = oc.fix_script(text)
                why = improve_faults(rec["question"], tier, fixed, text)
                row = written_row(rec, pair, text, tier if not why else "dropped", model,
                                  spelling_applied=applied, usage=dict(usage),
                                  **({"improve_faults": why} if why else {}))
                write_jsonl([row], drops_output if why else written_output, append=True)
                if why:
                    stats[f"{tier} dropped: check"] += 1
                if i % 25 == 0:
                    click.echo(f"  improve: {i}/{len(futures)}")
        jobs = failures
        if not jobs:
            break

    click.echo("Decisions: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    if jobs:
        raise click.ClickException(f"improve incomplete: {len(jobs)} failed; run again to finish it")
    click.echo(f"Written -> {written_output}")
