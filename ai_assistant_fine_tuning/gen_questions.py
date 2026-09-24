"""Step 1: Generate questions from a course, one per segment of an activity.

Two passes:

  1. Per lecture, all its activities at once, on the LOCAL model (free).  Writes
     the lecture `arc` and labels each activity `teaching` or `practice`.  Both
     need a lecture-wide view: "is this activity here to introduce material, or
     to exercise material introduced earlier" is unanswerable one activity at a
     time, and the arc is by definition about the whole lecture.

  2. Per teaching activity, on the TEACHER model.  Cuts the activity into
     consecutive segments and writes one question per segment, where the
     question is one whose answer lives in that part of the material.  The
     lecture arc from pass 1 rides along as input, so the activity can be placed
     in it.

The arc from pass 1 IS the synopsis: written once per lecture, prepended to
every question drawn from it.  Pass 2 writes, per segment, a `context`: a
self-contained explanation of what that question targets.  The context is
written, not sliced -- a raw slice leaves implicit whatever the surrounding
lecture had already established.  Arc + context have to stand alone together:
someone who has never seen the course must be able to answer from them.

No answer is generated here.  Answering happens downstream, against whichever
context that run actually trains on.  Two files, identical shape, one record per
question, differing only in what fills --- ГРАДИВО ---:

  full     the whole activity, capped at CONTEXT_CAP
  derived  the written per-question context

Writing the answer here would write it from the full activity while the derived
run only ever shows the model a paragraph -- training it to assert things its
context does not support.  Run gen-answers once per file to get two answer sets,
each derivable from its own context.

Both passes run concurrently, but only the API calls do: every file write
happens on the main thread as results arrive, so there is no shared writer.
Records therefore land in completion order rather than document order, which
nothing downstream depends on -- JSONL lines are independent and joins are by
`source` + question, never by line number.

Quizzes never appear: structure.json tags them `source_type: "quiz"` and
collect_lectures() skips them.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.io_jsonl import write_jsonl
from ai_assistant_fine_tuning.core import (
    CHARS_PER_TOKEN,
    CONTEXT_CAP,
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_CONCURRENCY,
    DEFAULT_INPUT_DIR,
    DEFAULT_LOCAL_CONCURRENCY,
    DEFAULT_LOCAL_TIMEOUT,
    DEFAULT_MAPPER_MODEL,
    DEFAULT_MODEL,
    DEFAULT_QUESTIONS_DERIVED_FILE,
    DEFAULT_QUESTIONS_FULL_FILE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    MAPPER_BUDGET,
    Activity,
    break_down_activity,
    build_system_message,
    build_teacher_client,
    collect_lectures,
    map_lecture,
)

logger = logging.getLogger(__name__)


def _lecture_prompt_content(activities: list[Activity]) -> tuple[str, bool]:
    """Join a lecture's activities into one pass-1 prompt body.

    A lecture that does not fit one call is sent as openings instead of in full.
    Machinery is not what makes these big -- stripping every directive off the
    whole corpus saves 1% -- so they are genuinely long lectures and there is
    nothing to reclaim.  Each activity gets an equal share of the budget.  The
    arc is a synopsis: titles plus openings still show it every activity and the
    order they come in, which is what it is summarising.  At MAPPER_BUDGET this
    fires on 37 of the 448 lectures in the dataset and the arc is thinner for
    those.

    Returns the prompt body and whether it had to be truncated.
    """
    bodies = [activity.path.read_text(encoding="utf-8") for activity in activities]
    truncated = sum(len(b) for b in bodies) > MAPPER_BUDGET * CHARS_PER_TOKEN
    if truncated:
        share = int(MAPPER_BUDGET * CHARS_PER_TOKEN / len(activities))
        bodies = [
            b if len(b) <= share else b[:share].rsplit("\n", 1)[0] + "\n\n[...]"
            for b in bodies
        ]

    content = "\n\n".join(
        f"<<< ACTIVITY {activity.number} — {activity.title} >>>\n\n{body}"
        for activity, body in zip(activities, bodies)
    )
    return content, truncated


def _tally(totals: list[int], usage) -> None:
    """Add one call's token usage to a running [prompt, completion] total."""
    if usage is not None:
        totals[0] += usage.prompt_tokens
        totals[1] += usage.completion_tokens


@click.command("gen-questions")
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    show_default=True,
    help="One course directory with a structure.json, or a dataset root of them.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("finetunedata/questions"),
    show_default=True,
    help="Output directory for questions_derived.jsonl / questions_full.jsonl.",
)
@click.option(
    "--context",
    type=click.Choice(["both", "derived", "full"], case_sensitive=False),
    default="both",
    show_default=True,
    help="Which context variant(s) to write.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Teacher model for pass 2 (one call per teaching activity).",
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 32),
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Parallel pass-2 requests to the teacher model. Rejections back off.",
)
@click.option(
    "--local-model",
    default=DEFAULT_MAPPER_MODEL,
    show_default=True,
    help="Local model for pass 1 (one call per lecture).",
)
@click.option(
    "--local-concurrency",
    type=click.IntRange(1, 32),
    default=DEFAULT_LOCAL_CONCURRENCY,
    show_default=True,
    help="Parallel pass-1 requests. Capped by the server's --max-num-seqs.",
)
@click.option(
    "--local-base-url",
    default=DEFAULT_VLLM_BASE_URL,
    show_default=True,
    help="Base URL of the local OpenAI-compatible API (vLLM) used by pass 1.",
)
@click.option(
    "--local-api-key",
    default=DEFAULT_VLLM_API_KEY,
    show_default=True,
    help="API key for the local endpoint.",
)
@click.option(
    "--azure-endpoint",
    default=None,
    help="Azure endpoint for the teacher model. [default: CHATAI_AZURE_ENDPOINT]",
)
@click.option(
    "--azure-api-version",
    default=DEFAULT_AZURE_API_VERSION,
    show_default=True,
    help="Azure API version. Versions before 2024-10-21 lack structured outputs.",
)
@click.option(
    "--skip-practice/--include-practice",
    default=True,
    show_default=True,
    help="Drop activities pass 1 labelled 'practice' instead of breaking them down.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output files instead of overwriting.",
)
def gen_questions(
    input_dir: Path,
    output_dir: Path,
    context: str,
    model: str,
    concurrency: int,
    local_model: str,
    local_concurrency: int,
    local_base_url: str,
    local_api_key: str,
    azure_endpoint: str | None,
    azure_api_version: str,
    skip_practice: bool,
    append: bool,
) -> None:
    """Generate questions + context-aware system messages from a course.

    Pass 1 maps each lecture on the local model; pass 2 breaks each teaching
    activity into segments on the teacher model and writes one record per
    segment, in the full-context and derived-context variants.
    """
    courses = collect_lectures(input_dir)
    if not courses:
        click.echo(
            f"No course with a structure.json found in {input_dir}. "
            "Step 1 works per lecture, so it needs the course structure."
        )
        raise SystemExit(1)

    total_lectures = sum(len(lectures) for lectures in courses.values())
    total_activities = sum(
        len(activities)
        for lectures in courses.values()
        for activities in lectures.values()
    )
    click.echo(
        f"{len(courses)} course(s), {total_lectures} lectures, "
        f"{total_activities} non-quiz activities in {input_dir}"
    )
    click.echo(f"pass 1: {local_model} @ {local_base_url} x{local_concurrency}")
    click.echo(f"pass 2: {model} x{concurrency}   skip practice: {skip_practice}")

    output_dir.mkdir(parents=True, exist_ok=True)
    derived_file = output_dir / DEFAULT_QUESTIONS_DERIVED_FILE
    full_file = output_dir / DEFAULT_QUESTIONS_FULL_FILE
    variants = ["derived", "full"] if context == "both" else [context]
    targets = {"derived": derived_file, "full": full_file}
    if not append:
        for variant in variants:
            targets[variant].write_text("", encoding="utf-8")
    click.echo("Output: " + ", ".join(str(targets[v]) for v in variants))

    # A hung call must not sit on the default 600s SDK timeout inside the
    # backoff ladder; max_retries=0 keeps the SDK from multiplying attempts.
    local_client = OpenAI(
        base_url=local_base_url,
        api_key=local_api_key,
        timeout=DEFAULT_LOCAL_TIMEOUT,
        max_retries=0,
    )
    teacher_client = build_teacher_client(azure_endpoint, azure_api_version)

    # ------------------------------------------------------------- pass 1
    arcs: dict[tuple[str, int], str] = {}
    roles: dict[tuple[str, str], str] = {}
    truncated: list[str] = []
    pass1_tokens = [0, 0]
    jobs = [
        (course, lecture_no, activities)
        for course, lectures in courses.items()
        for lecture_no, activities in sorted(lectures.items())
    ]

    def _map_one(job):
        course, lecture_no, activities = job
        content, was_truncated = _lecture_prompt_content(activities)
        lecture_map, usage = map_lecture(local_client, content, lecture_no, local_model)
        return lecture_map, usage, was_truncated

    click.echo(f"\npass 1 — {len(jobs)} lectures")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=local_concurrency) as pool:
        futures = {pool.submit(_map_one, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            course, lecture_no, activities = futures[future]
            try:
                lecture_map, usage, was_truncated = future.result()
            except Exception as exc:
                click.echo(f"  [{done:>3}/{len(jobs)}] {course} {lecture_no}: error: {exc}")
                logger.warning("Pass 1 failed for %s lecture %s: %s", course, lecture_no, exc)
                continue

            arcs[(course, lecture_no)] = lecture_map.arc
            for role in lecture_map.activities:
                roles[(course, role.activity)] = role.kind
            if was_truncated:
                truncated.append(f"{course} {lecture_no}")
            _tally(pass1_tokens, usage)
            labels = " ".join(
                f"{a.number}={roles.get((course, a.number), '?')[:5]}" for a in activities
            )
            click.echo(f"  [{done:>3}/{len(jobs)}] {course} lecture {lecture_no:>2}  {labels}")

    pass1_time = time.time() - t0

    # A course that mapped nothing means the local endpoint is gone, not that
    # its lectures are odd.  Stop here rather than writing empty files and
    # letting an unattended loop carry on spending pass-2 calls.
    unmapped = [c for c in courses if not any(k[0] == c for k in arcs)]
    if unmapped:
        raise click.ClickException(
            f"pass 1 mapped no lecture at all for: {', '.join(unmapped)}. "
            f"The local model at {local_base_url} is most likely unreachable "
            "(SSH tunnel down?). Stopping before pass 2 spends anything."
        )

    teaching: list[tuple[str, Activity]] = [
        (course, activity)
        for course, lectures in courses.items()
        for lecture_no, activities in sorted(lectures.items())
        if (course, lecture_no) in arcs
        for activity in activities
        if not skip_practice or roles.get((course, activity.number)) == "teaching"
    ]
    skipped = total_activities - len(teaching)
    if truncated:
        click.echo(
            f"\nlectures sent as openings (over {MAPPER_BUDGET} tok): "
            + ", ".join(truncated)
        )
    click.echo(
        f"\npass 1 done in {pass1_time:.0f}s — {len(teaching)} teaching, {skipped} skipped"
        f"  [{pass1_tokens[0]} in / {pass1_tokens[1]} out]"
    )

    # ------------------------------------------------------------- pass 2
    total_questions = 0
    failed = 0
    oversize: list[tuple[str, int]] = []
    pass2_tokens = [0, 0]

    def _break_down_one(item):
        course, activity = item
        arc = arcs[(course, int(activity.number.split(".")[0]))]
        content = activity.path.read_text(encoding="utf-8")
        breakdown, usage = break_down_activity(
            teacher_client, content, activity.number, arc, model
        )
        return arc, content, breakdown, usage

    click.echo(f"\npass 2 — {len(teaching)} activities")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_break_down_one, item): item for item in teaching}
        for done, future in enumerate(as_completed(futures), 1):
            course, activity = futures[future]
            label = f"{activity.number:<5} {activity.title[:36]:<36}"
            try:
                arc, content, breakdown, usage = future.result()
            except Exception as exc:
                failed += 1
                click.echo(f"  [{done:>3}/{len(teaching)}] {label} error: {exc}")
                logger.warning("Pass 2 failed for %s %s: %s", course, activity.number, exc)
                continue

            _tally(pass2_tokens, usage)
            if len(content) / CHARS_PER_TOKEN > CONTEXT_CAP and "full" in variants:
                oversize.append(
                    (f"{course} {activity.number}", int(len(content) / CHARS_PER_TOKEN))
                )

            # One record per question per variant.  They differ only in the
            # material block of the system message, so nothing downstream
            # changes.  Written here, on the main thread, never in a worker.
            contexts = {"derived": None, "full": content}
            activity_contexts = {"derived": breakdown.activity_context, "full": None}
            for segment in breakdown.segments:
                contexts["derived"] = segment.context
                for variant in variants:
                    if not contexts[variant]:
                        continue
                    write_jsonl(
                        [
                            {
                                "source": activity.rel,
                                "course": course,
                                "activity": activity.number,
                                "segment_kind": segment.segment_kind,
                                "system_message": build_system_message(
                                    arc, contexts[variant], activity_contexts[variant]
                                ),
                                "questions": [segment.question],
                            }
                        ],
                        targets[variant],
                        append=True,
                    )

            tasks = sum(1 for s in breakdown.segments if s.segment_kind == "task")
            total_questions += len(breakdown.segments)
            click.echo(
                f"  [{done:>3}/{len(teaching)}] {label} {len(breakdown.segments):>2} seg "
                f"({len(breakdown.segments) - tasks} exp, {tasks} task)"
            )

    pass2_time = time.time() - t0
    click.echo(
        f"\nDone in {pass1_time + pass2_time:.0f}s. {total_questions} questions from "
        f"{len(teaching) - failed}/{len(teaching)} activities"
    )
    click.echo(
        f"  pass 1  {pass1_time:>6.0f}s   {pass1_tokens[0]:>9} in / {pass1_tokens[1]:>8} out  (local)"
    )
    click.echo(
        f"  pass 2  {pass2_time:>6.0f}s   {pass2_tokens[0]:>9} in / {pass2_tokens[1]:>8} out  ({model})"
    )
    if failed:
        click.echo(f"  {failed} activities failed; see the warnings above")
    for variant in variants:
        click.echo(f"  {variant:<8} -> {targets[variant]}")
    if oversize:
        click.echo(
            f"\nWARNING: activity over {CONTEXT_CAP} tok -- its full-context record will "
            "not fit the\n         learner's context with room for an answer. Nothing was "
            "dropped or\n         truncated; decide what to do with these:"
        )
        for number, tokens in oversize:
            click.echo(f"  {number}  ~{tokens} tok")
