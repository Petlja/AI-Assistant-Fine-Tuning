"""Step 3: score every answer, then have the teacher improve it.

This command stops at the judged file.  It writes no training data: `make-sft`
and `make-dpo` do that, from the file this produces, and either of them can use
any row here.  Splitting it that way is deliberate -- pass 2 costs hours of
teacher calls, and the shape of a training file costs nothing, so the two must
not be welded together.

Two passes, each with its own output.  Pass 1 scores every answer into
`--scores-output`.  Pass 2 asks the teacher to improve each answer and writes the
result to `--written-output`, one row per question:

    question + the model's answer + the teacher's answer

`--write` decides how the teacher's answer is produced:

    repair   every answer is repaired in place -- faults removed, the author's
             own words kept.  The teacher's answer differs from the model's only
             by deletions and corrections.
    rewrite  every answer is written fresh from the material, without the
             teacher seeing the model's attempt.
    split    the worst `--rewrite-share` by substance is rewritten, the rest
             repaired.

The choice matters most for DPO.  A repaired pair is hard -- same words, faults
gone, no length gap -- while a rewritten pair is two different texts and a model
separates them almost immediately.  See `judging.py` for why the mode cannot be
decided inside a single call.

The two files are separate because the passes are days apart in practice: pass 1
is short and cheap, pass 2 runs for hours and gets stopped and resumed.  Keeping
them apart means the scores cannot be disturbed by a pass-2 checkpoint, and the
pass-2 file is self-contained -- it carries the material, the question, both
scores, the mode, the model's answer and the teacher's, so any training file can
be rebuilt from it alone without joining anything.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_CONCURRENCY,
    build_teacher_client,
)
from ai_assistant_fine_tuning.io_jsonl import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.judging import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_REWRITE_SHARE,
    JUDGE_NAME,
    JudgeConfig,
    load_prompts,
    score_answer,
    split_by_rank,
    write_answer,
)

logger = logging.getLogger(__name__)

DEFAULT_SCORES_OUTPUT_FILE = "scores.jsonl"
DEFAULT_WRITTEN_OUTPUT_FILE = "written.jsonl"

# Records between checkpoints. The as_completed loops run in the main thread, so
# rewriting a file from there needs no lock. A full rewrite costs tens of MB of
# I/O and buys back everything done since the last one -- at 250 that is a few
# minutes of work at risk instead of a whole pass.
CHECKPOINT_EVERY = 250


def record_key(record: dict) -> tuple[str, str]:
    """Identity of one answer.

    `source` repeats across a lesson's questions and `question` could in
    principle repeat across lessons, so neither is enough on its own.
    """
    return (record.get("source", ""), record.get("question", ""))


@click.command("judge")
@click.option(
    "--answers",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_ANSWERS_OUTPUT_FILE,
    show_default=True,
    help="Input JSONL file with answers (from gen-answers).",
)
@click.option(
    "--scores-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SCORES_OUTPUT_FILE,
    show_default=True,
    help="Pass 1 output: every answer with its substance and language scores.",
)
@click.option(
    "--written-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_WRITTEN_OUTPUT_FILE,
    show_default=True,
    help="Pass 2 output: question, the model's answer and the teacher's, "
         "self-contained. Every training file is derived from this one.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_JUDGE_MODEL,
    show_default=True,
    help="Teacher model used for both passes.",
)
@click.option(
    "--prompts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="TOML file with a judge's own name and its three prompts, replacing "
         "the built-in ones. For a batch that needs a different rubric.",
)
@click.option(
    "--write",
    "write_mode",
    type=click.Choice(["repair", "rewrite", "split"]),
    default="split",
    show_default=True,
    help="How the teacher produces its answer. repair: fix every answer in "
         "place, keeping the author's words. rewrite: answer afresh from the "
         "material, ignoring the model's attempt. split: rewrite the worst "
         "--rewrite-share, repair the rest.",
)
@click.option(
    "--rewrite-share",
    type=click.FloatRange(0.0, 1.0),
    default=DEFAULT_REWRITE_SHARE,
    show_default=True,
    help="With --write split, the share of the batch, worst first by substance, "
         "that is rewritten rather than repaired. Ignored otherwise.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 32),
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Parallel requests to the teacher model.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Reuse whatever the two output files already hold. Both are "
         f"checkpointed every {CHECKPOINT_EVERY} records, so an interrupted run "
         "costs minutes rather than a whole pass.",
)
@click.option(
    "--scores-only",
    is_flag=True,
    help="Run pass 1 and stop. Scoring is fast and cheap; writing is neither, "
         "so they are worth running on different days.",
)
@click.option(
    "--limit",
    type=click.IntRange(1),
    default=None,
    help="Only process the first N answers. For trying a change out.",
)
def judge(
    answers: Path,
    scores_output: Path,
    written_output: Path,
    model: str,
    prompts: Path | None,
    write_mode: str,
    rewrite_share: float,
    concurrency: int,
    resume: bool,
    scores_only: bool,
    limit: int | None,
) -> None:
    """Score answers with a teacher model, then have it improve each one.

    Pass 1 scores substance and Serbian separately. Pass 2 writes the teacher's
    answer beside the model's, repairing in place or writing afresh according to
    --write. No training data is produced here: run `make-sft` or `make-dpo` over
    the resulting file, as many times and in as many shapes as you like.
    """
    records = read_jsonl(answers)
    if not records:
        click.echo(f"No records found in {answers}")
        raise SystemExit(1)
    if limit:
        records = records[:limit]

    try:
        config = JudgeConfig(model=model, **(load_prompts(prompts) if prompts else {}))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    scored_before: dict[tuple[str, str], dict] = {}
    written_before: dict[tuple[tuple[str, str], str], dict] = {}
    if resume:
        if scores_output.exists():
            scored_before = {record_key(r): r for r in read_jsonl(scores_output)}
        if written_output.exists():
            # Keyed by mode as well: a record that crossed the cut because
            # --write or --rewrite-share changed carries the wrong kind of
            # answer and has to be written again.
            written_before = {
                (record_key(r), r["mode"]): r for r in read_jsonl(written_output)
            }
        # Scores from two rubrics are not on one scale, and the split ranks
        # them together.  Rows from before the `judge` field existed were all
        # scored by the built-in judge.
        others = {
            r.get("judge", JUDGE_NAME)
            for r in (*scored_before.values(), *written_before.values())
        } - {config.name}
        if others:
            raise click.ClickException(
                f"{scores_output} / {written_output} hold rows judged by "
                f"{', '.join(sorted(others))}, not {config.name}. Write to new "
                "output files, or pass --no-resume to start them over."
            )
        if scored_before or written_before:
            click.echo(
                f"Resuming: {len(scored_before)} scored, "
                f"{len(written_before)} written"
            )

    client = build_teacher_client()
    click.echo(f"{len(records)} answer(s), judge {config.name} on {config.model}")
    if write_mode == "split":
        click.echo(f"--write split: worst {rewrite_share:.0%} by substance "
                   "rewritten, the rest repaired")
    else:
        done = "repaired in place" if write_mode == "repair" else "written afresh"
        click.echo(f"--write {write_mode}: every answer {done}")

    # --- pass 1: score -----------------------------------------------------
    todo = [rec for rec in records if record_key(rec) not in scored_before]
    scored = [scored_before[record_key(rec)] for rec in records
              if record_key(rec) in scored_before]
    click.echo(f"\nPass 1: scoring {len(todo)} ({len(scored)} already scored)")
    failures = 0

    def _score(rec: dict) -> dict | None:
        try:
            result = score_answer(
                client,
                system_message=rec.get("system_message", ""),
                question=rec.get("question", ""),
                answer=rec.get("answer", ""),
                config=config,
            )
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", rec.get("source", "?"), exc)
            return None
        return {
            **rec,
            "substance_score": result.substance_score,
            "language_score": result.language_score,
            "judge": result.scorer,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_score, rec) for rec in todo]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    scored.append(result)
                else:
                    failures += 1
                if i % CHECKPOINT_EVERY == 0:
                    write_jsonl(scored, scores_output)
                click.echo(
                    f"  scored {i}/{len(todo)}  ok {i - failures} failed {failures}",
                    nl=False,
                )
                click.echo("\r", nl=False)
        click.echo(f"  scored {len(scored)}/{len(records)} total, {failures} failed")
        write_jsonl(scored, scores_output)

    if not scored:
        click.echo("Nothing scored successfully.")
        raise SystemExit(1)
    click.echo(f"Scores: {len(scored)} records -> {scores_output}")

    if scores_only:
        click.echo("\n--scores-only: stopping before pass 2.")
        return

    # --- decide the mode of each record ------------------------------------
    if write_mode == "repair":
        repair, rewrite = scored, []
        click.echo(f"\nRepairing all {len(repair)} records")
    elif write_mode == "rewrite":
        repair, rewrite = [], scored
        click.echo(f"\nRewriting all {len(rewrite)} records")
    else:
        repair, rewrite = split_by_rank(scored, rewrite_share)
        cut = rewrite[-1]["substance_score"] if rewrite else None
        click.echo(
            f"\nCut at substance <= {cut}: "
            f"{len(rewrite)} to rewrite, {len(repair)} to repair"
        )

    # --- pass 2: write -----------------------------------------------------
    written = [
        written_before[(record_key(rec), mode)]
        for group, mode in ((repair, "repair"), (rewrite, "rewrite"))
        for rec in group
        if (record_key(rec), mode) in written_before
    ]
    jobs = [
        (rec, mode)
        for group, mode in ((repair, "repair"), (rewrite, "rewrite"))
        for rec in group
        if (record_key(rec), mode) not in written_before
    ]
    click.echo(f"\nPass 2: writing {len(jobs)} ({len(written)} already written)")

    def _write(rec: dict, mode: str) -> dict | None:
        try:
            improved = write_answer(
                client,
                mode=mode,
                system_message=rec.get("system_message", ""),
                question=rec.get("question", ""),
                answer=rec.get("answer", ""),
                config=config,
            )
        except Exception as exc:
            logger.warning("%s failed for %s: %s", mode, rec.get("source", "?"), exc)
            return None
        # Self-contained on purpose: this row alone can become an SFT message
        # list or a DPO pair, with no lookup back into the scores file.
        return {**rec, "mode": mode, "improved_answer": improved}

    write_failures = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_write, rec, mode) for rec, mode in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    written.append(result)
                else:
                    write_failures += 1
                if i % CHECKPOINT_EVERY == 0:
                    write_jsonl(written, written_output)
                click.echo(
                    f"  written {i}/{len(jobs)}  "
                    f"ok {i - write_failures} failed {write_failures}",
                    nl=False,
                )
                click.echo("\r", nl=False)
        click.echo(f"  written {len(written)} total, {write_failures} failed")
    write_jsonl(written, written_output)

    click.echo(f"\nWritten: {len(written)} records -> {written_output}")
    click.echo("Next: make-sft or make-dpo over this file.")
