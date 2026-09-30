"""Build training files from a judged file.  Nothing here calls an API.

`judge` pays for the expensive part -- the teacher's answer beside the model's --
and stops.  These two commands turn that file into whatever a run needs, as often
as you like: the shape of a training file costs nothing, so it must never be
welded to the pass that cost hours.

Every row in a judged file carries both answers, so **either command can use
either kind of row**:

    make-sft   target is the teacher's answer.  The model's is ignored.
    make-dpo   chosen is the teacher's answer, rejected is the model's.

`--rows` picks which modes to take.  It matters more for DPO than for SFT: a
`repair` pair differs by deletions and corrections only, so the two sides are
close and the model has to learn something to tell them apart, while a `rewrite`
pair is two unrelated texts with the rejected side several times longer.  A run
on rewrite pairs alone reached zero loss in 40 of 219 steps and learnt nothing
after that.

`--shape` decides what SFT trains on, and it is not cosmetic:

    prompt-completion   TRL resolves completion_only_loss to True -- loss on the
                        answer tokens only
    messages            language modelling -- loss over the whole sequence,
                        lesson material included

The lesson material is handed to the model at inference and never generated, and
it is roughly three quarters of the tokens, so `messages` spends most of the
gradient on a task the model never performs.  Measured: a run under `messages`
lost 4.4 points of belebele-sr against the base model, where the equivalent
`prompt-completion` run lost 0.8.

`--no-think-loss` is for hybrid-reasoning models such as Qwen3.  The targets
carry no reasoning, so the chat template renders each one after an empty
`<think>\\n\\n</think>\\n\\n` block, and by default TRL counts that block as part
of the completion: the model is trained to skip thinking.  The flag writes
`chat_template_kwargs: {"enable_thinking": false}` on every row, which TRL
passes to the template, so the empty block moves into the prompt and the loss
covers the answer only (the same effect as ms-swift's `ignore_empty_think`).

A row whose judged record carries a reasoning trace (`reasoning`) is the
exception: it is rendered with `enable_thinking: true`, so the prompt ends at
`<|im_start|>assistant\\n` and the completion is `<think>\\n` trace `\\n</think>`
followed by the answer.  The trace is trained, not masked: it is what the model
learns to think from.  A file can mix both kinds of row.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from ai_assistant_fine_tuning.core import (
    DEFAULT_DPO_OUTPUT_FILE,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_SFT_OUTPUT_FILE,
)
from ai_assistant_fine_tuning.io_jsonl import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.judging import as_dpo_record, as_sft_record, has_trace

DEFAULT_WRITTEN_INPUT_FILE = "written.jsonl"

_ROWS = click.Choice(["all", "repair", "rewrite"])


def _select(records: list[dict], rows: str, *required: str) -> list[dict]:
    """Rows of the requested mode whose required fields all carry text.

    A blank answer cannot train anything: an SFT row built from one teaches the
    model to say nothing, and a DPO pair with an empty side holds no preference.
    """
    kept = []
    for record in records:
        if rows != "all" and record.get("mode") != rows:
            continue
        if all(str(record.get(field) or "").strip() for field in required):
            kept.append(record)
    return kept


def _report(records: list[dict], kept: list[dict], output: Path) -> None:
    modes: dict[str, int] = {}
    for record in kept:
        mode = record.get("mode", "?")
        modes[mode] = modes.get(mode, 0) + 1
    breakdown = ", ".join(f"{count} {mode}" for mode, count in sorted(modes.items()))
    dropped = len(records) - len(kept)
    if dropped:
        click.echo(f"Skipped {dropped} of {len(records)} rows (other mode, or blank answer)")
    click.echo(f"Wrote {len(kept)} rows ({breakdown}) -> {output}")


def _cap_by_tokens(
    rows: list[dict], max_tokens: int, tokenizer_model: str, describe
) -> list[dict]:
    """Drop rows whose rendered length exceeds a token budget.

    Dropped rather than truncated: truncation removes the end of the sequence,
    which is the answer, so a truncated row trains the model to stop mid
    sentence.  Measured through the model's own chat template, because a budget
    in characters would cut a different set of rows than `--max-seq-len` does.
    """
    try:
        from transformers import AutoTokenizer  # heavy, and only needed here
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise click.ClickException(
            "--max-tokens needs `transformers` to measure length through the "
            "model's chat template, and it is not installed in this "
            "environment. The data-generation venv on Windows deliberately has "
            "no torch/transformers. Either drop --max-tokens and cap the rows "
            "later, or run this command where training runs (the GPU box venv)."
        ) from exc

    tok = AutoTokenizer.from_pretrained(tokenizer_model)
    kept, dropped = [], 0
    for row in rows:
        length = len(tok.apply_chat_template(describe(row), tokenize=True))
        if length > max_tokens:
            dropped += 1
            continue
        kept.append(row)
    if dropped:
        click.echo(f"Dropped {dropped} rows over {max_tokens} tokens")
    return kept


def _written_option(func):
    return click.option(
        "--written",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=DEFAULT_WRITTEN_INPUT_FILE,
        show_default=True,
        help="Judged file from `judge` to build from.",
    )(func)


def _rows_option(func):
    return click.option(
        "--rows",
        type=_ROWS,
        default="all",
        show_default=True,
        help="Which judged rows to use, by the mode `judge --write` gave them.",
    )(func)


def _cap_options(func):
    func = click.option(
        "--tokenizer-model",
        default=DEFAULT_LOCAL_MODEL,
        show_default=True,
        help="Model whose chat template measures --max-tokens.",
    )(func)
    return click.option(
        "--max-tokens",
        type=click.IntRange(1),
        default=None,
        help="Drop rows longer than this once rendered. Omit to keep every row.",
    )(func)


@click.command("make-sft")
@_written_option
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_OUTPUT_FILE,
    show_default=True,
    help="Where to write the SFT rows.",
)
@_rows_option
@click.option(
    "--shape",
    type=click.Choice(["prompt-completion", "messages"]),
    default="prompt-completion",
    show_default=True,
    help="prompt-completion trains on the answer only; messages trains on the "
         "whole sequence, lesson material included.",
)
@click.option(
    "--no-think-loss",
    is_flag=True,
    help="Render rows with enable_thinking=false, so the empty think block a "
         "hybrid-reasoning template adds sits in the prompt and gets no loss. "
         "Rows with a reasoning trace are rendered with thinking on either way. "
         "prompt-completion only.",
)
@_cap_options
def make_sft(
    written: Path,
    output: Path,
    rows: str,
    shape: str,
    no_think_loss: bool,
    max_tokens: int | None,
    tokenizer_model: str,
) -> None:
    """Build an SFT file from a judged file, using the teacher's answer as target."""
    if no_think_loss and shape == "messages":
        # In a full conversation the template puts the empty block inside the
        # final assistant turn whatever enable_thinking says, and `messages`
        # trains on every token anyway.
        raise click.UsageError("--no-think-loss needs --shape prompt-completion")
    records = read_jsonl(written)
    click.echo(f"Read {len(records)} judged rows from {written}")
    kept = _select(records, rows, "system_message", "question", "improved_answer")

    if max_tokens:
        kept = _cap_by_tokens(
            kept,
            max_tokens,
            tokenizer_model,
            lambda r: as_sft_record(r)["messages"],
        )

    built = []
    for record in kept:
        messages = as_sft_record(record)["messages"]
        if shape == "messages":
            built.append({"messages": messages})
        else:
            row = {"prompt": messages[:2], "completion": messages[2:]}
            if has_trace(record):
                row["chat_template_kwargs"] = {"enable_thinking": True}
            elif no_think_loss:
                row["chat_template_kwargs"] = {"enable_thinking": False}
            built.append(row)

    write_jsonl(built, output)
    _report(records, kept, output)
    traced = sum(has_trace(r) for r in kept)
    click.echo(f"Shape: {shape}" + (", empty think block in the prompt" if no_think_loss else ""))
    if traced:
        click.echo(f"{traced} rows train a reasoning trace (thinking on)")


@click.command("make-dpo")
@_written_option
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_DPO_OUTPUT_FILE,
    show_default=True,
    help="Where to write the DPO pairs.",
)
@_rows_option
@_cap_options
def make_dpo(
    written: Path,
    output: Path,
    rows: str,
    max_tokens: int | None,
    tokenizer_model: str,
) -> None:
    """Build DPO pairs from a judged file: chosen is the teacher, rejected the model.

    `--rows repair` is the one worth reaching for first. Those pairs differ by
    faults only, with no length gap between the sides, so the objective cannot be
    satisfied in the first few dozen steps the way it is on rewrite pairs.
    """
    records = read_jsonl(written)
    click.echo(f"Read {len(records)} judged rows from {written}")
    kept = _select(
        records, rows, "system_message", "question", "improved_answer", "answer"
    )

    # A pair whose sides are identical -- the teacher changed nothing -- puts the
    # DPO loss at log 2 with a zero gradient. It costs a step and teaches nothing.
    unchanged = [
        r for r in kept
        if str(r["improved_answer"]).strip() == str(r["answer"]).strip()
    ]
    if unchanged:
        kept = [r for r in kept if r not in unchanged]
        click.echo(f"Dropped {len(unchanged)} pairs the teacher left unchanged")

    if max_tokens:
        kept = _cap_by_tokens(
            kept,
            max_tokens,
            tokenizer_model,
            # The longer side is what has to fit.
            lambda r: as_dpo_record(r)["prompt"] + as_dpo_record(r)["rejected"],
        )

    built = [as_dpo_record(record) for record in kept]
    write_jsonl(built, output)
    _report(records, kept, output)
    if built:
        preview = json.dumps(built[0], ensure_ascii=False)[:120]
        click.echo(f"First pair: {preview}...")
