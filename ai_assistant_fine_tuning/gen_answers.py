"""Step 2: Answer generated questions using a local vLLM model.

`--keep-reasoning` is for a hybrid-reasoning model served with a reasoning
parser (vLLM `--reasoning-parser qwen3`).  The server then returns the thinking
in its own `reasoning` field, apart from `content`, and without the flag that
field is discarded.  With it, each record carries the trace as `reasoning`, and
`make-sft` trains on it.  An answer cut off at `--max-tokens` is dropped under
the flag: a trace that runs out of tokens has no answer after it, or only the
start of one.

`--samples N` asks for N answers per question in one request (the server's `n`,
which shares the prompt's prefill).  Each becomes its own record, numbered by
`sample`, and a cut-off one is dropped on its own.  The records of one question
are written together, so a question's samples are never split across a crash.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import click
from openai import OpenAI

from ai_assistant_fine_tuning.io_jsonl import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_QUESTIONS_DERIVED_FILE,
    DEFAULT_VLLM_API_KEY,
    DEFAULT_VLLM_BASE_URL,
    write_readable_answers,
)

logger = logging.getLogger(__name__)


@click.command("gen-answers")
@click.option(
    "--questions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_QUESTIONS_DERIVED_FILE,
    show_default=True,
    help="Input JSONL file with generated questions (from gen-questions).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ANSWERS_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file path for answers.",
)
@click.option(
    "--base-url",
    default=DEFAULT_VLLM_BASE_URL,
    show_default=True,
    help="Base URL of the local OpenAI-compatible API (vLLM).",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_LOCAL_MODEL,
    show_default=True,
    help="Model name served by the local endpoint.",
)
@click.option(
    "--api-key",
    default=DEFAULT_VLLM_API_KEY,
    show_default=True,
    help="API key for the local endpoint.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output file instead of overwriting.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 64),
    default=8,
    show_default=True,
    help="Number of parallel requests to the local model.",
)
@click.option(
    "--keep-reasoning",
    is_flag=True,
    help="Save the server's reasoning trace as `reasoning`, and drop answers "
         "cut off at --max-tokens. Needs a server with a reasoning parser.",
)
@click.option("--temperature", type=float, default=0.5, show_default=True)
@click.option("--top-p", type=float, default=None, help="Omitted when unset.")
@click.option(
    "--top-k", type=int, default=None,
    help="vLLM sampling extension, sent in extra_body. Omitted when unset.",
)
@click.option(
    "--max-tokens", type=click.IntRange(1), default=8000, show_default=True,
    help="Completion budget, reasoning included.",
)
@click.option(
    "--samples", type=click.IntRange(1, 16), default=1, show_default=True,
    help="Answers per question, each its own record with a `sample` number.",
)
def gen_answers(
    questions: Path,
    output: Path,
    base_url: str,
    model: str,
    api_key: str,
    append: bool,
    concurrency: int,
    keep_reasoning: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int,
    samples: int,
) -> None:
    """Answer questions using a local vLLM model.

    Reads questions JSONL (from gen-questions), sends each question to the
    local model with the context-aware system message, and writes one
    answer record per question.
    """
    question_records = read_jsonl(questions)
    if not question_records:
        click.echo(f"No records found in {questions}")
        raise SystemExit(1)

    total_questions = sum(len(r.get("questions", [])) for r in question_records)
    click.echo(f"Loaded {len(question_records)} source(s), {total_questions} question(s)")
    click.echo(f"Local model: {model} @ {base_url}")
    click.echo(f"Concurrency: {concurrency}")
    click.echo(f"Output: {output}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    txt_output = output.with_suffix(".txt")
    if not append:
        output.write_text("", encoding="utf-8")
        txt_output.write_text("", encoding="utf-8")

    # Flatten all questions into work items
    work_items = []
    for rec in question_records:
        source = rec.get("source", "unknown")
        system_message = rec.get("system_message", "")
        for q in rec.get("questions", []):
            work_items.append((source, system_message, q))

    write_lock = Lock()
    total_answers = 0
    total_errors = 0
    total_cut = 0

    sampling = {"temperature": temperature}
    if top_p is not None:
        sampling["top_p"] = top_p
    if top_k is not None:
        sampling["extra_body"] = {"top_k": top_k}

    if samples > 1:
        sampling["n"] = samples

    def _answer_one(item):
        """The question's records, and how many samples were cut off."""
        source, system_message, q = item
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": q},
            ],
            max_completion_tokens=max_tokens,
            **sampling,
        )
        records, cut = [], 0
        for choice in sorted(completion.choices, key=lambda c: c.index):
            if keep_reasoning and choice.finish_reason == "length":
                cut += 1
                continue
            record = {
                "source": source,
                "system_message": system_message,
                "question": q,
                # Stripped: a server running --reasoning-parser leaves the blank
                # lines where the thinking block was, and this string ships into
                # the SFT/DPO records verbatim.
                "answer": (choice.message.content or "").strip(),
            }
            if keep_reasoning:
                record["reasoning"] = _reasoning(choice.message)
            if samples > 1:
                record["sample"] = choice.index
            records.append(record)
        return records, cut

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_answer_one, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                records, cut = future.result()
                total_cut += cut
                if not records:
                    continue
                with write_lock:
                    write_jsonl(records, output, append=True)
                    write_readable_answers(records, txt_output, append=True)
                    before = total_answers
                    total_answers += len(records)
                    if total_answers // 100 > before // 100:
                        click.echo(f"  Progress: {total_answers}/{len(work_items) * samples}")
            except Exception as exc:
                total_errors += 1
                source = futures[future][0]
                logger.warning("Failed for %s: %s", source, exc)

    cut = f", {total_cut} cut off at {max_tokens} tokens (dropped)" if keep_reasoning else ""
    click.echo(
        f"\nDone. Generated {total_answers} answers, {total_errors} errors{cut} -> {output}"
    )


def _reasoning(message) -> str:
    """The trace a reasoning parser split off the content.

    vLLM names the field `reasoning`; releases before the rename used
    `reasoning_content`.  Neither is part of the OpenAI schema, so the SDK keeps
    them among the message's extra fields.
    """
    extra = message.model_extra or {}
    return (extra.get("reasoning") or extra.get("reasoning_content") or "").strip()
