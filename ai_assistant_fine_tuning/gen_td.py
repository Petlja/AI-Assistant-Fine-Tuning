"""Step 3: Generate training data by evaluating answers and routing to SFT or DPO."""

import logging
from pathlib import Path

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batch_util import read_jsonl, write_jsonl
from ai_assistant_fine_tuning.core import (
    DEFAULT_ANSWERS_OUTPUT_FILE,
    DEFAULT_DPO_OUTPUT_FILE,
    DEFAULT_MODEL,
    DEFAULT_SFT_OUTPUT_FILE,
    route_evaluation,
    write_readable_evaluation,
)
from ai_assistant_fine_tuning.scoring import (
    DEFAULT_THRESHOLD,
    get_scorer,
    score_answer,
)

logger = logging.getLogger(__name__)


@click.command("gen-td")
@click.option(
    "--answers",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_ANSWERS_OUTPUT_FILE,
    show_default=True,
    help="Input JSONL file with answers (from gen-answers).",
)
@click.option(
    "--sft-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file for SFT training data (good answers).",
)
@click.option(
    "--dpo-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_DPO_OUTPUT_FILE,
    show_default=True,
    help="Output JSONL file for DPO training data (bad answers).",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI teacher model for evaluation.",
)
@click.option(
    "--threshold",
    type=click.IntRange(1, 100),
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Minimum score (1-100) for an answer to be routed to SFT. Below → DPO.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output files instead of overwriting.",
)
def gen_td(
    answers: Path,
    sft_output: Path,
    dpo_output: Path,
    model: str,
    threshold: int,
    append: bool,
) -> None:
    """Evaluate answers using a teacher model and route to SFT or DPO.

    Reads answers JSONL (from gen-answers), asks the teacher model to
    score each answer (1-100) and provide an improved version.

    - Score >= threshold  →  SFT record (system/user/assistant)
    - Score < threshold   →  DPO record (prompt, chosen=improved, rejected=original)
    """
    answer_records = read_jsonl(answers)
    if not answer_records:
        click.echo(f"No records found in {answers}")
        raise SystemExit(1)

    click.echo(f"Loaded {len(answer_records)} answer(s) to evaluate")
    click.echo(f"Teacher model: {model}")
    click.echo(f"Threshold: {threshold}/100 (>= SFT, < DPO)")

    client = OpenAI()
    scorer = get_scorer(model=model)

    sft_records: list[dict] = []
    dpo_records: list[dict] = []
    readable_items: list[dict] = []

    if not append:
        sft_output.write_text("", encoding="utf-8")
        dpo_output.write_text("", encoding="utf-8")

    for i, rec in enumerate(answer_records, 1):
        source = rec.get("source", "unknown")
        system_message = rec.get("system_message", "")
        question = rec.get("question", "")
        answer = rec.get("answer", "")

        click.echo(f"[{i}/{len(answer_records)}] Evaluating ({source}) ... ", nl=False)

        try:
            result = score_answer(
                client,
                system_message=system_message,
                question=question,
                answer=answer,
                config=scorer,
            )

            sft_rec, dpo_rec = route_evaluation(
                system_message, question, answer, result, threshold,
            )

            routed_to = "SFT" if sft_rec else "DPO"
            click.echo(f"score={result.score}/{result.scale_max} → {routed_to}")

            if sft_rec:
                sft_records.append(sft_rec)
                write_jsonl([sft_rec], sft_output, append=True)
            if dpo_rec:
                dpo_records.append(dpo_rec)
                write_jsonl([dpo_rec], dpo_output, append=True)

            readable_items.append({
                "source": source,
                "question": question,
                "answer": answer,
                "score": result.score,
                "improved_answer": result.improved_answer,
                "routed_to": routed_to,
            })
        except Exception as exc:
            click.echo(f"error: {exc}")
            logger.warning("Failed to evaluate Q from %s: %s", source, exc)

    # Write readable summary
    eval_txt = sft_output.with_name("evaluation.txt")
    if not append:
        eval_txt.write_text("", encoding="utf-8")
    if readable_items:
        write_readable_evaluation(readable_items, eval_txt, append=True)

    click.echo(f"\nDone. SFT: {len(sft_records)} records -> {sft_output}")
    click.echo(f"      DPO: {len(dpo_records)} records -> {dpo_output}")
