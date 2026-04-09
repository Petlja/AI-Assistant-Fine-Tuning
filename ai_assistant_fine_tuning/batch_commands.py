"""Batch lifecycle commands for OpenAI Batch API."""

from __future__ import annotations

# Setup path for direct script execution (before any relative imports)
import sys
from pathlib import Path

import json
from typing import Any

import click
from openai import OpenAI

from ai_assistant_fine_tuning.batching import (
    load_manifest,
    save_manifest,
    utc_now_iso,
    write_jsonl,
)

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}


def _safe_text_from_file_content(file_content: Any) -> str:
    text_attr = getattr(file_content, "text", None)
    if callable(text_attr):
        return text_attr()
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(file_content, "content", None)
    if callable(content_attr):
        data = content_attr()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    if isinstance(file_content, bytes):
        return file_content.decode("utf-8")

    return str(file_content)


def _update_manifest_batch_state(manifest: dict[str, Any], batch_obj: Any) -> None:
    batch_state = manifest.setdefault("batch", {})
    batch_state.update(
        {
            "id": batch_obj.id,
            "status": batch_obj.status,
            "input_file_id": batch_obj.input_file_id,
            "output_file_id": getattr(batch_obj, "output_file_id", None),
            "error_file_id": getattr(batch_obj, "error_file_id", None),
            "updated_at": utc_now_iso(),
        }
    )

    request_counts = getattr(batch_obj, "request_counts", None)
    if request_counts:
        batch_state["request_counts"] = {
            "total": request_counts.total,
            "completed": request_counts.completed,
            "failed": request_counts.failed,
        }


def _parse_chat_completion_content(line: dict[str, Any]) -> str:
    response = line.get("response") or {}
    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("Missing choices in response body")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        content = "".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing assistant message content")

    return content


def _transform_gentd(json_obj: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    pairs = json_obj.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("gentd output missing 'pairs' list")

    records: list[dict[str, Any]] = []
    readable: list[tuple[str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        readable.append((question, answer))

    if not records:
        raise ValueError("gentd output had no valid question-answer pairs")

    return records, readable


def _transform_gendpo(json_obj: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    pairs = json_obj.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("gendpo output missing 'pairs' list")

    records: list[dict[str, Any]] = []
    readable: list[tuple[str, str, str]] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            continue

        records.append(
            {
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        readable.append((prompt, chosen, rejected))

    if not records:
        raise ValueError("gendpo output had no valid preference pairs")

    return records, readable


def _write_readable_gentd(items: list[dict[str, Any]], output_path: Path, append: bool) -> None:
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            f.write(f"=== Source: {source} ===\n\n")
            for i, (question, answer) in enumerate(item["pairs"], 1):
                f.write(f"Q{i}: {question}\n")
                f.write(f"A{i}: {answer}\n\n")
            f.write("\n")


def _write_readable_gendpo(items: list[dict[str, Any]], output_path: Path, append: bool) -> None:
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for item in items:
            source = item["source"]
            f.write(f"=== Source: {source} ===\n\n")
            for i, (prompt, chosen, rejected) in enumerate(item["pairs"], 1):
                f.write(f"Prompt {i}: {prompt}\n")
                f.write(f"Chosen {i}: {chosen}\n")
                f.write(f"Rejected {i}: {rejected}\n\n")
            f.write("\n")


@click.command("batch-submit")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest JSON generated by gentd/gendpo batch mode.",
)
@click.option(
    "--batch-input",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional override path to batch input JSONL.",
)
def batch_submit(manifest: Path, batch_input: Path | None) -> None:
    """Submit a prepared batch input file to OpenAI Batch API."""
    manifest_obj = load_manifest(manifest)
    batch_input_path = batch_input or Path(manifest_obj["batch_input_file"])
    if not batch_input_path.exists():
        raise click.ClickException(f"Batch input file does not exist: {batch_input_path}")

    client = OpenAI()

    click.echo(f"Uploading batch input: {batch_input_path}")
    with open(batch_input_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")

    click.echo("Creating batch job...")
    batch_obj = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": manifest_obj.get("task", "unknown")},
    )

    manifest_obj.setdefault("batch", {})
    _update_manifest_batch_state(manifest_obj, batch_obj)
    manifest_obj["batch"]["submitted_at"] = utc_now_iso()
    manifest_obj["batch"]["input_file_id"] = uploaded_file.id

    save_manifest(manifest_obj, manifest)

    click.echo(f"Batch submitted: {batch_obj.id}")
    click.echo(f"Status: {batch_obj.status}")


@click.command("batch-status")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest JSON.",
)
def batch_status(manifest: Path) -> None:
    """Check status of an OpenAI batch and refresh manifest state."""
    manifest_obj = load_manifest(manifest)
    batch_id = manifest_obj.get("batch", {}).get("id")
    if not batch_id:
        raise click.ClickException("Manifest does not contain a batch id. Run batch-submit first.")

    client = OpenAI()
    batch_obj = client.batches.retrieve(batch_id)
    _update_manifest_batch_state(manifest_obj, batch_obj)
    save_manifest(manifest_obj, manifest)

    counts = manifest_obj.get("batch", {}).get("request_counts") or {}
    click.echo(f"Batch: {batch_id}")
    click.echo(f"Status: {batch_obj.status}")
    if counts:
        click.echo(
            "Requests: "
            f"total={counts.get('total', 0)} "
            f"completed={counts.get('completed', 0)} "
            f"failed={counts.get('failed', 0)}"
        )


@click.command("batch-fetch")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest JSON.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for final training JSONL output path.",
)
@click.option(
    "--readable-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for readable text output path.",
)
@click.option(
    "--append/--overwrite",
    default=False,
    show_default=True,
    help="Append to existing output files instead of overwriting.",
)
def batch_fetch(
    manifest: Path,
    output_path: Path | None,
    readable_output: Path | None,
    append: bool,
) -> None:
    """Fetch completed batch outputs and transform into training data JSONL."""
    manifest_obj = load_manifest(manifest)
    task = manifest_obj.get("task")
    if task not in {"gentd", "gendpo"}:
        raise click.ClickException("Manifest task must be either 'gentd' or 'gendpo'.")

    batch_id = manifest_obj.get("batch", {}).get("id")
    if not batch_id:
        raise click.ClickException("Manifest does not contain a batch id. Run batch-submit first.")

    client = OpenAI()
    batch_obj = client.batches.retrieve(batch_id)
    _update_manifest_batch_state(manifest_obj, batch_obj)

    if batch_obj.status not in TERMINAL_BATCH_STATES:
        save_manifest(manifest_obj, manifest)
        raise click.ClickException(
            f"Batch is not terminal yet (status={batch_obj.status}). Run batch-status and retry later."
        )

    output_file_id = getattr(batch_obj, "output_file_id", None)
    if not output_file_id:
        save_manifest(manifest_obj, manifest)
        raise click.ClickException("Batch has no output file yet.")

    output_text = _safe_text_from_file_content(client.files.content(output_file_id))
    output_lines: list[dict[str, Any]] = []
    for raw_line in output_text.splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            output_lines.append(json.loads(raw_line))

    requests = manifest_obj.get("requests", [])
    requests_by_id = {req["custom_id"]: req for req in requests}

    system_prompt = manifest_obj.get("system_prompt", "")
    final_records: list[dict[str, Any]] = []
    gentd_readable_items: list[dict[str, Any]] = []
    gendpo_readable_items: list[dict[str, Any]] = []

    for line in output_lines:
        custom_id = line.get("custom_id")
        if not custom_id:
            continue

        manifest_req = requests_by_id.get(custom_id)
        if manifest_req is None:
            continue

        error = line.get("error")
        response = line.get("response") or {}
        status_code = response.get("status_code")

        if error is not None or status_code != 200:
            manifest_req["status"] = "failed_api"
            continue

        try:
            content = _parse_chat_completion_content(line)
            parsed_json = json.loads(content)

            if task == "gentd":
                records, readable_pairs = _transform_gentd(parsed_json, system_prompt)
                gentd_readable_items.append(
                    {
                        "source": manifest_req.get("source_rel_path", custom_id),
                        "pairs": readable_pairs,
                    }
                )
            else:
                records, readable_pairs = _transform_gendpo(parsed_json, system_prompt)
                gendpo_readable_items.append(
                    {
                        "source": manifest_req.get("source_rel_path", custom_id),
                        "pairs": readable_pairs,
                    }
                )

            final_records.extend(records)
            manifest_req["status"] = "succeeded"
            manifest_req["generated_count"] = len(records)
        except Exception:
            manifest_req["status"] = "failed_parse"

    error_file_id = getattr(batch_obj, "error_file_id", None)
    if error_file_id:
        error_text = _safe_text_from_file_content(client.files.content(error_file_id))
        for raw_line in error_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            error_line = json.loads(raw_line)
            custom_id = error_line.get("custom_id")
            req = requests_by_id.get(custom_id)
            if req is not None and req.get("status") != "succeeded":
                req["status"] = "failed_api"

    unresolved = 0
    for req in requests:
        if req.get("status") not in {"succeeded", "failed_api", "failed_parse", "skipped_too_short"}:
            req["status"] = "missing_output"
            unresolved += 1

    final_output_path = output_path or Path(manifest_obj["final_output_file"])
    final_readable_path = readable_output or Path(manifest_obj["final_readable_file"])

    if final_records:
        write_jsonl(final_records, final_output_path, append=append)
        if task == "gentd":
            _write_readable_gentd(gentd_readable_items, final_readable_path, append=append)
        else:
            _write_readable_gendpo(gendpo_readable_items, final_readable_path, append=append)

    manifest_obj["fetched_at"] = utc_now_iso()
    manifest_obj["summary"] = {
        "succeeded_requests": sum(1 for req in requests if req.get("status") == "succeeded"),
        "failed_api_requests": sum(1 for req in requests if req.get("status") == "failed_api"),
        "failed_parse_requests": sum(1 for req in requests if req.get("status") == "failed_parse"),
        "missing_output_requests": unresolved,
        "generated_records": len(final_records),
    }
    save_manifest(manifest_obj, manifest)

    click.echo(f"Fetched {len(output_lines)} batch output line(s)")
    click.echo(f"Generated {len(final_records)} training record(s) -> {final_output_path}")
    click.echo(f"Manifest updated -> {manifest}")


@click.group()
def batch_cli() -> None:
    """Batch API lifecycle commands."""
    pass


batch_cli.add_command(batch_submit)
batch_cli.add_command(batch_status)
batch_cli.add_command(batch_fetch)
