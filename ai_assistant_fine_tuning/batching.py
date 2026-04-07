"""Shared helpers for OpenAI Batch API workflows."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO8601 format."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def short_sha1(text: str, length: int = 16) -> str:
    """Return a short SHA1 hex digest for stable IDs and change detection."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def make_custom_id(task: str, source_rel_path: Path, content: str) -> str:
    """Create a stable per-source custom ID for batch requests."""
    slug = source_rel_path.as_posix()
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", slug).strip("_")
    slug = slug[:48] or "source"
    digest = short_sha1(f"{source_rel_path.as_posix()}\n{content}")
    return f"{task}:{slug}:{digest}"


def write_jsonl(records: list[dict[str, Any]], output_path: Path, append: bool = False) -> None:
    """Write records to JSONL file."""
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def save_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Serialize manifest JSON with deterministic key ordering."""
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load manifest JSON from disk."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def create_chat_batch_request(
    *,
    custom_id: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    """Build one Batch API input line for /v1/chat/completions."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # JSON object output makes post-processing deterministic in batch mode.
            "response_format": {"type": "json_object"},
        },
    }
