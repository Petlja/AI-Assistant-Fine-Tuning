"""JSONL read/write shared by every step of the pipeline.

Lived in `batch_util.py` until the Batch API path was removed; the rest of the
pipeline had always used these two functions and nothing else from that module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict[str, Any]], output_path: Path, append: bool = False) -> None:
    """Write records one per line, UTF-8, without escaping Cyrillic.

    A whole-file write goes to a temporary file in the same directory and is
    then moved into place, because `gen-td` checkpoints by rewriting its judged
    file every few hundred records. Writing in place would truncate ~50 MB and
    spend seconds refilling it, and a crash inside that window would leave a
    half-written file -- which is worse than no checkpoint at all, since the
    partial last line makes `read_jsonl` raise and takes resume down with it.
    `os.replace` is atomic on Windows and POSIX alike for a same-directory move,
    so a reader sees either the previous checkpoint or the new one.

    `append=True` is for incremental single-record writes and is left alone:
    there is no whole file to lose, and a torn final line costs one record.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        with output_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    fd, tmp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output_path)
    except BaseException:
        # Leave the previous checkpoint intact, and do not leave litter behind.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
