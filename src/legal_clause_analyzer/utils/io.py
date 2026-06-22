"""Small filesystem I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_json(path: Path, payload: dict | list) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_json(path: Path) -> dict | list:
    """Read JSON from disk."""

    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to JSONL.

    Args:
        path: Output file path.
        rows: Iterable of serializable row dictionaries.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL rows into memory."""

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
