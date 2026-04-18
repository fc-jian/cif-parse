from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def dump_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write one JSON document and return the resolved output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return output_path


def dump_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write tabular rows as CSV and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write newline-delimited JSON rows and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
    return output_path
