from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any


def dump_json(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Write one JSON document and return the resolved output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
    )
    if indent is not None:
        text += "\n"
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        output_path.write_text(text, encoding="utf-8")
    return output_path


def load_json(path: str | Path) -> Any:
    """Read one JSON document from `.json` or `.json.gz`."""

    input_path = resolve_json_path(path)
    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(input_path.read_text(encoding="utf-8"))


def resolve_json_path(path: str | Path) -> Path:
    """Resolve a JSON artifact path, allowing `.json` / `.json.gz` fallback."""

    input_path = Path(path)
    candidates = [input_path]
    if input_path.suffix == ".json":
        candidates.append(input_path.with_name(f"{input_path.name}.gz"))
    elif input_path.suffix == ".gz":
        candidates.append(input_path.with_suffix(""))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"JSON artifact not found: {input_path}")


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
