#!/usr/bin/env python3
"""Split a CIF input list into size-balanced Slurm shard lists."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
from pathlib import Path


def _load_inputs(path: Path) -> list[str]:
    inputs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            inputs.append(line)
    return inputs


def _strip_mmcif_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in (".cif.gz", ".bcif.gz", ".cif", ".bcif"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _input_assembly_group_key(path: str) -> str:
    stem = _strip_mmcif_suffix(Path(path).name)
    match = re.search(r"^(.{4})(?:[-_].*)?[-_]assembly[-_]?[a-z0-9]+(?:$|[-_])", stem)
    if match:
        return f"pdb:{match.group(1)}"
    return f"path:{path}"


def split_input_list(input_list: Path, shard_dir: Path, num_shards: int) -> list[dict[str, object]]:
    paths = _load_inputs(input_list)
    if not paths:
        raise ValueError(f"No input paths in {input_list}")

    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.txt"):
        old.unlink()

    grouped: dict[str, list[str]] = {}
    for path in paths:
        grouped.setdefault(_input_assembly_group_key(path), []).append(path)

    weights: list[tuple[int, str, list[str]]] = []
    for group_key, group_paths in grouped.items():
        total_size = 0
        for path in group_paths:
            try:
                total_size += os.path.getsize(path)
            except OSError:
                pass
        weights.append((total_size, group_key, group_paths))

    actual_shards = min(max(1, int(num_shards)), len(weights))
    heap: list[tuple[int, int, int, list[str]]] = [(0, idx, 0, []) for idx in range(actual_shards)]
    heapq.heapify(heap)
    for size, _group_key, group_paths in sorted(weights, reverse=True):
        total, idx, group_count, bucket = heapq.heappop(heap)
        bucket.extend(group_paths)
        heapq.heappush(heap, (total + size, idx, group_count + 1, bucket))

    plan: list[dict[str, object]] = []
    for total, idx, group_count, bucket in sorted(heap, key=lambda item: item[1]):
        shard_path = shard_dir / f"shard_{idx:04d}.txt"
        shard_path.write_text("\n".join(bucket) + "\n", encoding="utf-8")
        plan.append({
            "shard": idx,
            "input_count": len(bucket),
            "group_count": group_count,
            "estimated_bytes": total,
            "path": str(shard_path),
        })

    (shard_dir.parent / "split_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--shards", required=True, type=int)
    args = parser.parse_args()

    plan = split_input_list(args.input_list, args.shard_dir, args.shards)
    total_inputs = sum(int(item["input_count"]) for item in plan)
    print(f"Wrote {len(plan)} shard list(s) for {total_inputs} input(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
