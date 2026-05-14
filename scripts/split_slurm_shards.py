#!/usr/bin/env python3
"""Split a CIF input list into size-balanced Slurm shard lists."""

from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path


def _load_inputs(path: Path) -> list[str]:
    inputs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            inputs.append(line)
    return inputs


def split_input_list(input_list: Path, shard_dir: Path, num_shards: int) -> list[dict[str, object]]:
    paths = _load_inputs(input_list)
    if not paths:
        raise ValueError(f"No input paths in {input_list}")

    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.txt"):
        old.unlink()

    weights: list[tuple[int, str]] = []
    for path in paths:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        weights.append((size, path))

    actual_shards = min(max(1, int(num_shards)), len(paths))
    heap: list[tuple[int, int, list[str]]] = [(0, idx, []) for idx in range(actual_shards)]
    heapq.heapify(heap)
    for size, path in sorted(weights, reverse=True):
        total, idx, bucket = heapq.heappop(heap)
        bucket.append(path)
        heapq.heappush(heap, (total + size, idx, bucket))

    plan: list[dict[str, object]] = []
    for total, idx, bucket in sorted(heap, key=lambda item: item[1]):
        shard_path = shard_dir / f"shard_{idx:04d}.txt"
        shard_path.write_text("\n".join(bucket) + "\n", encoding="utf-8")
        plan.append({
            "shard": idx,
            "input_count": len(bucket),
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
