from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from cif_parse.export import dump_csv_rows


T = TypeVar("T")


def write_signature_cluster_membership_csv(
    path: str | Path,
    signature_groups: Iterable[tuple[str, list[T]]],
    *,
    observation_id_field: str,
    observation_id: Callable[[T], str],
    extra_fields: Callable[[T], dict[str, Any]] | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for signature_cluster_id, members in signature_groups:
        signature_size = len(members)
        for member in sorted(members, key=observation_id):
            row: dict[str, Any] = {
                "signature_cluster_id": signature_cluster_id,
                "signature_cluster_size": signature_size,
                observation_id_field: observation_id(member),
                "pdb_id": getattr(member, "pdb_id", ""),
                "assembly_id": getattr(member, "assembly_id", "") or "",
                "signature_key": getattr(member, "signature_key", ""),
            }
            if extra_fields is not None:
                row.update(extra_fields(member))
            rows.append(row)
    return dump_csv_rows(path, rows)
