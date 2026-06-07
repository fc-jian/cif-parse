"""Persistent SQLite-backed USalign result cache.

Keys are derived from sequence hashes + source paths so that identical
sequences from different PDB entries are cached independently (structures
may differ due to crystallisation conditions) while identical sequences
within the same subcluster share the same key.

Layout
------
::

    {cache_dir}/
    ├── cache.db          # SQLite: alignment results + task definitions + results
    └── (no loose JSONL files — everything is in the database)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS alignment_cache (
    cache_key       TEXT PRIMARY KEY,
    seq_hash_query  TEXT NOT NULL,
    seq_hash_target TEXT NOT NULL,
    source_query    TEXT NOT NULL,
    source_target   TEXT NOT NULL,
    model           INTEGER NOT NULL DEFAULT 1,
    keep_h          INTEGER NOT NULL DEFAULT 0,
    -- USalign results (NULL = pending / not yet computed)
    tm_score_query  REAL,
    tm_score_target REAL,
    tm_score_max    REAL,
    rmsd            REAL,
    aligned_length  INTEGER,
    shorter_length_coverage  REAL,
    resolved_length_coverage REAL,
    -- metadata
    created_at      REAL NOT NULL,
    queried_at      REAL,
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cache_seq ON alignment_cache(seq_hash_query, seq_hash_target);
CREATE INDEX IF NOT EXISTS idx_cache_source ON alignment_cache(source_query, source_target);

CREATE TABLE IF NOT EXISTS phase1_tasks (
    task_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key       TEXT NOT NULL,
    query_monomer_id TEXT NOT NULL,
    target_monomer_id TEXT NOT NULL,
    query_pdb_path  TEXT NOT NULL,
    target_pdb_path TEXT NOT NULL,
    query_residue_count INTEGER NOT NULL,
    target_residue_count INTEGER NOT NULL,
    query_pdb_size  INTEGER NOT NULL DEFAULT 0,
    target_pdb_size INTEGER NOT NULL DEFAULT 0,
    sequence_cluster_id TEXT NOT NULL,
    subcluster_rep_id TEXT NOT NULL,
    seq_group_idx   INTEGER NOT NULL DEFAULT 0,
    round_num       INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    -- 'pending' | 'cached' | 'prefiltered' | 'completed' | 'failed'
    created_at      REAL NOT NULL,
    UNIQUE(cache_key)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON phase1_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_key ON phase1_tasks(cache_key);
CREATE INDEX IF NOT EXISTS idx_tasks_cluster ON phase1_tasks(sequence_cluster_id);
"""


def _now() -> float:
    return time.time()


def _seq_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode()).hexdigest()[:16]


@lru_cache(maxsize=1 << 20)
def _resolve_cached(path_str: str) -> str:
    return str(Path(path_str).resolve())


def alignment_cache_key(
    *,
    seq_query: str,
    seq_target: str,
    source_query: str,
    source_target: str,
    query_monomer_id: str = "",
    target_monomer_id: str = "",
    query_coordinate_fingerprint: str = "",
    target_coordinate_fingerprint: str = "",
    model: int = 1,
    keep_h: bool = False,
    usalign_mode: str = "monomer",
) -> str:
    """Deterministic 32-char hex key for a unique alignment pair."""
    raw = "|".join([
        _seq_hash(seq_query),
        _seq_hash(seq_target),
        _resolve_cached(source_query),
        _resolve_cached(source_target),
        str(query_monomer_id),
        str(target_monomer_id),
        str(query_coordinate_fingerprint),
        str(target_coordinate_fingerprint),
        str(int(model)),
        "1" if keep_h else "0",
        str(usalign_mode),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def get_fast_temp_dir(prefix: str = "cif_parse", fallback: str | Path | None = None) -> Path:
    """Return a writable temporary directory.

    Checks ``CIF_PARSE_TMPDIR``, then ``/tmp``, falling back to *fallback*.
    """
    candidates: list[str] = []
    env_dir = os.environ.get("CIF_PARSE_TMPDIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("/tmp")
    if fallback:
        candidates.append(str(fallback))
    for base in candidates:
        try:
            p = Path(base) / f"{prefix}_{os.getpid()}"
            p.mkdir(parents=True, exist_ok=True)
            # Verify writable
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
            return p
        except (OSError, PermissionError):
            continue
    raise RuntimeError("No writable temp directory found")


class AlignmentCacheDB:
    """SQLite-backed alignment cache + phase task storage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_CACHE_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AlignmentCacheDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # alignment cache (Phase 2 ↔ Phase 3)
    # ------------------------------------------------------------------

    def cache_lookup(
        self,
        cache_key: str,
    ) -> dict[str, Any] | None:
        """Return cached USalign result, or None if pending/failed."""
        row = self._get_conn().execute(
            "SELECT tm_score_query, tm_score_target, tm_score_max, rmsd, "
            "aligned_length, shorter_length_coverage, resolved_length_coverage, "
            "error_message "
            "FROM alignment_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        if row[7]:  # error_message is not NULL — failed permanently
            return None  # caller should treat as failure, not re-submit
        if row[2] is None:  # tm_score_max is NULL — pending
            return None
        return {
            "tm_score_query": row[0],
            "tm_score_target": row[1],
            "tm_score_min": min(row[0], row[1]),
            "tm_score_max": row[2],
            "rmsd": row[3],
            "aligned_length": row[4],
            "shorter_length_coverage": row[5],
            "resolved_length_coverage": row[6],
            "error_message": row[7],
        }

    def cache_upsert_pending(
        self,
        cache_key: str,
        *,
        seq_hash_query: str = "",
        seq_hash_target: str = "",
        source_query: str = "",
        source_target: str = "",
        model: int = 1,
        keep_h: bool = False,
    ) -> None:
        """Insert a placeholder row indicating a pending alignment."""
        self._get_conn().execute(
            """INSERT OR IGNORE INTO alignment_cache
               (cache_key, seq_hash_query, seq_hash_target, source_query, source_target,
                model, keep_h, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (cache_key, seq_hash_query, seq_hash_target, source_query, source_target,
             model, int(keep_h), _now()),
        )
        self._get_conn().commit()

    def cache_upsert_pending_many(
        self,
        keys: list[str],
        *,
        seq_hash_query: str = "",
        seq_hash_target: str = "",
        source_query: str = "",
        source_target: str = "",
        model: int = 1,
        keep_h: bool = False,
    ) -> None:
        """Batch-insert placeholder rows for many pending alignments."""
        now = _now()
        rows = [(k, seq_hash_query, seq_hash_target, source_query, source_target,
                 model, int(keep_h), now) for k in keys]
        self._get_conn().executemany(
            """INSERT OR IGNORE INTO alignment_cache
               (cache_key, seq_hash_query, seq_hash_target, source_query, source_target,
                model, keep_h, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        self._get_conn().commit()

    def cache_write_result(
        self,
        cache_key: str,
        result: dict[str, Any],
    ) -> None:
        """Write a completed USalign result."""
        self._get_conn().execute(
            """UPDATE alignment_cache SET
               tm_score_query=?, tm_score_target=?, tm_score_max=?, rmsd=?,
               aligned_length=?, shorter_length_coverage=?, resolved_length_coverage=?
               WHERE cache_key=?""",
            (
                result.get("tm_score_query"),
                result.get("tm_score_target"),
                result.get("tm_score_max"),
                result.get("rmsd"),
                result.get("aligned_length"),
                result.get("shorter_length_coverage"),
                result.get("resolved_length_coverage"),
                cache_key,
            ),
        )
        self._get_conn().commit()

    def cache_write_results_many(
        self,
        rows: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Write completed USalign results in one transaction."""
        if not rows:
            return
        self._get_conn().executemany(
            """UPDATE alignment_cache SET
               tm_score_query=?, tm_score_target=?, tm_score_max=?, rmsd=?,
               aligned_length=?, shorter_length_coverage=?, resolved_length_coverage=?,
               error_message=NULL
               WHERE cache_key=?""",
            [
                (
                    result.get("tm_score_query"),
                    result.get("tm_score_target"),
                    result.get("tm_score_max"),
                    result.get("rmsd"),
                    result.get("aligned_length"),
                    result.get("shorter_length_coverage"),
                    result.get("resolved_length_coverage"),
                    cache_key,
                )
                for cache_key, result in rows
            ],
        )
        self._get_conn().commit()

    def cache_write_error(self, cache_key: str, error: str) -> None:
        self._get_conn().execute(
            "UPDATE alignment_cache SET error_message=? WHERE cache_key=?",
            (error, cache_key),
        )
        self._get_conn().commit()

    def cache_write_errors_many(self, rows: list[tuple[str, str]]) -> None:
        """Write failed USalign results in one transaction."""
        if not rows:
            return
        self._get_conn().executemany(
            "UPDATE alignment_cache SET error_message=? WHERE cache_key=?",
            [(error, cache_key) for cache_key, error in rows],
        )
        self._get_conn().commit()

    def cache_get_records(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Return completed or failed cache records for the requested keys."""
        if not keys:
            return {}
        records: dict[str, dict[str, Any]] = {}
        chunk_size = 900
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._get_conn().execute(
                f"""SELECT cache_key, tm_score_query, tm_score_target, tm_score_max,
                           rmsd, aligned_length, shorter_length_coverage,
                           resolved_length_coverage, error_message
                    FROM alignment_cache
                    WHERE cache_key IN ({placeholders})""",
                chunk,
            ).fetchall()
            for row in rows:
                tm_score_query = row[1]
                tm_score_target = row[2]
                records[row[0]] = {
                    "tm_score_query": tm_score_query,
                    "tm_score_target": tm_score_target,
                    "tm_score_min": (
                        min(tm_score_query, tm_score_target)
                        if tm_score_query is not None and tm_score_target is not None
                        else None
                    ),
                    "tm_score_max": row[3],
                    "rmsd": row[4],
                    "aligned_length": row[5],
                    "shorter_length_coverage": row[6],
                    "resolved_length_coverage": row[7],
                    "error_message": row[8],
                }
        return records

    def populate_from(self, other: "AlignmentCacheDB") -> int:
        """Copy all completed entries from another cache DB.  Returns count."""
        rows = other._get_conn().execute(
            "SELECT * FROM alignment_cache WHERE tm_score_max IS NOT NULL"
        ).fetchall()
        if not rows:
            return 0
        cols = [d[0] for d in other._get_conn().execute("PRAGMA table_info(alignment_cache)")]
        placeholders = ",".join(["?"] * len(cols))
        self._get_conn().executemany(
            f"INSERT OR IGNORE INTO alignment_cache ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
        self._get_conn().commit()
        return len(rows)

    def cache_get_completed_keys(self) -> set[str]:
        """Return all cache_keys that have a completed USalign result."""
        rows = self._get_conn().execute(
            "SELECT cache_key FROM alignment_cache WHERE tm_score_max IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}

    def task_get_all_keys(self) -> set[str]:
        """Return all cache_keys currently present in phase1_tasks."""
        rows = self._get_conn().execute(
            "SELECT cache_key FROM phase1_tasks"
        ).fetchall()
        return {r[0] for r in rows}

    # ------------------------------------------------------------------
    # phase 1 task storage
    # ------------------------------------------------------------------

    def task_insert_many(self, rows: list[dict[str, Any]]) -> None:
        """Batch-insert alignment tasks for Phase 2 (idempotent)."""
        self._get_conn().executemany(
            """INSERT OR IGNORE INTO phase1_tasks
               (cache_key, query_monomer_id, target_monomer_id, query_pdb_path,
                target_pdb_path, query_residue_count, target_residue_count,
                query_pdb_size, target_pdb_size, sequence_cluster_id,
                subcluster_rep_id, seq_group_idx, round_num, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    r["cache_key"], r["query_monomer_id"], r["target_monomer_id"],
                    r["query_pdb_path"], r["target_pdb_path"],
                    r["query_residue_count"], r["target_residue_count"],
                    r.get("query_pdb_size", 0), r.get("target_pdb_size", 0),
                    r["sequence_cluster_id"], r["subcluster_rep_id"],
                    r.get("seq_group_idx", 0), r.get("round_num", 0),
                    r.get("status", "pending"), _now(),
                )
                for r in rows
            ],
        )
        self._get_conn().commit()

    def task_update_status(self, task_id: int, status: str) -> None:
        self._get_conn().execute(
            "UPDATE phase1_tasks SET status=? WHERE task_id=?",
            (status, task_id),
        )
        self._get_conn().commit()

    def task_get_pending(self) -> list[dict[str, Any]]:
        """Return all pending tasks, sorted by estimated cost descending."""
        rows = self._get_conn().execute(
            """SELECT task_id, cache_key, query_monomer_id, target_monomer_id,
                      query_pdb_path, target_pdb_path,
                      query_residue_count, target_residue_count,
                      query_pdb_size, target_pdb_size,
                      sequence_cluster_id, subcluster_rep_id
               FROM phase1_tasks WHERE status='pending'
               ORDER BY (query_residue_count * target_residue_count) DESC"""
        ).fetchall()
        return [
            {
                "task_id": r[0], "cache_key": r[1], "query_monomer_id": r[2],
                "target_monomer_id": r[3], "query_pdb_path": r[4], "target_pdb_path": r[5],
                "query_residue_count": r[6], "target_residue_count": r[7],
                "query_pdb_size": r[8], "target_pdb_size": r[9],
                "sequence_cluster_id": r[10], "subcluster_rep_id": r[11],
            }
            for r in rows
        ]

    def task_count_by_status(self) -> dict[str, int]:
        rows = self._get_conn().execute(
            "SELECT status, COUNT(*) FROM phase1_tasks GROUP BY status"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # phase 3: per-group task/result access
    # ------------------------------------------------------------------

    def tasks_for_group(self, sequence_cluster_id: str) -> list[dict[str, Any]]:
        """Return all tasks for a sequence group, ordered by round."""
        rows = self._get_conn().execute(
            """SELECT t.cache_key, t.query_monomer_id, t.target_monomer_id,
                      t.subcluster_rep_id, t.round_num, t.status,
                      t.query_residue_count, t.target_residue_count,
                      a.tm_score_max, a.rmsd, a.aligned_length,
                      a.shorter_length_coverage, a.resolved_length_coverage,
                      a.error_message
               FROM phase1_tasks t
               LEFT JOIN alignment_cache a ON t.cache_key = a.cache_key
               WHERE t.sequence_cluster_id=?
               ORDER BY t.round_num, t.task_id""",
            (sequence_cluster_id,),
        ).fetchall()
        return [
            {
                "cache_key": r[0], "query_monomer_id": r[1], "target_monomer_id": r[2],
                "subcluster_rep_id": r[3], "round_num": r[4], "status": r[5],
                "query_residue_count": r[6], "target_residue_count": r[7],
                "tm_score_max": r[8], "rmsd": r[9], "aligned_length": r[10],
                "shorter_length_coverage": r[11], "resolved_length_coverage": r[12],
                "error_message": r[13],
            }
            for r in rows
        ]

    def task_status_batch_update(self, task_ids: list[int], status: str) -> None:
        self._get_conn().executemany(
            "UPDATE phase1_tasks SET status=? WHERE task_id=?",
            [(status, tid) for tid in task_ids],
        )
        self._get_conn().commit()

    def task_status_batch_update_by_keys(self, cache_keys: list[str], status: str) -> None:
        """Update task status by cache key in one transaction."""
        if not cache_keys:
            return
        self._get_conn().executemany(
            "UPDATE phase1_tasks SET status=? WHERE cache_key=?",
            [(status, cache_key) for cache_key in cache_keys],
        )
        self._get_conn().commit()
