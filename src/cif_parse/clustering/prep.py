"""Pre-processing database for clustering: consolidated bundles, cif coordinate cache,
and hash-based incremental updates.

Usage
-----
The ``cif-parse-cluster prep`` command builds (or refreshes) a SQLite database
that replaces thousands of individual ``result.json.gz`` reads with a single
SQLite connection.  Parsed mmCIF atom arrays are cached so that structure
extraction can reuse them directly.

When the database is present the clustering pipeline automatically reads from
it; when absent it falls back to the original file-by-file behaviour.  The
database is written in WAL mode so that multiple readers can access it
concurrently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from cif_parse.export import load_case_output_bundles

LOGGER = logging.getLogger(__name__)

DEFAULT_DB_NAME = "clustering_prep.db"

# ── schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -65536;  -- 64 MB

-- Core: stores the full JSON payload of every case bundle, keyed by
-- (case_id, assembly_id).  bundle_json is the *un-gzipped* JSON string.
CREATE TABLE IF NOT EXISTS bundles (
    case_id         TEXT NOT NULL,
    assembly_id     TEXT NOT NULL DEFAULT '',
    pdb_id          TEXT NOT NULL DEFAULT '',
    source_path     TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL,
    bundle_json     TEXT NOT NULL,
    PRIMARY KEY (case_id, assembly_id)
);

-- Keyed by (source_path, assembly_id).  atom_array_blob is a pickled
-- biotite AtomArray.  quality_json stores experimental method, resolution etc.
-- chain_ops_json stores assembly chain operations for multimer extraction.
CREATE TABLE IF NOT EXISTS cif_cache (
    source_path     TEXT NOT NULL,
    assembly_id     TEXT,
    cache_key       TEXT PRIMARY KEY,
    source_hash     TEXT NOT NULL,
    atom_array_blob BLOB,
    quality_json    TEXT,
    chain_ops_json  TEXT
);

-- Tracks which case directories have already been ingested and at which hash.
CREATE TABLE IF NOT EXISTS prep_meta (
    case_dir        TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL,
    ingested_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bundles_case ON bundles(case_id);
CREATE INDEX IF NOT EXISTS idx_bundles_source ON bundles(source_path);
CREATE INDEX IF NOT EXISTS idx_cif_cache_source ON cif_cache(source_path);
"""


# ── hash helpers ─────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """SHA-256 of a single file's content."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):  # 1 MiB
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_case_dir(case_dir: Path) -> str:
    """Stable content hash for a case-output directory.

    We hash the *sorted* list of (filename, file_hash) tuples so that the hash
    is independent of filesystem ordering but still sensitive to content
    changes.
    """
    hasher = hashlib.sha256()
    for child_path in sorted(case_dir.iterdir()):
        if child_path.is_file() and child_path.name.endswith((".json", ".json.gz", ".gz")):
            hasher.update(child_path.name.encode())
            hasher.update(_hash_file(child_path).encode())
    return hasher.hexdigest()


def _hash_source_mtime(source_path: str) -> str:
    """Lightweight hash based on file size + mtime (fast proxy for content hash)."""
    p = Path(source_path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    return hashlib.sha256(f"{p.resolve()}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


# ── database builder ─────────────────────────────────────────────────────────


def _ensure_schema(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the prep database and ensure schema is current."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _ingest_one_case(
    case_dir: Path,
    conn: sqlite3.Connection,
    cif_files_directory: str | None = None,
) -> dict[str, Any]:
    """Ingest a single case-output directory into the database.

    Returns a summary dict with counts, or raises on fatal errors.
    """
    case_id = case_dir.name
    content_hash = _hash_case_dir(case_dir)

    # Check if already ingested with the same hash → skip
    cur = conn.execute("SELECT content_hash FROM prep_meta WHERE case_dir = ?", (str(case_dir.resolve()),))
    row = cur.fetchone()
    if row is not None and row[0] == content_hash:
        return {"case_id": case_id, "status": "skipped", "reason": "unchanged"}

    bundles = load_case_output_bundles(case_dir)
    if not bundles:
        return {"case_id": case_id, "status": "skipped", "reason": "no_bundles"}

    ingested_bundles = 0
    seen_source_paths: set[str] = set()
    for bundle in bundles:
        summary = bundle.get("structure_summary", {})
        source_path = str(summary.get("source_path", "") or "")
        if cif_files_directory:
            source_path = str(Path(cif_files_directory) / Path(source_path).name)
        assembly_ids = [str(a) for a in summary.get("assembly_ids", []) if str(a)]
        if not assembly_ids:
            assembly_ids = [""]

        pdb_id = str(summary.get("pdb_id", "") or "")
        bundle_json = json.dumps(bundle, ensure_ascii=False, sort_keys=True)

        for assembly_id in assembly_ids:
            conn.execute(
                "INSERT OR REPLACE INTO bundles(case_id, assembly_id, pdb_id, source_path, content_hash, bundle_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, assembly_id, pdb_id, source_path, content_hash, bundle_json),
            )
            ingested_bundles += 1

        if source_path:
            seen_source_paths.add(source_path)

    conn.execute(
        "INSERT OR REPLACE INTO prep_meta(case_dir, content_hash, ingested_at) VALUES (?, ?, ?)",
        (str(case_dir.resolve()), content_hash, time.time()),
    )

    return {
        "case_id": case_id,
        "status": "ingested",
        "num_bundles": ingested_bundles,
        "num_source_paths": len(seen_source_paths),
        "source_paths": sorted(seen_source_paths),
    }


def build_prep_database(
    inputs: Iterable[str | Path],
    db_path: str | Path,
    *,
    cif_files_directory: str | None = None,
    prep_jobs: int = 4,
    load_cif_cache: bool = True,
) -> dict[str, Any]:
    """Build (or refresh) the clustering prep database.

    Parameters
    ----------
    inputs: case-output directories or parent directories.
    db_path: path to the SQLite database file.
    cif_files_directory: optional override for mmCIF file locations.
    prep_jobs: number of parallel workers for case ingestion.
    load_cif_cache: if True, also pre-load mmCIF atom arrays into ``cif_cache``.

    Returns a manifest dict with summary statistics.
    """
    from cif_parse.clustering.common import discover_case_output_dirs

    db_path = Path(db_path)
    t0 = time.monotonic()

    case_dirs = discover_case_output_dirs(inputs)
    LOGGER.info("Prepping %d case directories into %s", len(case_dirs), db_path)

    conn = _ensure_schema(db_path)

    # Phase 1: ingest all case bundles into `bundles` table
    all_source_paths: set[str] = set()
    stats = {"total_cases": len(case_dirs), "ingested": 0, "skipped_unchanged": 0, "skipped_no_bundles": 0}
    failures: list[dict[str, str]] = []

    prep_jobs = max(1, int(prep_jobs))
    if prep_jobs <= 1 or len(case_dirs) <= 1:
        for case_dir in tqdm(case_dirs, desc="Ingesting case bundles", unit="case"):
            try:
                result = _ingest_one_case(case_dir, conn, cif_files_directory)
                if result["status"] == "ingested":
                    stats["ingested"] += 1
                    all_source_paths.update(result.get("source_paths", []))
                elif result.get("reason") == "unchanged":
                    stats["skipped_unchanged"] += 1
                else:
                    stats["skipped_no_bundles"] += 1
            except Exception as exc:
                LOGGER.warning("Failed to ingest case %s: %s", case_dir, exc)
                failures.append({"case_dir": str(case_dir), "error": str(exc)})
    else:
        from tqdm.contrib.concurrent import thread_map

        def _ingest(case_dir: Path) -> dict[str, Any]:
            try:
                return _ingest_one_case(case_dir, conn, cif_files_directory)
            except Exception as exc:
                LOGGER.warning("Failed to ingest case %s: %s", case_dir, exc)
                return {"case_id": case_dir.name, "status": "error", "error": str(exc)}

        results = thread_map(_ingest, case_dirs, max_workers=prep_jobs, desc="Ingesting case bundles", unit="case")
        for result in results:
            if result.get("status") == "ingested":
                stats["ingested"] += 1
                all_source_paths.update(result.get("source_paths", []))
            elif result.get("reason") == "unchanged":
                stats["skipped_unchanged"] += 1
            elif result.get("status") == "error":
                failures.append({"case_dir": result.get("case_id", ""), "error": result.get("error", "")})
            else:
                stats["skipped_no_bundles"] += 1

    conn.commit()
    LOGGER.info(
        "Phase 1 complete (%.1fs): %d ingested, %d skipped (unchanged), %d no-bundles, %d unique source paths",
        time.monotonic() - t0,
        stats["ingested"],
        stats["skipped_unchanged"],
        stats["skipped_no_bundles"],
        len(all_source_paths),
    )

    # Phase 2: pre-load mmCIF atom arrays into cif_cache
    cif_stats = {"cached": 0, "skipped": 0}
    if load_cif_cache and all_source_paths:
        t1 = time.monotonic()
        LOGGER.info("Phase 2: caching %d mmCIF source files", len(all_source_paths))

        # collect all (source_path, assembly_id) pairs needed
        cur = conn.execute("SELECT DISTINCT source_path, assembly_id FROM bundles WHERE source_path != ''")
        cache_keys: set[tuple[str, str]] = set()
        for source_path, assembly_id in cur.fetchall():
            if source_path in all_source_paths:
                cache_keys.add((source_path, assembly_id or None))

        if cache_keys:
            from biotite.structure.io.pdbx import get_assembly, get_structure
            from cif_parse.io import read_cif_file

            def _cache_one(args: tuple[str, str | None]) -> dict[str, Any]:
                source_path, assembly_id = args
                cache_key = f"{source_path}__{assembly_id or ''}"
                source_hash = _hash_source_mtime(source_path)
                cur2 = conn.execute("SELECT source_hash FROM cif_cache WHERE cache_key = ?", (cache_key,))
                row = cur2.fetchone()
                if row is not None and row[0] == source_hash:
                    return {"cache_key": cache_key, "status": "skipped"}
                try:
                    cif_file = read_cif_file(source_path)
                    quality_metadata = _read_quality(source_path, cif_file)
                    if assembly_id:
                        atom_array = get_assembly(cif_file, assembly_id=assembly_id, model=1, use_author_fields=False)
                    else:
                        atom_array = get_structure(cif_file, model=1, use_author_fields=False)
                    # Only cache if atom_array is non-empty
                    if atom_array is not None and len(atom_array) > 0:
                        atom_blob = pickle.dumps(atom_array, protocol=pickle.HIGHEST_PROTOCOL)
                        chain_ops_json = _read_chain_ops_json(source_path, assembly_id)
                        conn.execute(
                            "INSERT OR REPLACE INTO cif_cache(source_path, assembly_id, cache_key, source_hash, "
                            "atom_array_blob, quality_json, chain_ops_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (source_path, assembly_id, cache_key, source_hash, atom_blob,
                             json.dumps(quality_metadata) if quality_metadata else None,
                             chain_ops_json),
                        )
                        return {"cache_key": cache_key, "status": "cached"}
                    return {"cache_key": cache_key, "status": "empty"}
                except Exception as exc:
                    LOGGER.warning("Failed to cache mmCIF %s (assembly=%s): %s", source_path, assembly_id, exc)
                    return {"cache_key": cache_key, "status": "error", "error": str(exc)}

            cache_items = sorted(cache_keys)
            if len(cache_items) <= 1:
                for item in tqdm(cache_items, desc="Caching mmCIF structures", unit="cif"):
                    result = _cache_one(item)
                    if result["status"] == "cached":
                        cif_stats["cached"] += 1
                    else:
                        cif_stats["skipped"] += 1
            else:
                from tqdm.contrib.concurrent import thread_map as _thread_map_cache
                results = _thread_map_cache(_cache_one, cache_items, max_workers=min(prep_jobs, 4),
                                            desc="Caching mmCIF structures", unit="cif")
                for result in results:
                    if result.get("status") == "cached":
                        cif_stats["cached"] += 1
                    else:
                        cif_stats["skipped"] += 1
                conn.commit()

        LOGGER.info("Phase 2 complete (%.1fs): %d cached, %d skipped",
                    time.monotonic() - t1, cif_stats["cached"], cif_stats["skipped"])

    conn.commit()

    # Final manifest
    cur = conn.execute("SELECT COUNT(*) FROM bundles")
    total_bundles = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(DISTINCT source_path) FROM bundles WHERE source_path != ''")
    total_source_paths = cur.fetchone()[0]

    elapsed = time.monotonic() - t0
    LOGGER.info("Prep database built (%.1fs): %d bundles, %d source paths, %d failures",
                elapsed, total_bundles, total_source_paths, len(failures))

    return {
        "db_path": str(db_path.resolve()),
        "total_cases": stats["total_cases"],
        "ingested": stats["ingested"],
        "skipped_unchanged": stats["skipped_unchanged"],
        "skipped_no_bundles": stats["skipped_no_bundles"],
        "failures": len(failures),
        "total_bundles": total_bundles,
        "total_source_paths": total_source_paths,
        "cif_cached": cif_stats["cached"],
        "cif_skipped": cif_stats["skipped"],
        "elapsed_seconds": round(elapsed, 1),
    }


# ── quality / chain-ops helpers ──────────────────────────────────────────────


def _read_quality(source_path: str, cif_file=None) -> dict[str, Any] | None:
    """Extract experimental method and resolution from a mmCIF file."""
    try:
        from cif_parse.io import read_cif_file as _read
        from biotite.structure.io.pdbx import CIFBlock, CIFCategory
        cf = cif_file or _read(source_path)
        block = cf.block if isinstance(cf, CIFBlock) else cf
        if hasattr(block, "get"):
            exptl = block.get("_exptl.method") or block.get("_exptl_crystal.method") or ""
            if hasattr(exptl, "as_item"):
                exptl = str(exptl.as_item()) if exptl.as_item() else ""
            elif isinstance(exptl, (list, tuple)):
                exptl = str(exptl[0]) if exptl else ""
            else:
                exptl = str(exptl) if exptl else ""
            resolution = block.get("_refine.ls_d_res_high") or block.get("_reflns.d_resolution_high") or ""
            if hasattr(resolution, "as_item"):
                resolution = resolution.as_item()
            elif isinstance(resolution, (list, tuple)):
                resolution = resolution[0] if resolution else None
            try:
                resolution = float(resolution) if resolution else None
            except (ValueError, TypeError):
                resolution = None
            return {"method": exptl, "resolution": resolution}
    except Exception:
        pass
    return None


def _read_chain_ops_json(source_path: str, assembly_id: str | None) -> str | None:
    """Read assembly chain operations as a JSON string for multimer extraction."""
    try:
        from cif_parse.io import read_assembly_chain_operations
        _, chain_ops = read_assembly_chain_operations(source_path, assembly_id=assembly_id)
        if chain_ops:
            return json.dumps(chain_ops, ensure_ascii=False)
    except Exception:
        pass
    return None


# ── query helpers (used by collect functions) ────────────────────────────────


def open_prep_db(db_path: str | Path) -> sqlite3.Connection | None:
    """Open the prep database if it exists, else return None."""
    p = Path(db_path)
    if not p.exists():
        return None
    conn = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def iter_bundles(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Iterate over all case bundles in the database, yielding parsed dicts.

    Each dict has ``_case_id`` injected so that callers can group by case.
    """
    cur = conn.execute("SELECT case_id, assembly_id, pdb_id, source_path, bundle_json FROM bundles")
    for case_id, assembly_id, pdb_id, source_path, bundle_json in cur:
        bundle = json.loads(bundle_json)
        bundle["_case_id"] = case_id
        bundle["pdb_id"] = bundle.get("pdb_id") or pdb_id
        bundle["source_path"] = bundle.get("source_path") or source_path
        if isinstance(bundle.get("structure_summary"), dict):
            bundle["structure_summary"].setdefault("pdb_id", pdb_id)
            bundle["structure_summary"].setdefault("source_path", source_path)
            if assembly_id and "assembly_ids" not in bundle.get("structure_summary", {}):
                bundle["structure_summary"]["assembly_ids"] = [assembly_id]
        yield bundle


# ── unified bundle lookup (used by all collect functions) ─────────────────────


def load_bundles_for_collect(
    case_dirs: Iterable[str | Path],
    *,
    prep_db_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Load all case bundles from the prep database as a dict keyed by case-id.

    Returns ``None`` when no prep database is available (caller should fall back
    to file-by-file reading).  Otherwise returns ``{case_id: [bundle_dict, ...]}``
    where each bundle_dict has ``source_path`` and ``pdb_id`` already injected
    into the structure_summary sub-dict for backwards compatibility.
    """
    if prep_db_path is None:
        return None
    conn = open_prep_db(prep_db_path)
    if conn is None:
        return None

    bundles_by_case: dict[str, list[dict[str, Any]]] = {}
    for bundle_dict in iter_bundles(conn):
        case_id = bundle_dict.get("_case_id", "")
        if case_id:
            bundles_by_case.setdefault(case_id, []).append(bundle_dict)

    return bundles_by_case if bundles_by_case else None


def load_case_bundles(
    case_dir: Path,
    *,
    prep_bundles: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Return bundles for *case_dir*, preferring the prep database when available.

    Drop-in replacement for ``load_case_output_bundles(case_dir)`` in the
    clustering collect functions.
    """
    if prep_bundles is not None:
        return prep_bundles.get(case_dir.name, [])
    return load_case_output_bundles(case_dir)


def load_cif_from_cache(conn: sqlite3.Connection, source_path: str, assembly_id: str | None) -> dict[str, Any] | None:
    """Try to retrieve a cached mmCIF atom array from the database.

    Returns a dict with keys ``atom_array`` (biotite AtomArray), ``quality``,
    and ``chain_ops``, or None if not cached.
    """
    cache_key = f"{source_path}__{assembly_id or ''}"
    cur = conn.execute(
        "SELECT atom_array_blob, quality_json, chain_ops_json FROM cif_cache WHERE cache_key = ?",
        (cache_key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    atom_blob, quality_json, chain_ops_json = row
    if atom_blob is None:
        return None
    try:
        atom_array = pickle.loads(atom_blob)
        quality = json.loads(quality_json) if quality_json else None
        chain_ops = json.loads(chain_ops_json) if chain_ops_json else None
        return {"atom_array": atom_array, "quality": quality, "chain_ops": chain_ops}
    except Exception:
        return None
