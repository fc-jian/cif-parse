from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from tqdm import tqdm

from cif_parse.clustering.parallel import AlignmentTask, run_alignment_tasks
from cif_parse.clustering.protein_structures import USalignAlignmentResult


LOGGER = logging.getLogger(__name__)
MemberT = TypeVar("MemberT")
StructureT = TypeVar("StructureT")


def _make_identical_alignment(
    query_id: str,
    target_id: str,
    query_residue_count: int = 1,
    target_residue_count: int = 1,
) -> USalignAlignmentResult:
    """Return a synthetic perfect-match result for structurally identical members."""
    aligned_len = min(query_residue_count, target_residue_count)
    shorter_len = min(query_residue_count, target_residue_count)
    return USalignAlignmentResult(
        query_monomer_id=query_id,
        target_monomer_id=target_id,
        aligned_length=aligned_len,
        rmsd=0.0,
        tm_score_query=1.0,
        tm_score_target=1.0,
        min_tm_score=1.0,
        max_tm_score=1.0,
        shorter_length_coverage=1.0,
        resolved_length_coverage=1.0,
        meets_tm_threshold=True,
        meets_coverage_threshold=True,
    )


@dataclass(slots=True)
class GreedySignatureRefinementResult:
    alignment_cache: dict[tuple[str, str], Any]
    alignment_rows: list[dict[str, Any]]
    warning_rows: list[dict[str, Any]]
    cluster_members: list[tuple[str, str, list[Any], Any]]
    num_alignment_runs: int
    num_alignment_failures: int
    num_signature_clusters_split: int


def _structure_file_size(structure: Any) -> int:
    path = getattr(structure, "extracted_pdb_path", None)
    if path:
        try:
            return Path(path).stat().st_size
        except OSError:
            pass
    residue_count = int(getattr(structure, "residue_count", 0) or 0)
    atom_count = int(getattr(structure, "atom_count", 0) or 0)
    return residue_count * 1000 + atom_count


def _task_size(task: AlignmentTask) -> int:
    return _structure_file_size(task.query) + _structure_file_size(task.target)


def refine_signature_groups_greedy(
    signature_groups: list[tuple[str, list[MemberT]]],
    extracted_structures: dict[str, StructureT],
    *,
    member_id: Callable[[MemberT], str],
    structural_sort_key: Callable[[MemberT], Any],
    alignment_row: Callable[[str, Any], dict[str, Any]],
    alignment_failure_warning: Callable[[str, MemberT, MemberT, Exception], dict[str, Any]],
    unavailable_warning: Callable[[str, MemberT], dict[str, Any]],
    runner: Callable[..., Any],
    alignment_jobs: int,
    usalign_executable: str,
    tm_score_threshold: float,
    can_skip_alignment: Callable[[MemberT, MemberT], bool] | None = None,
    show_progress: bool = True,
) -> GreedySignatureRefinementResult:
    """Refine signature buckets with a stage-wide USalign queue.

    Greedy clustering still proceeds independently inside each signature group,
    but each representative-vs-candidate round is collected across all active
    groups and submitted as one size-sorted batch. This keeps external USalign
    workers full when most signature groups are small.
    """

    states: list[dict[str, Any]] = []
    for signature_cluster_id, members in tqdm(
        signature_groups,
        desc="Preparing high-order refinement groups",
        unit="sig-group",
        disable=not show_progress or len(signature_groups) < 2,
    ):
        extracted_members = [
            member for member in members if member_id(member) in extracted_structures
        ]
        unresolved_members = [
            member for member in members if member_id(member) not in extracted_structures
        ]
        states.append(
            {
                "signature_cluster_id": signature_cluster_id,
                "members": members,
                "extracted_members": extracted_members,
                "unresolved_members": unresolved_members,
                "pending": sorted(extracted_members, key=structural_sort_key),
                "local_clusters": [],
            }
        )

    alignment_cache: dict[tuple[str, str], Any] = {}
    alignment_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    num_alignment_runs = 0
    num_alignment_failures = 0

    while True:
        active_states: list[dict[str, Any]] = []
        alignment_tasks: list[AlignmentTask] = []
        for state in states:
            pending: list[MemberT] = state["pending"]
            if not pending:
                continue
            representative = pending[0]
            candidates = pending[1:]
            state["round_representative"] = representative
            state["round_candidates"] = candidates
            active_states.append(state)
            for candidate in candidates:
                pair_key = tuple(sorted((member_id(representative), member_id(candidate))))
                if pair_key in alignment_cache:
                    continue
                if can_skip_alignment is not None and can_skip_alignment(representative, candidate):
                    qs = extracted_structures.get(member_id(representative))
                    ts = extracted_structures.get(member_id(candidate))
                    alignment_cache[pair_key] = _make_identical_alignment(
                        member_id(representative), member_id(candidate),
                        query_residue_count=getattr(qs, "residue_count", 1) or 1,
                        target_residue_count=getattr(ts, "residue_count", 1) or 1,
                    )
                    continue
                alignment_tasks.append(
                    AlignmentTask(
                        key=pair_key,
                        query=extracted_structures[member_id(representative)],
                        target=extracted_structures[member_id(candidate)],
                        context={
                            "signature_cluster_id": state["signature_cluster_id"],
                            "representative": representative,
                            "candidate": candidate,
                        },
                    )
                )
        if not active_states:
            break

        alignment_tasks.sort(key=_task_size, reverse=True)
        if alignment_tasks:
            LOGGER.info(
                "Submitting %d high-order USalign tasks across %d signature groups (largest PDB pairs first)",
                len(alignment_tasks),
                len(active_states),
            )
        successes, failures = run_alignment_tasks(
            alignment_tasks,
            runner,
            max_workers=alignment_jobs,
            show_progress=show_progress,
            progress_desc="Running high-order USalign",
            usalign_executable=usalign_executable,
            tm_score_threshold=tm_score_threshold,
        )
        failure_by_key = {task.key: exc for task, exc in failures}
        for task, result in successes:
            alignment_cache[task.key] = result
        success_keys = {task.key for task, _ in successes}

        for state in active_states:
            signature_cluster_id = state["signature_cluster_id"]
            representative: MemberT = state["round_representative"]
            candidates: list[MemberT] = state["round_candidates"]
            assigned = [representative]
            remaining: list[MemberT] = []
            for candidate in candidates:
                pair_key = tuple(sorted((member_id(representative), member_id(candidate))))
                if pair_key in failure_by_key:
                    exc = failure_by_key[pair_key]
                    num_alignment_failures += 1
                    warning_rows.append(
                        alignment_failure_warning(
                            signature_cluster_id,
                            representative,
                            candidate,
                            exc,
                        )
                    )
                    remaining.append(candidate)
                    continue
                result = alignment_cache[pair_key]
                if pair_key in success_keys:
                    alignment_rows.append(alignment_row(signature_cluster_id, result))
                    num_alignment_runs += 1
                if result.max_tm_score >= tm_score_threshold:
                    assigned.append(candidate)
                else:
                    remaining.append(candidate)
            state["local_clusters"].append((assigned, representative))
            state["pending"] = remaining

    cluster_members: list[tuple[str, str, list[Any], Any]] = []
    num_signature_clusters_split = 0
    for state in tqdm(
        states,
        desc="Finalizing high-order refinement",
        unit="sig-group",
        disable=not show_progress or len(states) < 2,
    ):
        signature_cluster_id = state["signature_cluster_id"]
        members = state["members"]
        extracted_members = state["extracted_members"]
        unresolved_members = state["unresolved_members"]
        local_clusters: list[tuple[list[MemberT], MemberT]] = state["local_clusters"]
        if unresolved_members and extracted_members:
            for unresolved in sorted(unresolved_members, key=structural_sort_key):
                local_clusters.append(([unresolved], unresolved))
                warning_rows.append(unavailable_warning(signature_cluster_id, unresolved))
        elif not extracted_members and members:
            representative = min(members, key=structural_sort_key)
            local_clusters.append((list(members), representative))

        if len(local_clusters) > 1:
            num_signature_clusters_split += 1
        for members_in_cluster, representative in local_clusters:
            cluster_members.append(
                (
                    signature_cluster_id,
                    member_id(representative),
                    sorted(members_in_cluster, key=member_id),
                    representative,
                )
            )

    return GreedySignatureRefinementResult(
        alignment_cache=alignment_cache,
        alignment_rows=alignment_rows,
        warning_rows=warning_rows,
        cluster_members=cluster_members,
        num_alignment_runs=num_alignment_runs,
        num_alignment_failures=num_alignment_failures,
        num_signature_clusters_split=num_signature_clusters_split,
    )


def refine_signature_groups_three_phase(
    signature_groups: list[tuple[str, list[MemberT]]],
    extracted_structures: dict[str, StructureT],
    *,
    member_id: Callable[[MemberT], str],
    structural_sort_key: Callable[[MemberT], Any],
    alignment_row: Callable[[str, Any], dict[str, Any]],
    alignment_failure_warning: Callable[[str, MemberT, MemberT, Exception], dict[str, Any]],
    unavailable_warning: Callable[[str, MemberT], dict[str, Any]],
    runner: Callable[..., Any],
    alignment_jobs: int,
    usalign_executable: str,
    tm_score_threshold: float,
    can_skip_alignment: Callable[[MemberT, MemberT], bool] | None = None,
    show_progress: bool = True,
) -> GreedySignatureRefinementResult:
    """Three-phase refinement: enumerate all pairwise tasks upfront, execute in
    one globally-sorted batch, then greedily reconstruct clusters.

    This trades more USalign calls for full parallelism — all groups share
    a single ``run_alignment_tasks`` invocation instead of lock-step rounds.
    """

    # ---- Phase 1: enumerate ALL pairwise tasks across all groups --------
    alignment_tasks: list[AlignmentTask] = []
    group_states: list[dict[str, Any]] = []
    identical_pairs: dict[tuple[str, str], USalignAlignmentResult] = {}

    for signature_cluster_id, members in tqdm(
        signature_groups,
        desc="Preparing high-order USalign tasks",
        unit="sig-group",
        disable=not show_progress or len(signature_groups) < 2,
    ):
        extracted = [m for m in members if member_id(m) in extracted_structures]
        unresolved = [m for m in members if member_id(m) not in extracted_structures]
        pending = sorted(extracted, key=structural_sort_key)
        group_states.append({
            "signature_cluster_id": signature_cluster_id,
            "members": members,
            "extracted_members": extracted,
            "unresolved_members": unresolved,
            "pending_sorted": pending,
        })

        # Enumerate all pairwise comparisons (C(k,2)) for greedy later
        for i, rep_candidate in enumerate(pending):
            for j in range(i + 1, len(pending)):
                candidate = pending[j]
                pair_key = tuple(sorted((member_id(rep_candidate), member_id(candidate))))
                if can_skip_alignment is not None and can_skip_alignment(rep_candidate, candidate):
                    qs = extracted_structures.get(member_id(rep_candidate))
                    ts = extracted_structures.get(member_id(candidate))
                    identical_pairs[pair_key] = _make_identical_alignment(
                        member_id(rep_candidate), member_id(candidate),
                        query_residue_count=getattr(qs, "residue_count", 1) or 1,
                        target_residue_count=getattr(ts, "residue_count", 1) or 1,
                    )
                    continue
                alignment_tasks.append(AlignmentTask(
                    key=pair_key,
                    query=extracted_structures[member_id(rep_candidate)],
                    target=extracted_structures[member_id(candidate)],
                    context={
                        "signature_cluster_id": signature_cluster_id,
                        "representative": rep_candidate,
                        "candidate": candidate,
                    },
                ))

    # ---- Phase 2: execute all tasks in one globally-sorted batch ---------
    alignment_tasks.sort(key=_task_size, reverse=True)
    if alignment_tasks:
        LOGGER.info(
            "Three-phase: submitting %d high-order USalign tasks across %d groups",
            len(alignment_tasks), len(signature_groups),
        )
    successes, failures = run_alignment_tasks(
        alignment_tasks, runner,
        max_workers=alignment_jobs,
        show_progress=show_progress,
        progress_desc="Running high-order USalign",
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
    )

    alignment_cache: dict[tuple[str, str], Any] = dict(identical_pairs)
    failure_by_key: dict[tuple[str, str], Exception] = {}
    for task, exc in failures:
        failure_by_key[task.key] = exc
    for task, result in successes:
        alignment_cache[task.key] = result
    success_keys = {t.key for t, _ in successes}

    # ---- Phase 3: greedy reconstruction per group ------------------------
    alignment_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    cluster_members: list[tuple[str, str, list[Any], Any]] = []
    num_alignment_runs = 0
    num_alignment_failures = 0
    num_signature_clusters_split = 0

    for state in tqdm(
        group_states,
        desc="Reconstructing high-order clusters",
        unit="sig-group",
        disable=not show_progress or len(group_states) < 2,
    ):
        sig_id = state["signature_cluster_id"]
        members = state["members"]
        extracted = state["extracted_members"]
        unresolved = state["unresolved_members"]
        pending = sorted(extracted, key=structural_sort_key)
        local_clusters: list[tuple[list[MemberT], MemberT]] = []

        while pending:
            rep = pending[0]
            assigned = [rep]
            remaining = []
            for candidate in pending[1:]:
                pair_key = tuple(sorted((member_id(rep), member_id(candidate))))
                if pair_key in failure_by_key:
                    num_alignment_failures += 1
                    warning_rows.append(alignment_failure_warning(sig_id, rep, candidate, failure_by_key[pair_key]))
                    remaining.append(candidate)
                    continue
                result = alignment_cache.get(pair_key)
                if result is None:
                    remaining.append(candidate)
                    continue
                if pair_key in success_keys:
                    alignment_rows.append(alignment_row(sig_id, result))
                    num_alignment_runs += 1
                if result.max_tm_score >= tm_score_threshold:
                    assigned.append(candidate)
                else:
                    remaining.append(candidate)
            local_clusters.append((assigned, rep))
            pending = remaining

        if unresolved and extracted:
            for u in sorted(unresolved, key=structural_sort_key):
                local_clusters.append(([u], u))
                warning_rows.append(unavailable_warning(sig_id, u))
        elif not extracted and members:
            rep = min(members, key=structural_sort_key)
            local_clusters.append((list(members), rep))

        if len(local_clusters) > 1:
            num_signature_clusters_split += 1
        for cl_members, rep in local_clusters:
            cluster_members.append((sig_id, member_id(rep), sorted(cl_members, key=member_id), rep))

    return GreedySignatureRefinementResult(
        alignment_cache=alignment_cache,
        alignment_rows=alignment_rows,
        warning_rows=warning_rows,
        cluster_members=cluster_members,
        num_alignment_runs=num_alignment_runs,
        num_alignment_failures=num_alignment_failures,
        num_signature_clusters_split=num_signature_clusters_split,
    )
