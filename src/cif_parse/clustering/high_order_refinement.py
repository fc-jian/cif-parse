from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from tqdm import tqdm

from cif_parse.clustering.parallel import AlignmentTask, iter_alignment_task_results
from cif_parse.clustering.protein_structures import USalignAlignmentResult


LOGGER = logging.getLogger(__name__)
MemberT = TypeVar("MemberT")
StructureT = TypeVar("StructureT")


class HighOrderRefinementError(RuntimeError):
    """Raised when high-order structure refinement cannot be completed exactly."""


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
    num_members_skipped_missing_structure: int


def _structure_weight(structure: Any) -> int:
    atom_count = int(getattr(structure, "atom_count", 0) or 0)
    if atom_count > 0:
        return atom_count
    residue_count = int(getattr(structure, "residue_count", 0) or 0)
    if residue_count > 0:
        return residue_count
    path = getattr(structure, "extracted_pdb_path", None)
    if path:
        try:
            return Path(path).stat().st_size
        except OSError:
            pass
    return 0


def _refine_signature_groups_greedy_round_barrier(
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
    min_alignment_coverage_ratio: float = 0.50,
    can_skip_alignment: Callable[[MemberT, MemberT], bool] | None = None,
    show_progress: bool = True,
    fail_fast: bool = True,
) -> GreedySignatureRefinementResult:
    """Refine signature buckets with a stage-wide USalign queue.

    Greedy clustering still proceeds independently inside each signature group,
    but each representative-vs-candidate round is collected across all active
    groups and submitted as one size-sorted batch. This keeps external USalign
    workers full when most signature groups are small.
    """

    states: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    num_members_skipped_missing_structure = 0
    structure_weights = {
        structure_id: _structure_weight(structure)
        for structure_id, structure in extracted_structures.items()
    }
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
        if len(members) > 1 and unresolved_members:
            num_members_skipped_missing_structure += len(unresolved_members)
            for member in sorted(unresolved_members, key=member_id):
                warning_rows.append(unavailable_warning(signature_cluster_id, member))
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
    num_alignment_runs = 0
    num_alignment_failures = 0

    while True:
        active_states: list[dict[str, Any]] = []
        num_alignment_tasks = 0
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
                num_alignment_tasks += 1
        if not active_states:
            break

        active_states.sort(
            key=lambda state: max(
                (
                    structure_weights.get(member_id(state["round_representative"]), 0)
                    + structure_weights.get(member_id(candidate), 0)
                    for candidate in state["round_candidates"]
                    if tuple(
                        sorted(
                            (
                                member_id(state["round_representative"]),
                                member_id(candidate),
                            )
                        )
                    )
                    not in alignment_cache
                ),
                default=0,
            ),
            reverse=True,
        )

        def _iter_round_tasks():
            for state in active_states:
                representative = state["round_representative"]
                representative_id = member_id(representative)
                candidates = sorted(
                    state["round_candidates"],
                    key=lambda candidate: structure_weights.get(member_id(candidate), 0),
                    reverse=True,
                )
                for candidate in candidates:
                    candidate_id = member_id(candidate)
                    pair_key = tuple(sorted((representative_id, candidate_id)))
                    if pair_key in alignment_cache:
                        continue
                    yield AlignmentTask(
                        key=pair_key,
                        query=extracted_structures[representative_id],
                        target=extracted_structures[candidate_id],
                        context={
                            "signature_cluster_id": state["signature_cluster_id"],
                            "representative": representative,
                            "candidate": candidate,
                        },
                    )

        if num_alignment_tasks:
            LOGGER.info(
                "Submitting %d high-order USalign tasks across %d signature groups (largest structures first)",
                num_alignment_tasks,
                len(active_states),
            )
        failure_by_key: dict[tuple[str, str], Exception] = {}
        success_keys: set[tuple[str, str]] = set()
        for task, result, error in iter_alignment_task_results(
            _iter_round_tasks(),
            runner,
            max_workers=alignment_jobs,
            total=num_alignment_tasks,
            show_progress=show_progress,
            progress_desc="Running high-order USalign",
            usalign_executable=usalign_executable,
            tm_score_threshold=tm_score_threshold,
            min_alignment_coverage_ratio=min_alignment_coverage_ratio,
        ):
            if error is not None:
                if fail_fast:
                    raise HighOrderRefinementError(
                        "High-order USalign failed for "
                        f"{task.key[0]} vs {task.key[1]} in "
                        f"{task.context['signature_cluster_id']}: {error}"
                    ) from error
                failure_by_key[task.key] = error
                continue
            alignment_cache[task.key] = result
            success_keys.add(task.key)

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
                if result.meets_tm_threshold:
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
        local_clusters: list[tuple[list[MemberT], MemberT]] = state["local_clusters"]
        if not extracted_members and len(members) == 1:
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

    if num_members_skipped_missing_structure:
        LOGGER.warning(
            "Skipped %d high-order observations with unavailable extracted structures",
            num_members_skipped_missing_structure,
        )

    return GreedySignatureRefinementResult(
        alignment_cache=alignment_cache,
        alignment_rows=alignment_rows,
        warning_rows=warning_rows,
        cluster_members=cluster_members,
        num_alignment_runs=num_alignment_runs,
        num_alignment_failures=num_alignment_failures,
        num_signature_clusters_split=num_signature_clusters_split,
        num_members_skipped_missing_structure=num_members_skipped_missing_structure,
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
    min_alignment_coverage_ratio: float = 0.50,
    can_skip_alignment: Callable[[MemberT, MemberT], bool] | None = None,
    show_progress: bool = True,
    fail_fast: bool = True,
) -> GreedySignatureRefinementResult:
    """Compatibility wrapper for the bounded greedy high-order scheduler."""

    return refine_signature_groups_greedy(
        signature_groups,
        extracted_structures,
        member_id=member_id,
        structural_sort_key=structural_sort_key,
        alignment_row=alignment_row,
        alignment_failure_warning=alignment_failure_warning,
        unavailable_warning=unavailable_warning,
        runner=runner,
        alignment_jobs=alignment_jobs,
        usalign_executable=usalign_executable,
        tm_score_threshold=tm_score_threshold,
        min_alignment_coverage_ratio=min_alignment_coverage_ratio,
        can_skip_alignment=can_skip_alignment,
        show_progress=show_progress,
        fail_fast=fail_fast,
    )


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
    min_alignment_coverage_ratio: float = 0.50,
    can_skip_alignment: Callable[[MemberT, MemberT], bool] | None = None,
    show_progress: bool = True,
    fail_fast: bool = True,
) -> GreedySignatureRefinementResult:
    """Refine independent signature groups with one asynchronous worker pool."""

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    import heapq
    import itertools

    from cif_parse.clustering.parallel import _semaphored_runner, normalize_worker_count

    states: dict[str, dict[str, Any]] = {}
    warning_rows: list[dict[str, Any]] = []
    num_members_skipped_missing_structure = 0
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
        if len(members) > 1:
            num_members_skipped_missing_structure += len(unresolved_members)
            warning_rows.extend(
                unavailable_warning(signature_cluster_id, member)
                for member in sorted(unresolved_members, key=member_id)
            )
        states[signature_cluster_id] = {
            "signature_cluster_id": signature_cluster_id,
            "members": members,
            "extracted_members": extracted_members,
            "pending": sorted(extracted_members, key=structural_sort_key),
            "local_clusters": [],
        }

    alignment_jobs = normalize_worker_count(alignment_jobs)
    alignment_cache: dict[tuple[str, str], Any] = {}
    alignment_rows: list[dict[str, Any]] = []
    num_alignment_runs = 0
    num_alignment_failures = 0
    active_count = 0
    ready_states: list[tuple[int, str, int]] = []
    ready_counter = itertools.count()

    def _queue_state(state: dict[str, Any]) -> None:
        cursor = state["round_cursor"]
        representative_id = member_id(state["round_representative"])
        candidate = state["round_candidates"][cursor]
        pair_weight = (
            _structure_weight(extracted_structures[representative_id])
            + _structure_weight(extracted_structures[member_id(candidate)])
        )
        heapq.heappush(
            ready_states,
            (
                -pair_weight,
                state["signature_cluster_id"],
                next(ready_counter),
            ),
        )

    def _start_round(state: dict[str, Any]) -> None:
        nonlocal active_count
        while state["pending"]:
            representative = state["pending"][0]
            candidates = state["pending"][1:]
            if not candidates:
                state["local_clusters"].append(([representative], representative))
                state["pending"] = []
                return
            state["round_representative"] = representative
            state["round_candidates"] = candidates
            state["round_cursor"] = 0
            state["round_done"] = 0
            state["round_results"] = {}
            state["round_failures"] = {}
            _queue_state(state)
            active_count += 1
            return

    def _finish_round(state: dict[str, Any]) -> None:
        nonlocal active_count, num_alignment_failures
        representative = state["round_representative"]
        assigned = [representative]
        remaining: list[MemberT] = []
        for candidate in state["round_candidates"]:
            candidate_id = member_id(candidate)
            failure = state["round_failures"].get(candidate_id)
            if failure is not None:
                num_alignment_failures += 1
                warning_rows.append(
                    alignment_failure_warning(
                        state["signature_cluster_id"],
                        representative,
                        candidate,
                        failure,
                    )
                )
                remaining.append(candidate)
                continue
            result = state["round_results"][candidate_id]
            if result.meets_tm_threshold:
                assigned.append(candidate)
            else:
                remaining.append(candidate)
        state["local_clusters"].append((assigned, representative))
        state["pending"] = remaining
        active_count -= 1
        _start_round(state)

    for state in states.values():
        _start_round(state)

    future_to_item: dict[Any, tuple[dict[str, Any], MemberT, tuple[str, str]]] = {}
    max_pending = alignment_jobs * 4
    progress = tqdm(
        desc="Running high-order USalign",
        unit="alignment",
        disable=not show_progress,
    )
    with ThreadPoolExecutor(max_workers=alignment_jobs) as executor:
        while ready_states or future_to_item or active_count:
            capacity = max_pending - len(future_to_item)
            while ready_states and capacity > 0:
                _, signature_cluster_id, _ = heapq.heappop(ready_states)
                state = states[signature_cluster_id]
                cursor = state["round_cursor"]
                candidates = state["round_candidates"]
                if cursor >= len(candidates):
                    continue
                candidate = candidates[cursor]
                state["round_cursor"] = cursor + 1
                if state["round_cursor"] < len(candidates):
                    _queue_state(state)

                representative = state["round_representative"]
                representative_id = member_id(representative)
                candidate_id = member_id(candidate)
                pair_key = tuple(sorted((representative_id, candidate_id)))
                cached = alignment_cache.get(pair_key)
                if cached is not None:
                    state["round_results"][candidate_id] = cached
                    state["round_done"] += 1
                    progress.update(1)
                    if state["round_done"] == len(candidates):
                        _finish_round(state)
                    continue
                if can_skip_alignment is not None and can_skip_alignment(
                    representative,
                    candidate,
                ):
                    query_structure = extracted_structures[representative_id]
                    target_structure = extracted_structures[candidate_id]
                    result = _make_identical_alignment(
                        representative_id,
                        candidate_id,
                        query_residue_count=getattr(
                            query_structure,
                            "residue_count",
                            1,
                        )
                        or 1,
                        target_residue_count=getattr(
                            target_structure,
                            "residue_count",
                            1,
                        )
                        or 1,
                    )
                    alignment_cache[pair_key] = result
                    state["round_results"][candidate_id] = result
                    state["round_done"] += 1
                    progress.update(1)
                    if state["round_done"] == len(candidates):
                        _finish_round(state)
                    continue

                future = executor.submit(
                    _semaphored_runner,
                    runner,
                    extracted_structures[representative_id],
                    extracted_structures[candidate_id],
                    usalign_executable=usalign_executable,
                    tm_score_threshold=tm_score_threshold,
                    min_alignment_coverage_ratio=min_alignment_coverage_ratio,
                )
                future_to_item[future] = (state, candidate, pair_key)
                capacity -= 1

            if future_to_item and (
                len(future_to_item) >= max_pending or not ready_states
            ):
                done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
                for future in done:
                    state, candidate, pair_key = future_to_item.pop(future)
                    candidate_id = member_id(candidate)
                    representative = state["round_representative"]
                    try:
                        result = future.result()
                    except Exception as exc:
                        if fail_fast:
                            progress.close()
                            raise HighOrderRefinementError(
                                "High-order USalign failed for "
                                f"{pair_key[0]} vs {pair_key[1]} in "
                                f"{state['signature_cluster_id']}: {exc}"
                            ) from exc
                        state["round_failures"][candidate_id] = exc
                    else:
                        alignment_cache[pair_key] = result
                        state["round_results"][candidate_id] = result
                        alignment_rows.append(
                            alignment_row(state["signature_cluster_id"], result)
                        )
                        num_alignment_runs += 1
                    state["round_done"] += 1
                    progress.update(1)
                    if state["round_done"] == len(state["round_candidates"]):
                        _finish_round(state)
            elif not future_to_item and not ready_states and active_count:
                progress.close()
                raise RuntimeError("High-order refinement scheduler stalled")
    progress.close()

    cluster_members: list[tuple[str, str, list[Any], Any]] = []
    num_signature_clusters_split = 0
    for signature_cluster_id, _ in signature_groups:
        state = states[signature_cluster_id]
        members = state["members"]
        extracted_members = state["extracted_members"]
        local_clusters = state["local_clusters"]
        if not extracted_members and len(members) == 1:
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
        num_members_skipped_missing_structure=num_members_skipped_missing_structure,
    )
