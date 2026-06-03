from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from tqdm import tqdm

# Global semaphore to cap total USalign subprocesses across ALL stages
# (monomer, dimer, multimer, antibody, TCR).
_global_usalign_semaphore: threading.BoundedSemaphore | None = None
_global_usalign_limit: int | None = None


def set_global_usalign_limit(max_concurrent: int) -> None:
    """Cap the total number of concurrent USalign processes across all stages."""
    global _global_usalign_limit, _global_usalign_semaphore
    max_concurrent = normalize_worker_count(max_concurrent)
    if _global_usalign_limit != max_concurrent:
        _global_usalign_limit = max_concurrent
        _global_usalign_semaphore = threading.BoundedSemaphore(max_concurrent)


@dataclass(slots=True)
class AlignmentTask:
    """One external alignment command to be scheduled by a bounded worker pool."""

    key: tuple[str, str]
    query: Any
    target: Any
    context: dict[str, Any]


def normalize_worker_count(value: int | None) -> int:
    if value is None:
        return 1
    return max(1, int(value))


def _semaphored_runner(runner, query, target, **kwargs):
    if _global_usalign_semaphore is not None:
        _global_usalign_semaphore.acquire()
    try:
        return runner(query, target, **kwargs)
    finally:
        if _global_usalign_semaphore is not None:
            _global_usalign_semaphore.release()


def run_alignment_tasks(
    tasks: Iterable[AlignmentTask],
    runner: Callable[..., Any],
    *,
    max_workers: int = 1,
    show_progress: bool = True,
    progress_desc: str | None = None,
    progress_unit: str = "alignment",
    **runner_kwargs: Any,
) -> tuple[list[tuple[AlignmentTask, Any]], list[tuple[AlignmentTask, Exception]]]:
    """Run alignment tasks concurrently without materializing all futures at once."""

    task_list = list(tasks)
    if not task_list:
        return [], []

    max_workers = min(normalize_worker_count(max_workers), len(task_list))
    if max_workers <= 1:
        successes: list[tuple[AlignmentTask, Any]] = []
        failures: list[tuple[AlignmentTask, Exception]] = []
        task_iter = tqdm(
            task_list,
            desc=progress_desc or "Running alignments",
            unit=progress_unit,
            disable=not show_progress or len(task_list) < 2,
        )
        for task in task_iter:
            try:
                successes.append((task, _semaphored_runner(runner, task.query, task.target, **runner_kwargs)))
            except Exception as exc:
                failures.append((task, exc))
        return successes, failures

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        task_iter = iter(task_list)
        future_to_task = {}
        max_pending = min(len(task_list), max(max_workers * 4, max_workers))

        def _submit_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            future_to_task[
                executor.submit(_semaphored_runner, runner, task.query, task.target, **runner_kwargs)
            ] = task
            return True

        for _ in range(max_pending):
            if not _submit_next():
                break

        progress = tqdm(
            total=len(task_list),
            desc=progress_desc or "Running alignments",
            unit=progress_unit,
            disable=not show_progress or len(task_list) < 2,
        )
        try:
            while future_to_task:
                done, _ = wait(future_to_task, return_when=FIRST_COMPLETED)
                for future in done:
                    task = future_to_task.pop(future)
                    try:
                        successes.append((task, future.result()))
                    except Exception as exc:
                        failures.append((task, exc))
                    progress.update(1)
                    _submit_next()
        finally:
            progress.close()
    return successes, failures
