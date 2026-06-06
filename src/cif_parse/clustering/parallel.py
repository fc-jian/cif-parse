from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from tqdm import tqdm


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")

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


def iter_alignment_task_results(
    tasks: Iterable[AlignmentTask],
    runner: Callable[..., Any],
    *,
    max_workers: int = 1,
    total: int | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
    progress_unit: str = "alignment",
    **runner_kwargs: Any,
) -> Iterator[tuple[AlignmentTask, Any | None, Exception | None]]:
    """Yield alignment results while keeping only a bounded number of futures."""

    if total is not None and total <= 0:
        return

    max_workers = normalize_worker_count(max_workers)
    task_iter = iter(tasks)
    if max_workers <= 1:
        progress_iter = tqdm(
            task_iter,
            total=total,
            desc=progress_desc or "Running alignments",
            unit=progress_unit,
            disable=not show_progress or total == 1,
        )
        for task in progress_iter:
            try:
                yield task, _semaphored_runner(
                    runner,
                    task.query,
                    task.target,
                    **runner_kwargs,
                ), None
            except Exception as exc:
                yield task, None, exc
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        max_pending = max(max_workers * 4, max_workers)

        def _submit_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            future_to_task[
                executor.submit(
                    _semaphored_runner,
                    runner,
                    task.query,
                    task.target,
                    **runner_kwargs,
                )
            ] = task
            return True

        for _ in range(max_pending):
            if not _submit_next():
                break

        progress = tqdm(
            total=total,
            desc=progress_desc or "Running alignments",
            unit=progress_unit,
            disable=not show_progress or total == 1,
        )
        try:
            while future_to_task:
                done, _ = wait(future_to_task, return_when=FIRST_COMPLETED)
                for future in done:
                    task = future_to_task.pop(future)
                    try:
                        yield task, future.result(), None
                    except Exception as exc:
                        yield task, None, exc
                    progress.update(1)
                    _submit_next()
        finally:
            progress.close()


def iter_threaded_results(
    items: Iterable[ItemT],
    worker: Callable[[ItemT], ResultT],
    *,
    max_workers: int,
    total: int | None = None,
    show_progress: bool = True,
    progress_desc: str = "Processing",
    progress_unit: str = "item",
) -> Iterator[tuple[ItemT, ResultT | None, Exception | None]]:
    """Run a generic bounded thread pool and yield each item with its result."""

    max_workers = normalize_worker_count(max_workers)
    item_iter = iter(items)
    if max_workers <= 1:
        progress_iter = tqdm(
            item_iter,
            total=total,
            desc=progress_desc,
            unit=progress_unit,
            disable=not show_progress or total == 1,
        )
        for item in progress_iter:
            try:
                yield item, worker(item), None
            except Exception as exc:
                yield item, None, exc
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {}
        max_pending = max(max_workers * 4, max_workers)

        def _submit_next() -> bool:
            try:
                item = next(item_iter)
            except StopIteration:
                return False
            future_to_item[executor.submit(worker, item)] = item
            return True

        for _ in range(max_pending):
            if not _submit_next():
                break

        progress = tqdm(
            total=total,
            desc=progress_desc,
            unit=progress_unit,
            disable=not show_progress or total == 1,
        )
        try:
            while future_to_item:
                done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
                for future in done:
                    item = future_to_item.pop(future)
                    try:
                        yield item, future.result(), None
                    except Exception as exc:
                        yield item, None, exc
                    progress.update(1)
                    _submit_next()
        finally:
            progress.close()
