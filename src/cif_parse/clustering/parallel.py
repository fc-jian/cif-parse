from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable


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


def run_alignment_tasks(
    tasks: Iterable[AlignmentTask],
    runner: Callable[..., Any],
    *,
    max_workers: int = 1,
    **runner_kwargs: Any,
) -> tuple[list[tuple[AlignmentTask, Any]], list[tuple[AlignmentTask, Exception]]]:
    """Run alignment tasks concurrently while preserving caller-managed result ordering."""

    task_list = list(tasks)
    if not task_list:
        return [], []

    max_workers = min(normalize_worker_count(max_workers), len(task_list))
    if max_workers <= 1:
        successes: list[tuple[AlignmentTask, Any]] = []
        failures: list[tuple[AlignmentTask, Exception]] = []
        for task in task_list:
            try:
                successes.append((task, runner(task.query, task.target, **runner_kwargs)))
            except Exception as exc:
                failures.append((task, exc))
        return successes, failures

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(runner, task.query, task.target, **runner_kwargs): task
            for task in task_list
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                successes.append((task, future.result()))
            except Exception as exc:
                failures.append((task, exc))
    return successes, failures
