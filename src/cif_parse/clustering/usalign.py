"""Utilities shared by USalign call sites."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def run_usalign_command(command: Sequence[str]) -> str:
    """Run USalign and preserve its diagnostic output on failure."""

    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip()
            for part in (exc.stderr, exc.stdout)
            if part and part.strip()
        )
        if len(output) > 4000:
            output = output[-4000:]
        detail = output or "no diagnostic output"
        raise RuntimeError(
            f"USalign exited with status {exc.returncode}: {detail}"
        ) from exc
    return completed.stdout
