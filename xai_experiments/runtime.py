"""Runtime metadata helpers for reproducibility logs."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess
import sys


def runtime_metadata(cwd: Path) -> dict[str, object]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(cwd),
        "git": git_metadata(cwd),
    }


def git_metadata(cwd: Path) -> dict[str, object]:
    return {
        "commit": _git(["rev-parse", "HEAD"], cwd),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "status_short": _git(["status", "--short"], cwd),
    }


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
