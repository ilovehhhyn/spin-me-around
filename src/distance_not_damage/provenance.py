"""Capture enough run context to reproduce and audit an experiment."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

RUN_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunProvenance:
    """Environment and source revision associated with one experiment run."""

    schema_version: int
    created_at_utc: str
    python_version: str
    platform: str
    package_version: str
    torch_version: str
    numpy_version: str
    git_commit: str | None
    git_dirty: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_run_provenance(repository_path: Path) -> RunProvenance:
    """Capture runtime versions and Git state without mutating the repository."""
    git_commit, git_dirty = _read_git_state(repository_path)
    return RunProvenance(
        schema_version=RUN_ARTIFACT_SCHEMA_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_version=_installed_package_version(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def _installed_package_version() -> str:
    try:
        return version("distance-not-damage")
    except PackageNotFoundError:
        return "uninstalled"


def _read_git_state(repository_path: Path) -> tuple[str | None, bool | None]:
    git = shutil.which("git")
    if git is None:
        return None, None

    revision = subprocess.run(
        [git, "rev-parse", "HEAD"],
        cwd=repository_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        return None, None

    status = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_path,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return revision.stdout.strip(), dirty
