"""Reproducibility provenance: git hash, package versions, and resolved config."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "catboost",
    "rdkit",
    "bhpu",  # pubench ships in the bhpu distribution (shared src wheel)
)


def git_hash() -> str:
    """Return the current git commit hash, or ``"unknown"`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def package_versions() -> dict[str, str]:
    """Return the installed versions of the packages the pipeline depends on."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def provenance_record(config: dict[str, object]) -> dict[str, object]:
    """Assemble the full provenance record for a run.

    Parameters
    ----------
    config:
        The resolved experiment config (see :meth:`ExperimentConfig.to_dict`).

    Returns:
    -------
    dict[str, object]
        A dict with ``git_hash``, ``packages`` and the resolved ``config``.
    """
    return {
        "git_hash": git_hash(),
        "packages": package_versions(),
        "config": config,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as pretty JSON to ``path`` (creating parent dirs)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
