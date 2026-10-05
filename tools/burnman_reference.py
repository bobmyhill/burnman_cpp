"""Shared provenance checks for the optional Python BurnMan reference."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/reference/burnman.json").read_text())


def verify_reference(package):
    """Require the imported package to belong to the clean, pinned checkout."""
    source = Path(package.__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "The reference BurnMan must be imported from a Git checkout."
        ) from exc
    if commit != REFERENCE["commit"]:
        raise RuntimeError(
            f"Reference BurnMan is at {commit}; expected {REFERENCE['commit']}. "
            "Put the pinned checkout first on PYTHONPATH."
        )
    if changed:
        raise RuntimeError("The reference BurnMan checkout contains local changes.")
    if (source / "burnman/__init__.py").resolve() != Path(package.__file__).resolve():
        raise RuntimeError(
            "The imported reference does not belong to the pinned checkout."
        )
    return source
