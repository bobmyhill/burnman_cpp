# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Snapshot audits tolerate numerical roundoff but reject changed references."""

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("github_output", [False, True])
def test_reference_pin_cli_reads_policy_changes_from_any_directory(
    tmp_path, github_output
):
    root = Path(__file__).resolve().parents[2]
    tools = tmp_path / "tools"
    reference = tmp_path / "tests/reference"
    tools.mkdir()
    reference.mkdir(parents=True)
    script = tools / "burnman_reference.py"
    shutil.copyfile(root / "tools/burnman_reference.py", script)
    policy = json.loads((root / "tests/reference/burnman.json").read_text())
    for commit in ("a" * 40, "b" * 40):
        policy["commit"] = commit
        (reference / "burnman.json").write_text(json.dumps(policy))
        command = [sys.executable, str(script)]
        if github_output:
            command.append("--github-output")
        output = subprocess.check_output(command, cwd=tools, text=True).strip()
        if github_output:
            assert dict(line.split("=", 1) for line in output.splitlines()) == {
                "repository": "geodynamics/burnman",
                "commit": commit,
            }
        else:
            assert output == commit


@pytest.fixture
def covariance_audit(monkeypatch, tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "tools"))
    audit = importlib.import_module("export_covariance_data")
    names = np.array(["a", "b"], dtype="S")
    matrix = np.array([[1.0, 0.5], [0.5, 2.0]], dtype="<f8")
    path = tmp_path / "covariance.h5"
    with h5py.File(path, "w") as data:
        data.attrs.update(audit.attributes("example"))
        data.create_dataset("endmember_names", data=names)
        data.create_dataset(
            "covariance_matrix",
            data=matrix,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
    return audit, path, names, matrix


def test_covariance_audit_accepts_original_export_commit(covariance_audit, monkeypatch):
    audit, path, names, matrix = covariance_audit
    monkeypatch.setitem(audit.REFERENCE, "commit", "a" * 40)
    assert audit.check_file(path, "example", names, matrix)


@pytest.mark.parametrize("change", ["value", "name", "units", "commit"])
def test_covariance_audit_rejects_changed_data_and_invalid_metadata(
    covariance_audit, change
):
    import h5py

    audit, path, names, matrix = covariance_audit
    with h5py.File(path, "r+") as data:
        if change == "value":
            data["covariance_matrix"][0, 0] += 1.0
        elif change == "name":
            data["endmember_names"][0] = "c"
        elif change == "units":
            data.attrs["units"] = "wrong units"
        else:
            data.attrs["burnman_commit"] = "not-a-commit"
    assert not audit.check_file(path, "example", names, matrix)


@pytest.fixture(scope="module")
def ps1994_audit():
    if shutil.which("clang-format") is None:
        pytest.skip("The optional reference audit requires clang-format.")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[2] / "tools"))
        audit = importlib.import_module("generate_ps1994_reference")
    return audit, audit.OUTPUT.read_text()


def test_ps1994_audit_accepts_roundoff_and_reformatting(ps1994_audit):
    audit, stored = ps1994_audit
    generated = stored.replace("0.04140096296511458", "0.04140096296511459")
    generated = generated.replace(", ", ",\n")
    assert generated != stored
    audit.check_snapshot(stored, generated)


@pytest.mark.parametrize(
    "original,changed,message",
    [
        ("0.04140096296511458", "0.04150096296511458", "volume"),
        ("-338143.8857849859", "-338143.8856849859", "Gibbs energy"),
        ("35.58984708602805, 2}", "35.58984708602805, 3}", "stable roots"),
        ("tests/reference/burnman.json", "invalid_policy.json", "provenance"),
    ],
)
def test_ps1994_audit_rejects_changed_physics_or_provenance(
    ps1994_audit, original, changed, message
):
    audit, stored = ps1994_audit
    generated = stored.replace(original, changed, 1)
    assert generated != stored
    with pytest.raises((AssertionError, ValueError), match=message):
        audit.check_snapshot(stored, generated)
