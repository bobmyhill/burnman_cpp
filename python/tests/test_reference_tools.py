"""Snapshot audits tolerate numerical roundoff but reject changed references."""

import importlib
from pathlib import Path
import shutil

import pytest

pytest.importorskip("scipy")


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
        ("39b582cd23954fabd3bdbbee6526a522b1d97bc9", "invalid_commit", "provenance"),
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
