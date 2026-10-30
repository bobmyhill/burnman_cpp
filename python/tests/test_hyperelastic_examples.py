# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Execute both complete upstream hyperelastic walkthroughs and render plots."""

import builtins
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

from test_anisotropic_examples import ROOT, run_example

EXAMPLES = ["example_hyperelastic_mineral.py", "example_hyperelastic_stress_rates.py"]


@pytest.fixture(scope="module", params=EXAMPLES)
def example(request, tmp_path_factory):
    cwd = tmp_path_factory.mktemp(request.param[:-3])
    original_import = builtins.__import__

    def native_import(name, *args, **kwargs):
        if name == "burnman" or name.startswith("burnman."):
            raise ImportError("The examples must run without Python BurnMan")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "examples"))
        patch.delitem(sys.modules, "example_hyperelastic_mineral", raising=False)
        patch.setattr(builtins, "__import__", native_import)
        actual = run_example(ROOT / "examples" / request.param, cwd)
    return request.param, actual


def test_complete_native_examples(example):
    name, (namespace, figures, contours, curves, output) = example
    assert not contours
    assert all(np.isfinite(values).all() for values in curves)
    if name == EXAMPLES[0]:
        assert [len(axes) for axes in figures] == [3, 2, 6]
        assert namespace["relative_error"] < 1e-5  # Original work/energy check.
        assert set(namespace["cell_paths"]) == {"a", "c"}
        assert all(v.shape == (11, 3, 3) for v in namespace["cell_paths"].values())
        assert "Sliding along a" in output and "Sliding along c" in output
    else:
        assert [len(axes) for axes in figures] == [2, 4]
        assert namespace["n_steps"] == 8  # Full 8, 16, 32, 64 step study.
        for name in namespace["RATES"]:
            assert name in output
        # The final axis compares the three consistent objective rates with
        # the two uncorrected tangents. Each consistent error must converge.
        for curve in curves[-5:][::2]:
            assert_allclose(curve[:, 0], [8, 16, 32, 64])
            assert np.all(np.diff(curve[:, 1]) < 0)
            assert curve[-1, 1] < 0.4 * curve[0, 1]


def test_optional_reference_examples(example, tmp_path):
    reference = pytest.importorskip("burnman")
    name, actual = example
    examples = Path(reference.__file__).resolve().parents[1] / "examples"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(examples))
        patch.delitem(sys.modules, "example_hyperelastic_mineral", raising=False)
        expected = run_example(examples / name, tmp_path)
    assert actual[1] == expected[1]  # Figure layouts, labels, and artists.
    assert len(actual[3]) == len(expected[3])
    for native, python in zip(actual[3], expected[3]):
        assert_allclose(native, python, rtol=2e-5, atol=2e-6)
    if name == EXAMPLES[0]:
        for key in ("a", "c"):
            assert_allclose(
                actual[0]["cell_paths"][key], expected[0]["cell_paths"][key], rtol=1e-9
            )
