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

"""Run the complete EOS example and compare it with pinned Python BurnMan."""

from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def run_example(backend, path, cwd):
    """Snapshot each fit before later fits mutate the mineral; close plot windows."""
    fits, bands, figures = [], [], []
    fitting = False
    output = StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(cwd)

        def capture_fit(original):
            @wraps(original)
            def fit(*args, **kwargs):
                nonlocal fitting
                # Python's PTV wrapper calls the PT-property wrapper internally.
                if fitting:
                    return original(*args, **kwargs)
                fitting = True
                try:
                    model = original(*args, **kwargs)
                finally:
                    fitting = False
                snapshot = {
                    key: np.array(getattr(model, key), copy=True)
                    for key in (
                        "popt",
                        "pcov",
                        "data",
                        "data_covariances",
                        "data_mle",
                        "weighted_residuals",
                        "flags",
                        "goodness_of_fit",
                    )
                }
                snapshot["fit_params"] = list(model.fit_params)
                snapshot["param_tolerance"] = kwargs.get("param_tolerance")
                fits.append(snapshot)
                return model

            return fit

        for name in ("fit_PTV_data", "fit_PTp_data"):
            patch.setattr(
                backend.eos_fitting,
                name,
                capture_fit(getattr(backend.eos_fitting, name)),
            )
        original_bands = backend.nonlinear_fitting.confidence_prediction_bands

        def capture_bands(**kwargs):
            result = original_bands(**kwargs)
            bands.append(
                (
                    np.array(kwargs["x_array"], copy=True),
                    kwargs["confidence_interval"],
                    np.array(result, copy=True),
                )
            )
            return result

        patch.setattr(
            backend.nonlinear_fitting, "confidence_prediction_bands", capture_bands
        )

        def show():
            for number in plt.get_fignums():
                figure = plt.figure(number)
                figure.canvas.draw()
                figures.append(
                    [
                        (
                            ax.get_title(),
                            ax.get_xlabel(),
                            ax.get_ylabel(),
                            ax.get_legend_handles_labels()[1],
                            len(ax.collections),
                            len(ax.lines),
                        )
                        for ax in figure.axes
                    ]
                )
            # Emulate closing each interactive window before the next figure.
            plt.close("all")

        patch.setattr(plt, "show", show)
        plt.close("all")
        try:
            with redirect_stdout(output):
                namespace = runpy.run_path(str(path), run_name="__main__")
        finally:
            plt.close("all")
    return fits, bands, figures, namespace, output.getvalue()


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    # Data paths must work when the current directory is unrelated to examples/.
    return run_example(
        bm, ROOT / "examples/example_fit_eos.py", tmp_path_factory.mktemp("fit-eos")
    )


def test_example_fits_and_refits_outliers(example):
    fits, _, _, namespace, output = example
    assert len(fits) == 4
    assert [fit["data"].shape for fit in fits] == [(24, 3), (61, 3), (94, 3), (92, 3)]
    assert [len(fit["popt"]) for fit in fits] == [3, 5, 7, 7]
    assert [fit["param_tolerance"] for fit in fits] == [None, 1.0e-3, 1.0e-3, 1.0e-5]
    assert_allclose(
        [fit["goodness_of_fit"] for fit in fits],
        [
            0.4759109278322042,
            1.2212888873611405,
            1.3184130423390952,
            0.32060210303150455,
        ],
        rtol=1.0e-7,
    )
    assert_array_equal(namespace["indices"], [68, 70])
    keep = np.ones(len(fits[2]["data"]), dtype=bool)
    keep[namespace["indices"]] = False
    for key in ("data", "data_covariances", "flags"):
        assert_array_equal(fits[3][key], fits[2][key][keep])
    assert_array_equal(fits[3]["flags"][:20], ["H"] * 20)
    assert_array_equal(fits[3]["flags"][20:], ["V"] * 72)
    assert np.max(np.abs(fits[3]["weighted_residuals"])) < 3.0
    for fit in fits:
        assert np.isfinite(fit["popt"]).all()
        assert np.isfinite(fit["pcov"]).all()
        assert (np.diag(fit["pcov"]) > 0.0).all()
    assert "Optimized equation of state (outliers removed):" in output
    assert "There are 2 outliers (at the 90.0% confidence interval)" in output


def test_example_includes_all_figures_and_uncertainty_sections(example):
    _, bands, figures, namespace, _ = example
    assert [len(figure) for figure in figures] == [
        3,
        1,
        1,
        10,
        2,
        1,
        21,
        1,
        2,
        1,
        21,
        2,
        1,
        1,
        1,
        4,
        1,
    ]
    assert figures[1][0][0] == "Stishovite EoS (room temperature)"
    assert figures[2][0][2] == "Bulk modulus (GPa)"
    assert figures[13][0][0] == "Periclase volumes at 1 bar"
    assert "Dubrovinsky and Saxena (1997)" in figures[13][0][3]
    assert "Maximum likelihood estimates" in figures[14][0][3]
    assert [axis[2] for axis in figures[15]] == [
        "P wave velocity (km/s)",
        "Bulk modulus (GPa)",
        "Thermal expansion (/K)",
        "Thermal pressure (MPa/K)",
    ]
    assert figures[16][0][2] == "Gibbs energy uncertainty (kJ/mol)"
    assert len(bands) == 14
    for queries, confidence, values in bands:
        assert queries.shape == (101, 3)
        assert confidence == 0.95
        assert values.shape == (4, 101)
        # Derived-property prediction bands are undefined in the upstream API.
        assert np.isfinite(values[:2]).all()
        assert (values[0] <= values[1]).all()
    for i in (0, 1, 2, 4, 5, 6, 7, 12):
        assert_array_equal(bands[i][0][:, 1], np.full(101, 298.15))
    for i in (3, 8, 9, 10, 11, 13):
        assert_array_equal(bands[i][0][:, 1], np.full(101, 2000.0))
    assert namespace["DS1997_data"].shape == (30, 3)


def test_optional_python_example_parity(example):
    reference = pytest.importorskip("burnman")
    reference_root = Path(reference.__file__).resolve().parents[1]
    expected = run_example(
        reference,
        reference_root / "examples/example_fit_eos.py",
        reference_root / "examples",
    )
    assert len(example[0]) == len(expected[0]) == 4
    for actual, python in zip(example[0], expected[0]):
        for key in ("data", "data_covariances", "flags"):
            assert_array_equal(actual[key], python[key])
        assert actual["fit_params"] == python["fit_params"]
        assert actual["param_tolerance"] == python["param_tolerance"]
        assert_allclose(actual["popt"], python["popt"], rtol=1.0e-5)
        # Covariances involve finite differences and are more sensitive to rounding.
        assert_allclose(actual["pcov"], python["pcov"], rtol=2.0e-4)
        assert_allclose(actual["data_mle"], python["data_mle"], rtol=1.0e-7)
        assert_allclose(
            actual["weighted_residuals"],
            python["weighted_residuals"],
            rtol=1.0e-5,
            atol=2.0e-6,
        )
        assert_allclose(
            actual["goodness_of_fit"], python["goodness_of_fit"], rtol=1.0e-7
        )
    # The native example uses "Gibbs energy"; the pinned reference adds "free".
    # Adapt only this final y-axis label, keeping all other metadata checks exact.
    reference_axis = expected[2][-1][0]
    assert reference_axis[2] == "Gibbs free energy uncertainty (kJ/mol)"
    expected[2][-1][0] = (
        *reference_axis[:2],
        "Gibbs energy uncertainty (kJ/mol)",
        *reference_axis[3:],
    )
    assert example[2] == expected[2]  # Full figure sequence, labels and artist counts.
    assert_array_equal(example[3]["indices"], expected[3]["indices"])
    assert_allclose(
        example[3]["probabilities"], expected[3]["probabilities"], atol=1.0e-10
    )
    assert len(example[1]) == len(expected[1]) == 14
    for (x, confidence, values), (py_x, py_confidence, py_values) in zip(
        example[1], expected[1]
    ):
        assert confidence == py_confidence
        assert_allclose(x, py_x, rtol=1.0e-7)
        assert_allclose(values, py_values, rtol=2.0e-6, equal_nan=True)
    for name in (
        "PVT_MgO_Dewaele_et_al_2000.dat",
        "Hazen_1976_TV_periclase.dat",
        "Victor_Douglas_1963_deltaH_MgO.dat",
        "Dubrovinsky_Saxena_1997_TV_periclase.dat",
    ):
        assert (example[3]["DATA_DIR"] / name).read_bytes() == (
            reference_root / "burnman/data/input_fitting" / name
        ).read_bytes()
