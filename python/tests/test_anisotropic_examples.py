# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Run the upstream anisotropic walkthroughs through the native API."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.tools.eos import check_anisotropic_eos_consistency
from test_anisotropic_mineral import make_model

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from burnman_cpp.tools import plot

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ["example_anisotropy.py", "example_anisotropic_mineral.py"]


def run_example(path, cwd):
    figures, contours, curves = [], [], []
    output = StringIO()
    with pytest.MonkeyPatch.context() as patch, plt.rc_context():
        patch.chdir(cwd)
        original = Axes.contourf

        def contourf(self, *args, **kwargs):
            contours.append(np.array(args[2], copy=True))
            return original(self, *args, **kwargs)

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
                curves.extend(
                    line.get_xydata().copy()
                    for ax in figure.axes
                    if ax.name != "polar"
                    for line in ax.lines
                )
            plt.close("all")

        patch.setattr(Axes, "contourf", contourf)
        patch.setattr(plt, "show", show)
        plt.close("all")
        try:
            with redirect_stdout(output), np.printoptions():
                namespace = runpy.run_path(str(path), run_name="__main__")
        finally:
            plt.close("all")
    return namespace, figures, contours, curves, output.getvalue()


@pytest.fixture(scope="module", params=EXAMPLES)
def example(request, tmp_path_factory):
    cwd = tmp_path_factory.mktemp(request.param[:-3])
    return request.param, cwd, run_example(ROOT / "examples" / request.param, cwd)


def test_complete_examples(example):
    name, cwd, (namespace, figures, contours, _, output) = example
    if name == EXAMPLES[0]:
        assert [len(axes) for axes in figures] == [12]
        assert len(contours) == 6
        assert (cwd / "output_figures/example_anisotropy.png").stat().st_size > 1000
        assert "Basalt isotropic elastic properties" in output
        assert "Talc elastic properties" in output
        assert_allclose(namespace["Vp"], np.sqrt(88e9 / 2735.0))
    else:
        assert [len(axes) for axes in figures] == [2, 9, 18]
        assert len(contours) == 9
        assert namespace["alphas"].shape == (101, 4)
        assert namespace["C"].shape == (101, 6, 6)
        assert namespace["m"].pressure == 3e9
        assert "Model forsterite properties at 1.00 GPa and 1600.00 K" in output
    assert all(np.isfinite(values).all() for values in contours)


def test_optional_reference_examples(example, tmp_path):
    reference = pytest.importorskip("burnman")
    name, _, actual = example
    (tmp_path / "output_figures").mkdir()
    expected = run_example(
        Path(reference.__file__).resolve().parents[1] / "examples" / name, tmp_path
    )
    assert actual[1] == expected[1]  # Complete figure layouts, labels and artists.
    for collection in (2, 3):  # Contoured properties and temperature/pressure curves.
        assert len(actual[collection]) == len(expected[collection])
        for native, python in zip(actual[collection], expected[collection]):
            assert_allclose(native, python, rtol=1e-8, atol=1e-10)
    if name == EXAMPLES[1]:
        for key in ("alphas", "extensions", "vectors", "C"):
            assert_allclose(actual[0][key], expected[0][key], rtol=1e-8, atol=1e-10)


def test_velocity_arrows_follow_native_polarization_columns(monkeypatch):
    mineral = bm.CubicMaterial(3500.0, [240e9, 90e9, 70e9])
    make_lines = plot._make_lines
    calls = 0

    def checked_lines(positions, vectors, scales):
        nonlocal calls
        mode = 1 + calls % 2
        velocities, _ = mineral.wave_velocities(positions)
        acoustic = mineral.christoffel_tensor(positions)
        assert_allclose(
            np.einsum("...ij,...j->...i", acoustic, vectors) / 1e9,
            mineral.density * velocities[..., mode, None] ** 2 * vectors / 1e9,
            atol=1e-12,
        )
        calls += 1
        return make_lines(positions, vectors, scales)

    monkeypatch.setattr(plot, "_make_lines", checked_lines)
    lines = plot._make_velocity_lines(mineral, 0.02)
    assert calls == 6
    assert all(np.isfinite(values).all() for values in lines)


@pytest.mark.parametrize("system", ["orthorhombic", "triclinic"])
def test_checker_uses_native_derivatives_and_detects_inconsistency(system, capsys):
    mineral = make_model(system)
    states = []
    assert check_anisotropic_eos_consistency(
        mineral,
        verbose=True,
        equilibration_function=lambda m: states.append((m.pressure, m.temperature)),
    )
    assert len(states) > 5
    assert "All EoS consistency constraints satisfied" in capsys.readouterr().out

    class Inconsistent:
        def __getattr__(self, name):
            value = getattr(mineral, name)
            return 1.1 * value if name == "thermal_expansivity_tensor" else value

    assert not check_anisotropic_eos_consistency(Inconsistent())
