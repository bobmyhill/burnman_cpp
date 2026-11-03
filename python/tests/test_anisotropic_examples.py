# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native anisotropic polarization plotting and EOS consistency checks."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.tools.eos import check_anisotropic_eos_consistency
from test_anisotropic_mineral import make_model

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from burnman_cpp.tools import plot


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
