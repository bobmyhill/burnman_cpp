# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Physical constraints, phase changes and atomic modes on equilibrium paths."""

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24
from conftest import oxide_params


@pytest.mark.parametrize("pair", ["PT", "PS", "PV", "TS", "TV", "SV"])
@pytest.mark.parametrize("vary", [0, 1, 2])
def test_linear_and_constant_coordinate_paths(pair, vary):
    phase = bm.Mineral(oxide_params())
    phase.set_state(4.0e9, 1000.0)
    reference = dict(
        P=4.0e9, T=1000.0, S=2 * phase.molar_entropy, V=2 * phase.molar_volume
    )
    start = {
        k: reference[k] * (0.999 if vary in (j, 2) else 1.0) for j, k in enumerate(pair)
    }
    end = {
        k: reference[k] * (1.001 if vary in (j, 2) else 1.0) for j, k in enumerate(pair)
    }
    path = bm.calculate_equilibrium_path(
        {"Mg": 2.0, "O": 2.0},
        [phase],
        start,
        end,
        n_points=3,
        pressure_range=(2.0e9, 6.0e9),
        temperature_range=(700.0, 1400.0),
    )
    assert path.success.all(), path.diagnostics
    assert not path.diagnostics
    for key in pair:
        np.testing.assert_allclose(
            path.coordinates(key), np.linspace(start[key], end[key], 3), rtol=1.0e-7
        )
    np.testing.assert_allclose(path.molar_amounts, 2.0, atol=1.0e-9)
    np.testing.assert_allclose(path.atomic_amounts, 4.0, atol=1.0e-9)
    np.testing.assert_allclose(path.fractions(), 1.0)


def test_atomic_modes_account_for_different_formula_sizes_and_bulk_scale():
    phases = [bm.Mineral(oxide_params()), SLB24.qtz()]
    path = bm.calculate_equilibrium_path(
        {"Mg": 1.5, "Si": 0.5, "O": 2.5},
        phases,
        {"P": 1.0e8, "T": 1000.0},
        {"P": 2.0e8, "T": 1100.0},
        n_points=3,
    )
    assert path.success.all(), path.diagnostics
    np.testing.assert_allclose(path.fractions("molar"), [[0.75, 0.25]] * 3, atol=1.0e-8)
    np.testing.assert_allclose(
        path.fractions("atomic"), [[2 / 3, 1 / 3]] * 3, atol=1.0e-8
    )
    np.testing.assert_allclose(path.atomic_amounts.sum(axis=1), 4.5, atol=1.0e-8)


def test_path_discovers_new_phases_and_follows_lowest_gibbs_energy():
    phases = [SLB24.qtz(), SLB24.coes(), SLB24.st()]
    path = bm.calculate_equilibrium_path(
        {"Si": 1.0, "O": 2.0},
        phases,
        {"P": 1.0e9, "T": 1000.0},
        {"P": 20.0e9, "T": 1300.0},
        n_points=5,
    )
    assert path.success.all(), path.diagnostics
    assert all(not phase.has_state() for phase in phases)
    expected = []
    for state in path.states:
        for phase in phases:
            phase.set_state(state.pressure, state.temperature)
        expected.append(np.argmin([p.molar_gibbs for p in phases]))
    assert len(set(expected)) > 1
    np.testing.assert_array_equal(path.molar_amounts.argmax(axis=1), expected)


def test_reverse_path_and_coordinate_key_order():
    phase = bm.Mineral(oxide_params())
    path = bm.calculate_equilibrium_path(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        {"T": 1200.0, "P": 5.0e9},
        {"P": 3.0e9, "T": 1000.0},
        n_points=3,
    )
    assert path.success.all(), path.diagnostics
    np.testing.assert_allclose(
        path.coordinates("P"), [5.0e9, 4.0e9, 3.0e9], rtol=1.0e-8
    )
    np.testing.assert_allclose(
        path.coordinates("T"), [1200.0, 1100.0, 1000.0], rtol=1.0e-8
    )


def test_curved_path_properties_match_the_pure_mineral_without_mutating_it():
    phase = bm.Mineral(oxide_params())
    pressure = np.array([1.0e9, 5.0e9, 2.0e9])
    temperature = np.array([1000.0, 1100.0, 1400.0])
    names = ["density", "shear_wave_velocity", "thermal_expansivity"]
    path = bm.sample_equilibrium_path(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        {"T": temperature, "P": pressure},
        properties=names,
    )
    assert path.success.all(), path.diagnostics
    assert not path.property_diagnostics
    assert not phase.has_state()
    for i, (p, t) in enumerate(zip(pressure, temperature)):
        phase.set_state(p, t)
        for name in names:
            assert path.values[name][i] == pytest.approx(
                getattr(phase, name), rel=1.0e-9
            )


def test_path_relaxed_expansivity_matches_equilibrium_volume_response():
    # The active MgO-FeO face retains distinct equilibrated high/low-spin Fe.
    # Its metamorphic expansivity must agree with new volume equilibria.
    fp = SLB24.ferropericlase()
    bulk = {"Mg": 0.9, "Fe": 0.1, "O": 1.0}
    path = bm.calculate_equilibrium_path(
        bulk,
        [fp],
        {"P": 110.0e9, "T": 1999.9},
        {"P": 110.0e9, "T": 2000.1},
        n_points=3,
        properties=["thermal_expansivity"],
    )
    assert path.success.all(), path.diagnostics
    assert not path.property_diagnostics
    volume = path.coordinates("V")
    derivative = (volume[-1] - volume[0]) / 0.2 / volume[1]
    assert path.values["thermal_expansivity"][1] == pytest.approx(
        derivative, rel=1.0e-5
    )


@pytest.mark.parametrize(
    "coordinates",
    [
        {"P": [1.0e9], "T": [1000.0, 1100.0]},
        {"P": [], "T": []},
        {"P": [1.0e9, np.nan], "T": [1000.0, 1100.0]},
        {"P": [1.0e9], "X": [0.5]},
    ],
)
def test_sampled_path_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        bm.sample_equilibrium_path(
            {"Mg": 1.0, "O": 1.0}, [bm.Mineral(oxide_params())], coordinates
        )


def test_failed_and_excluded_states_remain_missing_in_modes_and_plots(monkeypatch):
    from types import SimpleNamespace
    import burnman_cpp.tools.equilibrium_path as paths

    def state(success=True, outside=False):
        return SimpleNamespace(
            success=success,
            outside_model_domain=outside,
            message="test domain or convergence failure",
            pressure=1.0e9,
            temperature=1000.0,
            phases=[SimpleNamespace(candidate_index=0, amount=1.0, composition=[1.0])],
        )

    states = [state(), state(False), state(outside=True), state()]
    monkeypatch.setattr(
        paths._core, "_equilibrium_path", lambda *args: (states, {}, [], 0)
    )
    path = bm.calculate_equilibrium_path(
        {"Mg": 1.0, "O": 1.0},
        [bm.Mineral(oxide_params())],
        {"P": 1.0e9, "T": 1000.0},
        {"P": 2.0e9, "T": 1100.0},
        n_points=4,
    )
    np.testing.assert_array_equal(path.success, [True, False, False, True])
    assert np.isnan(path.fractions()[1:3]).all()
    assert len(path.diagnostics) == 2
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = bm.plot_mode_box(path, x=np.arange(4))
    assert np.isnan(ax.lines[0].get_ydata()[1:3]).all()
    plt.close(fig)


@pytest.mark.parametrize(
    "start,end",
    [
        ({"P": 1.0e9, "T": 1000.0}, {"P": 2.0e9, "S": 20.0}),
        ({"P": 1.0e9, "T": 1000.0}, {"P": 1.0e9, "T": 1000.0}),
        ({"P": 1.0e9, "X": 0.2}, {"P": 2.0e9, "X": 0.3}),
        ({"P": 1.0e9, "T": 1000.0}, {"P": 2.0e9, "T": np.nan}),
    ],
)
def test_path_rejects_ambiguous_or_nonphysical_endpoints(start, end):
    with pytest.raises(ValueError):
        bm.calculate_equilibrium_path(
            {"Mg": 1.0, "O": 1.0}, [bm.Mineral(oxide_params())], start, end
        )
