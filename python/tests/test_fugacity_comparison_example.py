# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Saved-state oxygen potentials, O2 stoichiometry and comparison masking."""

from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm

from conftest import make_model
from test_composition_contours import stored_grid
from test_phase_diagram import pure

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def example(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Fugacity comparison must only read stored equilibria")

    for namespace in (bm, bm._core):
        for name in (
            "stable_equilibrium",
            "equilibrate",
            "_equilibrate",
            "phase_diagram",
            "refine_phase_diagram",
            "phase_diagram_grid",
            "_phase_diagram_grid",
        ):
            if hasattr(namespace, name):
                monkeypatch.setattr(namespace, name, forbidden)
    return runpy.run_path(
        str(EXAMPLES / "example_pyrolite_fugacity_comparison_step2_contours.py")
    )


def iron_oxide_grid(phases, indices=None):
    indices = range(len(phases)) if indices is None else indices
    return stored_grid(
        phases,
        phase_states=lambda p, t: [
            dict(
                id=3 * i,
                candidate_index=i,
                name=phases[i].name,
                amount=1.0,
                composition=[1.0],
            )
            for i in indices
        ],
    )


def test_relative_log_fugacity_has_the_O2_factor_and_correct_sign(example):
    wuestite = pure("wuestite", {"Fe": 1.0, "O": 1.0})
    hematite = pure("hematite", {"Fe": 2.0, "O": 3.0})
    shift = 5000.0
    shifted = bm.CombinedMineral(
        [hematite], [1.0], [shift, 0.0, 0.0], name="shifted hematite"
    )
    phases = [wuestite, hematite, shifted]
    first = iron_oxide_grid(phases, (0, 1))
    second = iron_oxide_grid(phases, (0, 2))
    difference, (first_mu, second_mu), diagnostics = example["log_fugacity_difference"](
        first, second, phases
    )
    # O2 = 2 Fe2O3 - 4 FeO, so shifting hematite by dG shifts mu(O2) by 2 dG.
    np.testing.assert_allclose(second_mu - first_mu, 2.0 * shift, rtol=1e-12)
    assert not any(diagnostics.values())
    expected = (
        2.0 * shift / (bm.constants.gas_constant * first.y[:, None] * np.log(10.0))
    )
    np.testing.assert_allclose(
        difference, np.broadcast_to(expected, difference.shape), rtol=1e-12
    )
    assert (difference > 0).all()
    same, _, diagnostics = example["log_fugacity_difference"](first, first, phases)
    np.testing.assert_array_equal(same, 0.0)
    assert not any(diagnostics.values())
    reverse, _, _ = example["log_fugacity_difference"](second, first, phases)
    np.testing.assert_allclose(reverse, -difference, rtol=1e-12)


def test_undefined_oxygen_potentials_and_failed_nodes_stay_masked(example):
    phase = pure("periclase", {"Mg": 1.0, "O": 1.0})
    grid = iron_oxide_grid([phase])
    values, diagnostics = example["oxygen_potential_grid"](grid, [phase])
    assert np.isnan(values).all()
    assert len(diagnostics) == values.size
    assert all("not defined" in message for message in diagnostics)

    phases = [
        pure("wuestite", {"Fe": 1.0, "O": 1.0}),
        pure("hematite", {"Fe": 2.0, "O": 3.0}),
    ]
    good = iron_oxide_grid(phases)
    data = good.to_dict()
    data["solutions"][0][0] = None
    data["equilibrium_success"][0][0] = False
    failed = bm.PhaseDiagramPropertyGrid.from_dict(data)
    difference, _, diagnostics = example["log_fugacity_difference"](
        good, failed, phases
    )
    assert np.isnan(difference[0, 0])
    np.testing.assert_array_equal(difference[1:], 0.0)
    assert not any(diagnostics.values())


def test_active_faces_restore_without_logarithms_of_empty_sites(example):
    oxide = bm.Solution(make_model(), [0.0, 1.0], name="oxide")
    hematite = pure("hematite", {"Fe": 2.0, "O": 3.0})
    grid = stored_grid(
        [oxide, hematite],
        phase_states=lambda p, t: [
            dict(
                id=0,
                candidate_index=0,
                name="oxide",
                amount=1.0,
                composition=[0.0, 1.0],
            ),
            dict(
                id=3, candidate_index=1, name="hematite", amount=1.0, composition=[1.0]
            ),
        ],
    ).to_dict()
    for row in grid["solutions"]:
        for state in row:
            state["phase_bases"][0] = [[0.0, 1.0]]
            state["phase_compositions"][0] = [1.0]
    restored = bm.PhaseDiagramPropertyGrid.from_dict(grid)
    values, diagnostics = example["oxygen_potential_grid"](restored, [oxide, hematite])
    assert not diagnostics
    assert np.isfinite(values).all()


def test_comparison_rejects_different_pressure_temperature_nodes(example):
    phases = [
        pure("wuestite", {"Fe": 1.0, "O": 1.0}),
        pure("hematite", {"Fe": 2.0, "O": 3.0}),
    ]
    first = iron_oxide_grid(phases)
    data = first.to_dict()
    data["x"] = [p + 1.0 for p in first.x]
    with pytest.raises(ValueError, match="same P-T grid"):
        example["log_fugacity_difference"](
            first, bm.PhaseDiagramPropertyGrid.from_dict(data), phases
        )


def test_compositions_preserve_cations_and_change_only_oxygen():
    example = runpy.run_path(str(EXAMPLES / "example_pyrolite_fugacity_comparison.py"))
    first, second = example["compositions"]().values()
    assert {e: n for e, n in first.items() if e != "O"} == {
        e: n for e, n in second.items() if e != "O"
    }
    assert second["O"] - first["O"] == pytest.approx(0.5 * first["Fe"] * 0.07)


def test_fmq_matches_the_assemblage_oxygen_chemical_potential(example):
    from types import SimpleNamespace
    from burnman_cpp.minerals import SLB_2024

    grid = SimpleNamespace(
        x=np.array([20.0e9, 25.0e9]), y=np.array([1773.15, 2023.15, 2273.15])
    )
    potentials = example["fmq_potential_grid"](grid)
    buffer = bm.Assemblage(
        [SLB_2024.fayalite(), SLB_2024.magnetite(), SLB_2024.quartz()],
        [1.0 / 3.0] * 3,
    )
    assert potentials.shape == (3, 2)
    for j, temperature in enumerate(grid.y):
        for i, pressure in enumerate(grid.x):
            buffer.set_state(pressure, temperature)
            assert potentials[j, i] == pytest.approx(
                buffer.chemical_potential([{"O": 2.0}])[0], abs=1e-8
            )
