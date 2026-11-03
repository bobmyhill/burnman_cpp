# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Reconstruction preserves saved equilibria and owns independent phase models."""

import gc

import numpy as np
import pytest
import burnman_cpp as bm

from conftest import make_model, oxide_params
from test_composition_contours import stored_grid
from test_property_grid import single_phase


@pytest.fixture
def retained(single_phase):
    source, phase = single_phase
    grid = bm.phase_diagram_grid(
        source,
        [phase],
        ["density", "molar_entropy"],
        resolution=(3, 3),
        store_solutions=True,
        relaxed=False,
    )
    return grid, phase


def forbid_solves(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reconstruction must not solve equilibria")

    for namespace in (bm, bm._core):
        for name in (
            "phase_diagram",
            "stable_equilibrium",
            "equilibrate",
            "_equilibrate",
            "phase_diagram_grid",
            "_phase_diagram_grid",
        ):
            if hasattr(namespace, name):
                monkeypatch.setattr(namespace, name, forbidden)


def test_native_grid_binds_models_and_snapshots_outlive_it(retained, monkeypatch):
    grid, phase = retained
    saved = grid.solution_at(1, 1)
    expected = {name: values[1, 1] for name, values in grid.values.items()}
    assert bm.reconstruct_assemblage(saved).density == pytest.approx(
        expected["density"]
    )
    data = grid.to_dict()

    def detached_snapshot():
        transient = bm.PhaseDiagramPropertyGrid.from_dict(data)
        transient.bind_phases([phase])
        return transient.solution_at(1, 1)

    saved = detached_snapshot()
    snapshot = saved.to_dict()
    phase.set_state(9e9, 1500.0)
    forbid_solves(monkeypatch)
    del grid
    gc.collect()
    first = bm.reconstruct_assemblage(saved)
    second = bm.reconstruct_assemblage(saved)
    for name, value in expected.items():
        assert getattr(first, name) == pytest.approx(value, rel=1e-12)
    assert first.pressure == saved.state.pressure
    assert first.temperature == saved.state.temperature
    assert first.n_moles == saved.n_moles
    first.set_state(7e9, 1400.0)
    assert second.get_phase(0).pressure == saved.state.pressure
    assert phase.pressure == 9e9
    assert saved.to_dict() == snapshot


@pytest.mark.parametrize("suffix", ["json", "json.gz"])
def test_loaded_grid_binds_original_models_without_changing_file(
    retained, tmp_path, suffix
):
    grid, phase = retained
    path = bm.save_property_grid(grid, tmp_path / f"grid.{suffix}")
    original = path.read_bytes()
    loaded = bm.load_property_grid(path, phases=[phase])
    saved = loaded.solution_at(1, 1)
    assert bm.reconstruct_assemblage(saved).density == pytest.approx(
        grid.values["density"][1, 1]
    )
    assert loaded.to_dict() == grid.to_dict()
    assert path.read_bytes() == original
    unbound = bm.load_property_grid(path).solution_at(1, 1)
    with pytest.raises(ValueError, match="No candidate models"):
        bm.reconstruct_assemblage(unbound)
    assert bm.reconstruct_assemblage(unbound, phases=[phase]).density == pytest.approx(
        grid.values["density"][1, 1]
    )


def test_binding_rejects_different_candidate_names_and_preserves_old_models(retained):
    grid, phase = retained
    wrong = phase.copy()
    wrong.set_name("different candidate")
    with pytest.raises(ValueError, match="names/order"):
        grid.bind_phases([wrong])
    assert bm.reconstruct_assemblage(grid.solution_at(1, 1)).density == pytest.approx(
        grid.values["density"][1, 1]
    )


def test_coexisting_instances_preserve_amounts_and_separate_compositions(monkeypatch):
    phase = bm.Solution(make_model(), [0.5, 0.5], name="oxide")
    grid = stored_grid(
        [phase],
        phase_states=lambda p, t: [
            dict(id=i, candidate_index=0, name=f"oxide #{i+1}", amount=n, composition=x)
            for i, n, x in ((0, 1.0, [0.2, 0.8]), (1, 3.0, [0.8, 0.2]))
        ],
    ).to_dict()
    for row in grid["solutions"]:
        for point in row:
            point["molar_fractions"] = [0.25, 0.75]
    grid = bm.PhaseDiagramPropertyGrid.from_dict(grid)
    grid.bind_phases([phase])
    forbid_solves(monkeypatch)
    saved = grid.solution_at(0, 0)
    restored = bm.reconstruct_assemblage(saved, averaging_scheme=bm.AveragingType.Reuss)
    assert restored.n_moles == 4.0
    np.testing.assert_array_equal(restored.molar_fractions, [0.25, 0.75])
    np.testing.assert_array_equal(restored.get_phase(0).molar_fractions, [0.2, 0.8])
    np.testing.assert_array_equal(restored.get_phase(1).molar_fractions, [0.8, 0.2])
    restored.get_phase(0).set_composition([0.7, 0.3])
    np.testing.assert_array_equal(restored.get_phase(1).molar_fractions, [0.8, 0.2])
    np.testing.assert_array_equal(phase.molar_fractions, [0.5, 0.5])
    np.testing.assert_array_equal(
        bm.reconstruct_assemblage(saved).get_phase(0).molar_fractions, [0.2, 0.8]
    )


def test_active_basis_restores_saved_coefficients_without_empty_sites(monkeypatch):
    phase = bm.Solution(make_model(), [1.0, 0.0], name="oxide")
    data = stored_grid(
        [phase],
        phase_states=lambda p, t: [
            dict(
                id=0,
                candidate_index=0,
                name="oxide",
                amount=1.0,
                composition=[1.0, 0.0],
            )
        ],
    ).to_dict()
    for row in data["solutions"]:
        for point in row:
            point["phase_bases"] = [[[1.0, 0.0]]]
            point["phase_compositions"] = [[1.0]]
    grid = bm.PhaseDiagramPropertyGrid.from_dict(data)
    grid.bind_phases([phase])
    forbid_solves(monkeypatch)
    assemblage = bm.reconstruct_assemblage(grid.solution_at(0, 0))
    assert assemblage.n_endmembers == 1
    assert np.isfinite(assemblage.endmember_partial_gibbs).all()
    assert assemblage.get_phase(0).formula == {"Mg": 1.0, "O": 1.0}
    np.testing.assert_array_equal(assemblage.molar_fractions, [1.0])


def test_relaxed_candidates_are_restored_as_frozen_ordinary_solutions(monkeypatch):
    endmembers = [
        (bm.Mineral(oxide_params("Mg", shift)), "[Mg]O") for shift in (0.0, 1000.0)
    ]
    base = bm.Solution(bm.IdealSolution(endmembers), [0.5, 0.5], name="spin")
    phase = bm.RelaxedSolution(base, [[1.0, -1.0]], [[0.5, 0.5]])
    grid = stored_grid(
        [phase],
        phase_states=lambda p, t: [
            dict(
                id=0,
                candidate_index=0,
                name=phase.name,
                amount=1.0,
                composition=[0.2, 0.8],
            )
        ],
    )
    grid.bind_phases([phase])
    forbid_solves(monkeypatch)
    assemblage = bm.reconstruct_assemblage(grid.solution_at(0, 0))
    assert not isinstance(assemblage.get_phase(0), bm.RelaxedSolution)
    assemblage.set_state(8e9, 1600.0)
    np.testing.assert_array_equal(assemblage.get_phase(0).molar_fractions, [0.2, 0.8])


@pytest.mark.parametrize("damage", ["amount", "basis", "moles", "candidate"])
def test_reconstruction_rejects_corrupt_retained_state(retained, damage):
    grid, phase = retained
    data = grid.to_dict()
    point = data["solutions"][1][1]
    if damage == "amount":
        point["phases"][0]["amount"] *= 2.0
    elif damage == "basis":
        point["phase_bases"][0] = [[1.0, 0.0]]
    elif damage == "moles":
        point["n_moles"] = 0.0
    else:
        point["phases"][0]["candidate_index"] = 1
    saved = bm.PhaseDiagramPropertyGrid.from_dict(data).solution_at(1, 1)
    with pytest.raises(ValueError, match="Invalid|disagree"):
        bm.reconstruct_assemblage(saved, phases=[phase])
