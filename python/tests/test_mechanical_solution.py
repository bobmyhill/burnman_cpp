# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Mechanical mixing without shared lattice sites or mixing entropy."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011


def make_model(reference=False):
    minerals, models = SLB_2011, bm
    if reference:
        minerals = pytest.importorskip("burnman.minerals.SLB_2011")
        models = pytest.importorskip("burnman.classes.solutionmodel")
    # Different structures, with labels that do not define shared sites.
    return models.MechanicalSolution(
        [(minerals.forsterite(), ""), (minerals.periclase(), "[Mg]O")]
    )


@pytest.mark.parametrize("temperature", [0.0, 300.0, 2000.0])
@pytest.mark.parametrize("x", [[0.4, 0.6], [1.0, 0.0], [2.0, 1.0], [1.5, -0.5]])
def test_no_mixing_contributions(temperature, x):
    model = make_model()
    assert isinstance(model, bm.SolutionModel)
    assert model.site_formulae == ["", "[Mg]O"]
    assert model.n_endmembers == 2
    assert model.n_sites == 0
    for name in (
        "excess_gibbs_free_energy",
        "excess_volume",
        "excess_entropy",
        "excess_enthalpy",
        "Cp_excess",
        "alphaV_excess",
        "VoverKT_excess",
    ):
        assert getattr(model, "compute_" + name)(1e9, temperature, x) == 0.0
    for name in ("gibbs_free_energies", "entropies", "volumes"):
        assert_array_equal(
            getattr(model, "compute_excess_partial_" + name)(1e9, temperature, x),
            np.zeros(2),
        )
    for name in ("gibbs", "entropy", "volume"):
        assert_array_equal(
            getattr(model, f"compute_{name}_hessian")(1e9, temperature, x),
            np.zeros((2, 2)),
        )
    for name in ("activities", "activity_coefficients"):
        assert_array_equal(
            getattr(model, "compute_" + name)(1e9, temperature, x), np.ones(2)
        )


@pytest.mark.parametrize("x", [[0.4, 0.6], [1.0, 0.0], [2.0, 1.0], [1.5, -0.5]])
def test_linear_properties_and_independent_state(x):
    model = make_model()
    solution = bm.Solution(model, x)
    solution.set_state(3e9, 1200.0)
    endmembers = [entry[0] for entry in model.endmembers]
    for mineral in endmembers:
        mineral.set_state(3e9, 1200.0)
    for prop, partial in (
        ("molar_gibbs", "partial_gibbs"),
        ("molar_entropy", "partial_entropies"),
        ("molar_volume", "partial_volumes"),
    ):
        values = np.array([getattr(m, prop) for m in endmembers])
        assert_allclose(getattr(solution, prop), np.dot(x, values), rtol=1e-14)
        assert_array_equal(getattr(solution, partial), values)
    assert_allclose(
        solution.molar_heat_capacity_p,
        np.dot(x, [m.molar_heat_capacity_p for m in endmembers]),
        rtol=1e-14,
    )
    copy = solution.copy()
    assert isinstance(copy.solution_model, bm.MechanicalSolution)
    assert copy.solution_model is not solution.solution_model
    copy.set_state(1e9, 500.0)
    copy.set_composition([3.0, -1.0])
    assert_array_equal(solution.molar_fractions, x)
    assert solution.solution_model.endmembers[0][0].temperature == 1200.0
    assert copy.solution_model.endmembers[0][0].temperature == 500.0
    assert not model.endmembers[0][0].has_state()
    # Recomposition must invalidate the cached Gibbs energy.
    solution.set_composition([0.0, 1.0])
    assert solution.molar_gibbs == endmembers[1].molar_gibbs


@pytest.mark.parametrize("x", [[0.4, 0.6], [1.0, 0.0], [2.0, 1.0], [1.5, -0.5]])
@pytest.mark.parametrize("pressure,temperature", [(1e5, 300.0), (3e9, 1200.0)])
def test_python_reference_and_adapter(x, pressure, temperature):
    burnman = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    reference = burnman.Solution(solution_model=make_model(True), molar_fractions=x)
    reference.set_state(pressure, temperature)
    native = bm.Solution(make_model(), x)
    native.set_state(pressure, temperature)
    adapted = from_burnman(reference)
    assert isinstance(adapted.solution_model, bm.MechanicalSolution)
    assert adapted.solution_model.site_formulae == ["", "[Mg]O"]
    assert_array_equal(adapted.molar_fractions, x)
    adapted.set_state(pressure, temperature)
    for prop in (
        "molar_gibbs",
        "molar_entropy",
        "molar_volume",
        "molar_enthalpy",
        "molar_helmholtz",
        "molar_internal_energy",
        "molar_heat_capacity_p",
        "molar_heat_capacity_v",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "partial_gibbs",
        "partial_entropies",
        "partial_volumes",
        "activities",
        "activity_coefficients",
    ):
        assert_allclose(getattr(native, prop), getattr(reference, prop), rtol=2e-8)
        assert_allclose(getattr(adapted, prop), getattr(reference, prop), rtol=2e-8)


def test_validation_and_single_endmember():
    with pytest.raises(ValueError, match="at least one"):
        bm.MechanicalSolution([])
    with pytest.raises(ValueError, match="None"):
        bm.MechanicalSolution([(None, "")])
    model = make_model()
    solution = bm.Solution(model, [0.4, 0.6])
    for x in ([1.0], [0.5, np.nan], [0.5, np.inf]):
        with pytest.raises(ValueError, match="coefficients"):
            bm.Solution(model, x)
        with pytest.raises(ValueError, match="coefficients"):
            solution.set_composition(x)
        with pytest.raises(ValueError, match="coefficients"):
            model.compute_activities(1e9, 300.0, x)
    single = bm.Solution(bm.MechanicalSolution([(SLB_2011.periclase(), "")]), [2.0])
    single.set_state(1e9, 1000.0)
    mineral = SLB_2011.periclase()
    mineral.set_state(1e9, 1000.0)
    assert single.molar_volume == 2.0 * mineral.molar_volume
    assert_array_equal(single.activities, [1.0])
