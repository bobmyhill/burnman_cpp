# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Selected reaction responses and Python BurnMan RelaxedComposite parity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011

REACTIONS = np.array([[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0]])


@pytest.fixture
def transition():
    ol, wad = SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_wadsleyite()
    for phase in (ol, wad):
        phase.set_composition([0.9, 0.1])
    assemblage = bm.Assemblage([ol, wad], [1.0, 0.0])
    composition = dict(assemblage.formula)
    prm = bm.get_equilibration_parameters(assemblage, composition)
    boundaries = []
    for index in (1, 0):
        result = bm.equilibrate(
            composition,
            assemblage,
            [
                bm.TemperatureConstraint(1600.0),
                bm.PhaseFractionConstraint(index, 0.0, prm),
            ],
            tol=[1.0, 1.0e-6, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9],
        )
        assert result.sol_array.item().success, result.sol_array.item().message
        boundaries.append(assemblage.pressure)
    return assemblage, composition, np.array(boundaries)


def vectors(assemblage, mode):
    if mode == "full":
        return assemblage.reaction_basis
    if mode == "empty":
        return np.empty((0, assemblage.n_endmembers))
    return REACTIONS[[0 if mode == "exchange" else 1]]


@pytest.mark.parametrize("mode", ["full", "exchange", "phase_change", "empty"])
def test_selected_responses_preserve_full_equilibrium(transition, mode):
    assemblage, composition, boundaries = transition
    relaxed = bm.RelaxedComposite(assemblage, vectors(assemblage, mode))
    assert isinstance(relaxed, bm.Assemblage)
    assert relaxed.unrelaxed is assemblage
    assert relaxed.phases[0] is assemblage.phases[0]
    assert not relaxed.has_state()
    relaxed.set_state(boundaries.mean(), 1600.0)
    assert_allclose(assemblage.reaction_affinities, 0.0, atol=1.0e-5)
    assert_allclose(relaxed.molar_fractions, assemblage.molar_fractions)
    assert_allclose(relaxed.dndq, vectors(assemblage, mode).T)
    assert relaxed.dqdz_relaxed.shape == (relaxed.n_relaxation_vectors, 2)
    for element, amount in composition.items():
        assert relaxed.n_moles * relaxed.formula[element] == pytest.approx(amount)
    assert relaxed.molar_gibbs == pytest.approx(assemblage.molar_gibbs)
    assert relaxed.K_T <= assemblage.isothermal_bulk_modulus_reuss * (1.0 + 1.0e-12)
    assert relaxed.C_p >= assemblage.molar_heat_capacity_p * (1.0 - 1.0e-12)
    assert relaxed.C_v == pytest.approx(
        relaxed.C_p - relaxed.alpha**2 * relaxed.K_T * relaxed.V * relaxed.T
    )
    assert relaxed.K_S == pytest.approx(relaxed.K_T * relaxed.C_p / relaxed.C_v)
    if mode == "empty":
        assert relaxed.C_p == pytest.approx(assemblage.molar_heat_capacity_p)
        assert relaxed.K_T == pytest.approx(assemblage.isothermal_bulk_modulus_reuss)
    # Python receives independent snapshots of cached matrices.
    original = relaxed.dqdz_relaxed.copy()
    matrix = relaxed.dqdz_relaxed
    matrix[:] = 0.0
    assert_allclose(relaxed.dqdz_relaxed, original)


def test_full_relaxation_matches_finite_equilibrium_derivatives_and_mole_scaling(
    transition,
):
    assemblage, composition, boundaries = transition
    relaxed = bm.RelaxedComposite(assemblage, assemblage.reaction_basis)
    p, t = boundaries.mean(), 1600.0
    bulk = {element: 7.0 * amount for element, amount in composition.items()}
    relaxed.set_state(p, t, bulk_composition=bulk)
    assert relaxed.n_moles == pytest.approx(7.0)
    assert relaxed.V == pytest.approx(7.0 * relaxed.molar_volume)
    assert relaxed.C_p == pytest.approx(7.0 * relaxed.molar_heat_capacity_p)
    analytical = relaxed.dndq @ relaxed.dqdz_relaxed
    kt, alpha, cp = relaxed.K_T, relaxed.alpha, relaxed.molar_heat_capacity_p
    for column, step in ((0, 1.0e4), (1, 0.02)):
        points = []
        for sign in (1.0, -1.0):
            state = [p, t]
            state[column] += sign * step
            relaxed.set_state(*state, bulk_composition=bulk)
            points.append(
                (
                    bm.get_endmember_amounts(assemblage),
                    relaxed.molar_volume,
                    relaxed.molar_entropy,
                )
            )
        derivative = (points[0][0] - points[1][0]) / (2.0 * step)
        assert_allclose(derivative, analytical[:, column], rtol=2.0e-5, atol=1.0e-15)
        dv = (points[0][1] - points[1][1]) / (2.0 * step)
        ds = (points[0][2] - points[1][2]) / (2.0 * step)
        if column == 0:
            assert dv == pytest.approx(-relaxed.molar_volume / kt, rel=2.0e-5)
            assert ds == pytest.approx(-alpha * relaxed.molar_volume, rel=2.0e-5)
        else:
            assert dv == pytest.approx(alpha * relaxed.molar_volume, rel=2.0e-5)
            assert t * ds == pytest.approx(cp, rel=2.0e-5)


def test_state_copy_evaluate_and_lifetime(transition):
    assemblage, composition, boundaries = transition
    relaxed = bm.RelaxedComposite(assemblage, REACTIONS[[0]])
    p = boundaries.mean()
    relaxed.set_state(p, 1600.0, bulk_composition=composition)
    old_cp = relaxed.C_p
    assemblage.set_state(p + 1.0e6, 1600.0)
    relaxed.copy_state_from_unrelaxed()
    assert relaxed.P == p + 1.0e6
    assert relaxed.C_p != old_cp
    pressures = np.linspace(*boundaries, 5)[1:-1].reshape(1, 3)
    temperatures = np.full_like(pressures, 1600.0)
    for initially_set in (True, False):
        if not initially_set:
            relaxed = bm.RelaxedComposite(assemblage, REACTIONS[[0]])
        values = relaxed.evaluate(
            ["K_S", "alpha", "C_p", "molar_mass"], pressures, temperatures
        )
        assert values.shape == (4, 1, 3)
        assert np.isfinite(values).all()
        assert relaxed.has_state() == initially_set
        if initially_set:
            assert relaxed.P == p + 1.0e6
        matrices = relaxed.evaluate(["dqdz_relaxed"], pressures, temperatures)
        assert matrices.shape == (1, 1, 3, 1, 2)
        mixed = relaxed.evaluate(["K_T", "dqdz_relaxed"], pressures, temperatures)
        assert isinstance(mixed, list)
        with pytest.raises(ValueError, match="same shape"):
            relaxed.evaluate(["K_T"], pressures, temperatures.ravel())
        with pytest.raises(ValueError):
            relaxed.evaluate(["K_T"], pressures * np.nan, temperatures)
        assert relaxed.has_state() == initially_set
    # Shared ownership keeps the underlying assemblage alive.
    del assemblage
    relaxed.set_state(p, 1600.0)
    assert np.isfinite(relaxed.K_T)
    clone = relaxed.copy()
    assert isinstance(clone, bm.RelaxedComposite)
    assert clone.phases[0] is not relaxed.phases[0]
    assert clone.phases[0] is clone.unrelaxed.phases[0]
    assert clone.molar_heat_capacity_p == pytest.approx(relaxed.C_p)
    assert clone.evaluate(["K_T"], [p], [1600.0]).shape == (1, 1)


def test_invalid_reaction_vectors(transition):
    assemblage, _, _ = transition
    for invalid in (np.ones((1, 4)), np.zeros((1, 3)), np.full((1, 4), np.nan)):
        with pytest.raises(ValueError):
            bm.RelaxedComposite(assemblage, invalid)
    with pytest.raises(ValueError, match="assemblage"):
        bm.RelaxedComposite(None, REACTIONS)


@pytest.mark.parametrize("mode", ["full", "exchange", "phase_change", "empty"])
def test_optional_python_reference(transition, mode):
    reference = pytest.importorskip("burnman")
    from burnman.minerals import SLB_2011

    assemblage, composition, boundaries = transition
    native = bm.RelaxedComposite(assemblage, vectors(assemblage, mode))
    phases = [SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_wadsleyite()]
    for phase in phases:
        phase.set_composition([0.9, 0.1])
    ordinary = reference.Composite(phases, [0.5, 0.5])
    comparison = reference.RelaxedComposite(ordinary, vectors(assemblage, mode))
    bulk = {element: 3.0 * amount for element, amount in composition.items()}
    for fraction in (0.1, 0.5, 0.9):
        pressure = boundaries[0] + fraction * np.diff(boundaries).item()
        native.set_state(pressure, 1600.0, bulk_composition=bulk)
        # Use matching equilibrium estimates to compare the response formulas
        # independently of differences in the two Newton solvers.
        for phase, native_phase in zip(phases, assemblage.phases):
            phase.set_composition(native_phase.molar_fractions)
        ordinary.set_fractions(native.molar_fractions)
        comparison.set_state(pressure, 1600.0, bulk_composition=bulk)
        for name in (
            "molar_fractions",
            "dqdz_relaxed",
            "_d2Gdqdq_fixed_PT",
            "_d2Gdqdz",
            "_d2Gdzdz_Q",
            "_d2Gdzdz",
            "K_T",
            "K_S",
            "alpha",
            "C_p",
            "C_v",
            "gr",
            "molar_volume",
            "molar_gibbs",
            "molar_entropy",
            "molar_mass",
            "internal_energy",
            "helmholtz",
            "gibbs",
            "enthalpy",
            "volume",
            "entropy",
            "heat_capacity_p",
            "heat_capacity_v",
        ):
            assert_allclose(
                getattr(native, name),
                getattr(comparison, name),
                rtol=2.0e-7,
                atol=1.0e-13,
                err_msg=name,
            )
