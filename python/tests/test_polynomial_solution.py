# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Polynomial composition derivatives and mineral-dependent excesses."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011


def model(reference=False, transformed=False):
    if reference:
        minerals = pytest.importorskip("burnman.minerals.SLB_2011")
        models = pytest.importorskip("burnman.classes.solutionmodel")
    else:
        minerals, models = SLB_2011, bm
    a, b = minerals.forsterite(), minerals.fayalite()
    return models.PolynomialSolution(
        [(a, "[Mg]2SiO4"), (b, "[Fe]2SiO4")],
        ESV_interactions=[
            [12000.0, 3.0, 1e-6, 0, 1, 1, 1],
            [4000.0, -1.0, -2e-7, 0, 3, 1, 1],
        ],
        interaction_endmembers=[minerals.forsterite(), minerals.fayalite()],
        endmember_coefficients_and_interactions=[[0.2, -0.2, 0, 1, 1, 1]],
        transformation_matrix=(
            np.array([[0.8, 0.2], [0.1, 0.9]]) if transformed else None
        ),
    )


@pytest.mark.parametrize("transformed", [False, True])
def test_reference(transformed):
    native, python = model(False, transformed), model(True, transformed)
    for p, t, x in [(1e9, 300.0, [0.8, 0.2]), (20e9, 1200.0, [0.3, 0.7])]:
        python.set_composition(x)
        python.set_state(p, t)
        names = {
            "excess_gibbs_free_energy": "excess_gibbs_energy",
            "excess_partial_gibbs_free_energies": "excess_partial_gibbs_energies",
            "excess_entropy": "excess_entropy",
            "excess_volume": "excess_volume",
            "excess_partial_entropies": "excess_partial_entropies",
            "excess_partial_volumes": "excess_partial_volumes",
            "gibbs_hessian": "gibbs_hessian",
            "entropy_hessian": "entropy_hessian",
            "volume_hessian": "volume_hessian",
            "Cp_excess": "Cp_excess",
            "alphaV_excess": "alphaV_excess",
            "VoverKT_excess": "VoverKT_excess",
            "activities": "activities",
            "activity_coefficients": "activity_coefficients",
        }
        for actual, expected in names.items():
            assert_allclose(
                getattr(native, "compute_" + actual)(p, t, x),
                getattr(python, expected)(p, t, x),
                rtol=1e-9,
                atol=1e-12,
                err_msg=actual,
            )


def test_equation_of_state_derivatives_and_copy():
    s = bm.Solution(model(), [0.4, 0.6])
    p, t, dp, dt = 10e9, 1000.0, 1e5, 0.01
    s.set_state(p, t)
    v, kt, alpha, cp = (
        s.molar_volume,
        s.isothermal_bulk_modulus_reuss,
        s.thermal_expansivity,
        s.molar_heat_capacity_p,
    )
    values = []
    for pp, tt in [(p - dp, t), (p + dp, t), (p, t - dt), (p, t + dt)]:
        s.set_state(pp, tt)
        values.append((s.molar_volume, s.molar_entropy))
    assert_allclose(-(values[1][0] - values[0][0]) / (2 * dp * v), 1 / kt, rtol=1e-7)
    assert_allclose((values[3][0] - values[2][0]) / (2 * dt * v), alpha, rtol=1e-7)
    assert_allclose(t * (values[3][1] - values[2][1]) / (2 * dt), cp, rtol=1e-7)
    clone = s.copy()
    assert (
        clone.solution_model.interaction_endmembers[0]
        is not s.solution_model.interaction_endmembers[0]
    )
    s.set_state(p, t)
    g = s.molar_gibbs
    clone.set_state(5e9, 500.0)
    assert s.molar_gibbs == g
    s.set_state_with_volume(v, t)
    assert_allclose(s.pressure, p, rtol=1e-10)


def test_multicomponent_hessian_endpoints_and_validation():
    members = [(SLB_2011.forsterite(), f"[{x}]2SiO4") for x in ("Mg", "Fe", "Ca")]
    poly = bm.PolynomialSolution(members, [[2e4, 0.0, 0.0, 0, 2, 1, 1, 2, 1]])
    ideal = bm.IdealSolution(members)
    x = np.array([0.2, 0.3, 0.5])
    p, t = 1e9, 1000.0
    h = poly.compute_gibbs_hessian(p, t, x) - ideal.compute_gibbs_hessian(p, t, x)
    step = 1e-6
    for d in (np.array([1.0, -1.0, 0.0]), np.array([0.0, 1.0, -1.0])):
        mu = []
        for sign in (-1.0, 1.0):
            xx = x + sign * step * d
            mu.append(
                poly.compute_excess_partial_gibbs_free_energies(p, t, xx)
                - ideal.compute_excess_partial_gibbs_free_energies(p, t, xx)
            )
        assert_allclose((mu[1] - mu[0]) / (2 * step), h @ d, rtol=1e-8, atol=1e-6)
    for x in np.eye(3):
        assert np.isfinite(poly.compute_excess_partial_volumes(p, t, x)).all()
        assert np.isfinite(poly.compute_volume_hessian(p, t, x)).all()
    for row in (
        [1.0, 0.0, 0.0, 4, 1],
        [1.0, 0.0, 0.0, 0, 0.5, 1],
        [1.0, 0.0, 0.0, 0, -1],
    ):
        with pytest.raises(ValueError):
            bm.PolynomialSolution(members, [row])
    with pytest.raises(ValueError):
        bm.PolynomialSolution(members, transformation_matrix=np.eye(2))


def test_relaxation_escapes_a_symmetry_breaking_stationary_state():
    from test_anisotropic_mineral import make_model

    a = make_model()
    # G = (T-1000)*Q^2 + 1000*Q^4, up to a composition-independent term.
    poly = bm.PolynomialSolution(
        [(a, "[Mg]2SiO4"), (a, "[Mg]2SiO4")],
        [[-4000.0, 4.0, 0.0, 0, 1, 1, 1], [16000.0, 0.0, 0.0, 0, 2, 1, 2]],
    )
    scalar = bm.RelaxedSolution(
        bm.Solution(poly, [0.5, 0.5]), [[1.0, -1.0]], [[0.5, 0.5]]
    )
    source = bm.AnisotropicSolution(
        "Landau test",
        poly,
        lambda *args: (np.zeros((6, 6)),) * 3 + (np.zeros((3, 3, 2)),),
        {},
    )
    anisotropic = bm.RelaxedAnisotropicSolution(source, [[1.0, -1.0]], [[0.5, 0.5]])
    for s in (scalar, anisotropic):
        s.set_composition([1.0])
        for temperature in (1500.0, 500.0, 1500.0, 500.0):
            s.set_state(1e9, temperature)
            q = s.molar_fractions[0] - s.molar_fractions[1]
            assert_allclose(abs(q), 0.0 if temperature > 1000.0 else 0.5, atol=2e-6)


def test_parameter_snapshot_updates_are_validated_and_reset_caches():
    m = SLB_2011.forsterite()
    m.set_state(1e9, 1000.0)
    v = m.molar_volume
    params = m.params
    params.V_0 *= 1.02
    assert m.molar_volume == v
    m.set_parameters(params)
    assert m.pressure == 1e9 and m.temperature == 1000.0
    expected = bm.Mineral(params)
    expected.set_state(1e9, 1000.0)
    assert m.molar_volume != v
    assert_allclose(m.molar_volume, expected.molar_volume, rtol=1e-12)
    good = m.molar_volume
    params.molar_mass = -1.0
    with pytest.raises(ValueError):
        m.set_parameters(params)
    assert m.molar_volume == good


def test_volume_array_evaluation_preserves_solution_state():
    s = bm.Solution(model(), [0.4, 0.6])
    pressures = np.array([[1e9, 5e9], [10e9, 20e9]])
    temperatures = np.array([[300.0, 500.0], [1000.0, 1500.0]])
    volumes = s.evaluate(["V"], pressures, temperatures)[0]
    assert not s.has_state()
    values = s.evaluate_with_volumes(
        ["pressure", "molar_volume"], volumes, temperatures
    )
    assert not s.has_state()
    assert_allclose(values[0], pressures, rtol=1e-10)
    assert_allclose(values[1], volumes, rtol=1e-12)
    s.set_state(1e9, 300.0)
    state = s.molar_gibbs
    s.evaluate_with_volumes(["pressure"], volumes, temperatures)
    assert s.pressure == 1e9 and s.molar_gibbs == state
    with pytest.raises(ValueError):
        s.evaluate_with_volumes(["pressure"], [0.0], [300.0])
