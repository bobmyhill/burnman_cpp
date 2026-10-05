import gc

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params

R = 8.31446261815324


def test_mineral_parameters_and_reference_properties():
    params = oxide_params()
    params["equation_of_state"] = "bm3"
    mineral = bm.Mineral(params)
    with pytest.raises(RuntimeError, match="set_state"):
        _ = mineral.density
    mineral.set_state(0.0, 300.0)
    assert mineral.molar_volume == pytest.approx(params["V_0"])
    assert mineral.density == pytest.approx(params["molar_mass"] / params["V_0"])
    assert mineral.get_density() == mineral.density
    copied_params = mineral.params
    copied_params.V_0 *= 2.0
    mineral.reset()
    assert mineral.molar_volume == pytest.approx(params["V_0"])
    eos = bm.make_eos("bm3")
    volume = eos.compute_volume(1e9, 300.0, mineral.params)
    assert eos.compute_pressure(300.0, volume, mineral.params) == pytest.approx(
        1e9, rel=1e-9
    )


@pytest.mark.parametrize("alias", ["n", "napfu"])
def test_fractional_atom_count_parameters(alias):
    params = oxide_params()
    params.pop("n")
    params[alias] = 2.5
    params["formula"] = {"Mg": 1.25, "O": 1.25}
    mineral = bm.Mineral(params)
    assert mineral.params.napfu == 2.5
    assert mineral.formula == params["formula"]
    typed = bm.MineralParams()
    typed.napfu = 81.5
    assert typed.napfu == 81.5


def test_ideal_solution_and_composition_cache(solution):
    solution.set_state(1e9, 2000.0)
    np.testing.assert_allclose(solution.activities, [0.8, 0.2])
    np.testing.assert_allclose(solution.activity_coefficients, [1.0, 1.0])
    expected = R * 2000 * (0.8 * np.log(0.8) + 0.2 * np.log(0.2))
    assert solution.excess_gibbs == pytest.approx(expected)
    old_mass = solution.molar_mass
    assert solution.formula["Mg"] == pytest.approx(0.8)
    solution.set_composition([0.3, 0.7])
    np.testing.assert_allclose(solution.activities, [0.3, 0.7])
    assert solution.formula["Mg"] == pytest.approx(0.3)
    assert solution.molar_mass != old_mass
    expected = R * 2000 * (0.3 * np.log(0.3) + 0.7 * np.log(0.7))
    assert solution.excess_gibbs == pytest.approx(expected)


def test_regular_solution_activities(endmembers):
    model = bm.SymmetricRegularSolution(endmembers, energy_interaction=[[13000.0]])
    solution = bm.Solution(model, [0.7, 0.3])
    solution.set_state(1e9, 1800.0)
    expected_gamma = np.exp(13000.0 * np.array([0.3, 0.7]) ** 2 / (R * 1800.0))
    np.testing.assert_allclose(
        solution.activity_coefficients, expected_gamma, rtol=1e-12
    )
    np.testing.assert_allclose(
        solution.activities, expected_gamma * [0.7, 0.3], rtol=1e-12
    )
    asymmetric = bm.AsymmetricRegularSolution(endmembers, [1.0, 1.0], [[13000.0]])
    np.testing.assert_allclose(
        asymmetric.compute_activities(1e9, 1800.0, [0.7, 0.3]), solution.activities
    )


def test_shared_models_have_independent_state(endmembers):
    model = bm.IdealSolution(endmembers)
    a = bm.Solution(model, [0.8, 0.2])
    b = bm.Solution(model, [0.2, 0.8])
    a.set_state(1e9, 1000.0)
    volume_a = a.molar_volume
    b.set_state(5e9, 2000.0)
    a.reset()
    assert a.molar_volume == pytest.approx(volume_a)


@pytest.mark.parametrize("trace", [2.0e-8, 5.0e-7, 3.0e-6])
def test_trace_fraction_hessian_differentiates_chemical_potentials(endmembers, trace):
    model = bm.IdealSolution(endmembers)
    fractions = np.array([1.0 - trace, trace])
    direction = np.array([-1.0, 1.0])
    h = trace * 1.0e-3
    derivative = (
        model.compute_excess_partial_gibbs_free_energies(
            1.0e9, 600.0, fractions + h * direction
        )
        - model.compute_excess_partial_gibbs_free_energies(
            1.0e9, 600.0, fractions - h * direction
        )
    ) / (2 * h)
    np.testing.assert_allclose(
        model.compute_gibbs_hessian(1.0e9, 600.0, fractions) @ direction,
        derivative,
        rtol=2.0e-6,
        atol=0.05,
    )


def test_assemblage_lifetimes_and_numpy_copies(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    del solution
    gc.collect()
    assemblage.set_state(1e9, 2000.0)
    assert assemblage.get_phase(0) is assemblage.get_phase(-1)
    assert np.isfinite(assemblage.density)
    matrix = assemblage.stoichiometric_matrix
    snapshot = matrix.copy()
    matrix[:] = 0.0
    np.testing.assert_array_equal(assemblage.stoichiometric_matrix, snapshot)
    assemblage.clear_computed_properties()
    del assemblage
    gc.collect()
    np.testing.assert_array_equal(matrix, 0.0)


def test_endmember_amounts_with_second_solution(two_phase_assemblage):
    np.testing.assert_allclose(
        bm.get_endmember_amounts(two_phase_assemblage), [0.32, 0.08, 0.12, 0.48]
    )


@pytest.mark.parametrize(
    "fractions", [[1.0], [-0.1, 1.1], [np.nan, 1.0], [0.0, 0.0], [0.6, 0.6]]
)
def test_invalid_solution_fractions(solution, fractions):
    with pytest.raises(ValueError):
        solution.set_composition(fractions)


def test_invalid_parameters_and_models(endmembers):
    with pytest.raises(ValueError, match="Unknown mineral parameter"):
        bm.mineral_params({"typo": 1.0})
    with pytest.raises(ValueError, match="more than once"):
        bm.mineral_params({"n": 2, "napfu": 2})
    with pytest.raises(ValueError, match="Unknown equation of state"):
        bm.make_eos("unknown")
    with pytest.raises(ValueError):
        bm.Mineral({"equation_of_state": "slb3", "molar_mass": 0.04})
    with pytest.raises(ValueError):
        bm.IdealSolution([])
    with pytest.raises(ValueError):
        bm.SymmetricRegularSolution(endmembers, [[1.0, 2.0]])
    with pytest.raises(ValueError):
        bm.AsymmetricRegularSolution(endmembers, [1.0], [[1.0]])
    with pytest.raises(ValueError):
        bm.Assemblage([], [])
