import gc

import numpy as np
import pytest

import burnman_cpp as bm

R = 8.31446261815324
COMPOSITION = {"Mg": 0.5, "Fe": 0.5, "O": 1.0}


def test_single_solution_equilibrium(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    result = bm.equilibrate(
        {"Mg": 0.6, "Fe": 0.4, "O": 1.0},
        assemblage,
        [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
        store_iterates=True,
    )
    assert result.sol_array.shape == (1, 1)
    solve = result.sol_array[0, 0]
    assert solve.success, solve.message
    assert solve.code == 0
    np.testing.assert_allclose(assemblage.get_phase(0).molar_fractions, [0.6, 0.4])
    assert solve.F_norm < 1e-8
    assert solve.iteration_history is not None
    assert len(solve.iteration_history.x) > 1
    assert result.prm.n_parameters == len(solve.x)


def test_two_solution_equilibrium_mass_balance_and_partition(two_phase_assemblage):
    result = bm.equilibrate(
        COMPOSITION,
        two_phase_assemblage,
        [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
        tol=1e-9,
    )
    solve = result.sol_array[0, 0]
    assert solve.success, solve.message
    expected_mg = 1.0 / (1.0 + np.exp(-10000.0 / (R * 2000.0)))
    np.testing.assert_allclose(
        two_phase_assemblage.molar_fractions, [0.5, 0.5], atol=1e-8
    )
    np.testing.assert_allclose(
        two_phase_assemblage.get_phase(0).molar_fractions,
        [expected_mg, 1.0 - expected_mg],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        two_phase_assemblage.get_phase(1).molar_fractions,
        [1.0 - expected_mg, expected_mg],
        atol=1e-8,
    )
    amounts = bm.get_endmember_amounts(two_phase_assemblage)
    totals = two_phase_assemblage.stoichiometric_matrix.T @ amounts
    np.testing.assert_allclose(
        totals, [COMPOSITION[e] for e in two_phase_assemblage.elements], atol=1e-9
    )
    np.testing.assert_allclose(two_phase_assemblage.reaction_affinities, 0.0, atol=1e-6)


def test_constraint_grid_and_result_lifetime(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    result = bm.equilibrate(
        COMPOSITION,
        assemblage,
        [
            [bm.PressureConstraint(p) for p in (1e9, 2e9)],
            [bm.TemperatureConstraint(t) for t in (1500.0, 2000.0)],
        ],
    )
    grid = result.sol_array
    assert grid.shape == (2, 2)
    assert all(solve.success for solve in grid.flat)
    for i, pressure in enumerate((1e9, 2e9)):
        for j, temperature in enumerate((1500.0, 2000.0)):
            np.testing.assert_allclose(grid[i, j].x[:2], [pressure, temperature])
    del result
    gc.collect()
    bm.set_composition_and_state_from_parameters(assemblage, grid[0, 0].x)
    assert assemblage.temperature == 1500.0
    assert assemblage.pressure == 1e9


def test_equilibrium_jacobian_against_finite_differences(two_phase_assemblage):
    assemblage = two_phase_assemblage
    result = bm.equilibrate(
        COMPOSITION,
        assemblage,
        [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
        tol=1e-9,
    )
    solve = result.sol_array[0, 0]
    assert solve.success, solve.message
    parameters = result.prm

    def residual(values):
        bm.set_composition_and_state_from_parameters(assemblage, values)
        amounts = bm.get_endmember_amounts(assemblage)
        return np.concatenate(
            [
                [values[0] - 1e9, values[1] - 2000.0],
                assemblage.reaction_affinities,
                assemblage.reduced_stoichiometric_matrix.T @ amounts
                - parameters.reduced_composition_vector,
            ]
        )

    steps = [1e5, 0.01] + [1e-6] * (len(solve.x) - 2)
    columns = []
    for index, step in enumerate(steps):
        delta = np.zeros_like(solve.x)
        delta[index] = step
        columns.append(
            (residual(solve.x + delta) - residual(solve.x - delta)) / (2.0 * step)
        )
    numerical = np.column_stack(columns)
    np.testing.assert_allclose(solve.J, numerical, rtol=2e-6, atol=1e-5)
    bm.set_composition_and_state_from_parameters(assemblage, solve.x)


def test_free_composition_and_phase_composition_constraint(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    free = [{"Mg": -1.0, "Fe": 1.0, "O": 0.0}]
    parameters = bm.get_equilibration_parameters(assemblage, COMPOSITION, free)
    assert len(solution.site_names) == 2
    constraint = bm.PhaseCompositionConstraint(
        0, solution.site_names, [1.0, 0.0], [1.0, 1.0], 0.3, assemblage, parameters
    )
    result = bm.equilibrate(
        COMPOSITION,
        assemblage,
        [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0), constraint],
        free_compositional_vectors=free,
    )
    solve = result.sol_array[0, 0, 0]
    assert solve.success, solve.message
    assert assemblage.get_phase(0).molar_fractions[0] == pytest.approx(0.3)
    assert solve.x[-1] == pytest.approx(0.2)


def test_invalid_constraints(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    with pytest.raises(ValueError, match="constraint groups"):
        bm.equilibrate(COMPOSITION, assemblage, [bm.PressureConstraint(1e9)])
    with pytest.raises(ValueError, match="empty"):
        bm.equilibrate(COMPOSITION, assemblage, [[], bm.TemperatureConstraint(2000.0)])
    with pytest.raises(ValueError, match="n_parameters"):
        bm.equilibrate(
            COMPOSITION,
            assemblage,
            [bm.LinearXConstraint([1.0], 0.0), bm.TemperatureConstraint(2000.0)],
        )
    with pytest.raises(ValueError, match="absent"):
        bm.equilibrate(
            {**COMPOSITION, "Si": 1.0},
            assemblage,
            [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
        )
    with pytest.raises(IndexError):
        assemblage.get_phase(1)
    with pytest.raises(ValueError, match="n_parameters"):
        bm.equilibrate(
            COMPOSITION,
            assemblage,
            [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
            tol=[1.0, 0.001],
        )
    nested = bm.Assemblage([assemblage], [1.0])
    with pytest.raises(ValueError, match="flat assemblage"):
        bm.equilibrate(
            COMPOSITION,
            nested,
            [bm.PressureConstraint(1e9), bm.TemperatureConstraint(2000.0)],
        )


def test_volume_constraint_jacobian(solution):
    assemblage = bm.Assemblage([solution], [1.0])
    assemblage.set_state(1e9, 2000.0)
    composition = dict(assemblage.formula)
    target_volume = assemblage.molar_volume
    result = bm.equilibrate(
        composition,
        assemblage,
        [bm.PressureConstraint(1e9), bm.VolumeConstraint(target_volume)],
    )
    solve = result.sol_array[0, 0]
    assert solve.success, solve.message
    values = solve.x.copy()
    values[1] += 0.01
    bm.set_composition_and_state_from_parameters(assemblage, values)
    volume_plus = assemblage.n_moles * assemblage.molar_volume
    values[1] -= 0.02
    bm.set_composition_and_state_from_parameters(assemblage, values)
    volume_minus = assemblage.n_moles * assemblage.molar_volume
    numerical = (volume_plus - volume_minus) / 0.02
    assert solve.J[1, 1] == pytest.approx(numerical, rel=1e-6, abs=1e-16)


@pytest.mark.parametrize(
    "constraint_type, property_name",
    [
        (bm.VolumeConstraint, "molar_volume"),
        (bm.EntropyConstraint, "molar_entropy"),
    ],
)
def test_solve_for_temperature(solution, constraint_type, property_name):
    assemblage = bm.Assemblage([solution], [1.0])
    assemblage.set_state(2e9, 1800.0)
    composition = dict(assemblage.formula)
    target = getattr(assemblage, property_name) * assemblage.n_moles
    assemblage.set_state(2e9, 1500.0)
    result = bm.equilibrate(
        composition,
        assemblage,
        [bm.PressureConstraint(2e9), constraint_type(target)],
        tol=1e-8,
    )
    solve = result.sol_array[0, 0]
    assert solve.success, solve.message
    assert assemblage.temperature == pytest.approx(1800.0, rel=1e-8)


def test_per_parameter_tolerances(two_phase_assemblage):
    result = bm.equilibrate(
        COMPOSITION,
        two_phase_assemblage,
        [bm.PressureConstraint(100.0e9), bm.TemperatureConstraint(2000.0)],
        tol=[1.0, 1.0e-6, 1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9],
    )
    solve = result.sol_array.item()
    assert solve.success, solve.message
    np.testing.assert_allclose(
        two_phase_assemblage.molar_fractions, [0.5, 0.5], atol=1.0e-9
    )
    np.testing.assert_allclose(
        two_phase_assemblage.reaction_affinities, 0.0, atol=1.0e-6
    )
