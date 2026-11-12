# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Composition bounds must survive phase discovery, continuation and basis changes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import burnman_cpp as bm
from burnman_cpp.optimize.composition_fitting import fit_composition_to_solution
from burnman_cpp.tools import thermobarometry
from conftest import make_model, oxide_params


def isochemical_solution(constraints=None):
    mineral = bm.Mineral(oxide_params("Mg"))
    model = bm.IdealSolution([(mineral, "[A]O"), (mineral, "[B]O")])
    return bm.Solution(
        model,
        [0.9, 0.1],
        name="isochemical oxide",
        endmember_constraints=constraints,
    )


def candidates(transformed=False):
    solution = bm.Solution(make_model(), [0.9, 0.1], name="bounded oxide")
    solution.endmember_constraints = [[0.2, -0.8]]  # p_Fe <= 0.2
    iron = solution.solution_model.endmembers[1][0]
    if transformed:
        solution = bm.transform_solution_to_new_basis(
            solution, [[1.0, 0.0], [0.5, 0.5]], [0.8, 0.2], "bounded oxide"
        )
    return [solution, iron]


def test_evaluate_endmember_constraints():
    phase = bm.Solution(make_model(), [0.8, 0.2])
    assert phase.evaluate_endmember_constraints([0.5, 0.5]).size == 0
    phase.endmember_constraints = [[0.2, -0.8], [0.8, -0.2]]
    assert_allclose(phase.evaluate_endmember_constraints([0.5, 0.5]), [-0.3, 0.3])
    with pytest.raises(ValueError, match="molar_fractions"):
        phase.evaluate_endmember_constraints([1.0])


@pytest.mark.parametrize("transformed", [False, True])
def test_active_bound_is_respected_in_stable_equilibrium_and_continuation(transformed):
    phases = candidates(transformed)
    rock = bm.EquilibratedComposite({"MgO": 0.4, "FeO": 0.6}, phases)
    for pressure in [1e9, 1.01e9, 1e9]:
        rock.set_state(pressure, 1000.0)
        state = rock.equilibrium_state
        assert state.success, state.message
        assert state.mass_balance_error < 1e-8
        assert len(state.phases) == 2
        for phase in rock.phases:
            if phase.name == "bounded oxide":
                assert phase.formula["Fe"] == pytest.approx(0.2, abs=1e-8)
    assert_allclose(
        phases[0].endmember_constraints,
        [[0.2, -0.3]] if transformed else [[0.2, -0.8]],
    )
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 3
    diagram = bm.phase_diagram(
        {"Mg": 0.4, "Fe": 0.6, "O": 1.0},
        phases,
        (0.9e9, 1.1e9),
        (950.0, 1050.0),
        settings,
    )
    assert diagram.resolved, diagram.diagnostics
    for sample in diagram.samples:
        assert sample.success, sample.message
        assert sample.mass_balance_error < 1e-8


def test_constraint_changes_no_thermodynamic_evaluation_or_site_formula():
    bounded, _ = candidates()
    unrestricted = bm.Solution(make_model(), [0.9, 0.1], name="unrestricted")
    for phase in [bounded, unrestricted]:
        phase.set_state(1e9, 1000.0)
    assert bounded.molar_gibbs == pytest.approx(unrestricted.molar_gibbs)
    assert_allclose(bounded.endmember_occupancies, unrestricted.endmember_occupancies)
    assert (
        bounded.composition_bounds.shape[1]
        == unrestricted.composition_bounds.shape[1] + 1
    )


@pytest.mark.parametrize("bounds", [[[1.0]], [[np.nan, 0.0]]])
def test_invalid_constraint_width_or_values_are_rejected(bounds):
    phases = candidates()
    with pytest.raises(ValueError, match="Endmember constraints"):
        phases[0].endmember_constraints = bounds


def test_empty_composition_domain_is_rejected():
    phases = candidates()
    phases[0].endmember_constraints = [[-1.0, -1.0]]
    with pytest.raises(ValueError, match="nonempty bounded domain"):
        bm.stable_equilibrium({"Mg": 0.4, "Fe": 0.6, "O": 1.0}, phases, 1e9, 1000.0)


def test_constraints_validate_on_assignment_and_clear_without_changing_energy():
    phase = bm.Solution(make_model(), [0.8, 0.2], endmember_constraints=[[0.2, -0.8]])
    phase.set_state(1e9, 1000.0)
    gibbs = phase.molar_gibbs
    assert_allclose(phase.composition_bounds[:, -1], [0.2, -0.8])
    for invalid in ([0.2, -0.8], [[1.0]], [[np.nan, 0.0]]):
        with pytest.raises(ValueError):
            phase.endmember_constraints = invalid
        assert_allclose(phase.endmember_constraints, [[0.2, -0.8]])
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        phase.endmember_constraints = [[-1.0, -1.0]]
    assert phase.molar_gibbs == gibbs
    phase.endmember_constraints = None
    assert phase.endmember_constraints is None
    assert phase.composition_bounds.shape[1] == phase.endmember_occupancies.shape[1]
    phase.endmember_constraints = [[0.2, -0.8]]
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        phase.set_composition([0.5, 0.5])
    assert_allclose(phase.molar_fractions, [0.5, 0.5])


def test_solution_polytope_and_basis_transform_preserve_cap():
    phase = bm.Solution(make_model(), [0.8, 0.2], endmember_constraints=[[0.2, -0.8]])
    poly = bm.solution_polytope_from_endmember_occupancies(
        phase.endmember_occupancies,
        endmember_constraints=phase.endmember_constraints,
    )
    assert_allclose(np.sort(poly.vertices[:, 1]), [0.0, 0.2], atol=1e-10)
    assert_allclose(
        poly.endmember_occupancies, poly.vertices @ phase.endmember_occupancies
    )

    basis = np.array([[0.7, 0.3], [0.6, 0.4]])
    assert np.all(phase.endmember_constraints @ basis.T < 0.0)
    child = bm.transform_solution_to_new_basis(
        phase, basis, [2.0, -1.0], "bounded transformed oxide"
    )
    assert_allclose(child.endmember_constraints, phase.endmember_constraints @ basis.T)
    assert_allclose(child.molar_fractions @ basis, [0.8, 0.2])
    child_poly = bm.solution_polytope_from_endmember_occupancies(
        child.endmember_occupancies,
        endmember_constraints=child.endmember_constraints,
    )
    assert_allclose(
        np.sort((child_poly.vertices @ basis)[:, 1]), [0.0, 0.2], atol=1e-10
    )
    mineral = bm.transform_solution_to_new_basis(phase, [[0.0, 1.0]])
    assert not isinstance(mineral, bm.Solution)


def test_unrepresentable_polytope_constraint_is_rejected():
    with pytest.raises(ValueError, match="cannot be expressed in site occupancies"):
        bm.solution_polytope_from_endmember_occupancies(
            [[1.0, 0.0], [1.0, 0.0]],
            endmember_constraints=[[1.0, -1.0]],
        )


def test_polytope_simplification_and_composition_fitting_respect_cap():
    phase = isochemical_solution([[0.2, -0.8]])
    poly = bm.composite_polytope_at_constrained_composition(
        bm.Assemblage([phase], [1.0]), phase.formula
    )
    assert_allclose(np.sort(poly.vertices[:, 1]), [0.0, 0.2], atol=1e-10)

    oxide = bm.Solution(make_model(), [0.8, 0.2], endmember_constraints=[[0.2, -0.8]])
    amounts, _, _ = fit_composition_to_solution(
        oxide, ["Mg_A", "Fe_A"], [0.2, 1.8], np.eye(2)
    )
    assert amounts[1] <= 0.2 + 1e-8


def test_simplification_preserves_constraint_in_reduced_basis():
    mg = bm.Mineral(oxide_params("Mg"))
    fe = bm.Mineral(oxide_params("Fe"))
    phase = bm.Solution(
        bm.IdealSolution([(mg, "[A]O"), (mg, "[B]O"), (fe, "[C]O")]),
        [0.8, 0.2, 0.0],
        endmember_constraints=[[0.25, -0.75, 0.25]],
    )
    reduced = bm.simplify_composite_with_composition(
        bm.Assemblage([phase], [1.0]), mg.formula
    ).phases[0]
    assert isinstance(reduced, bm.Solution)
    assert reduced.n_endmembers == 2
    assert_allclose(
        reduced.endmember_constraints,
        phase.endmember_constraints @ reduced.basis.T,
    )
    poly = bm.solution_polytope_from_endmember_occupancies(
        reduced.endmember_occupancies,
        endmember_constraints=reduced.endmember_constraints,
    )
    assert np.max((poly.vertices @ reduced.basis)[:, 1]) <= 0.25 + 1e-10


def test_relaxation_properties_ignore_cap_at_pressure_and_volume():
    phase = isochemical_solution([[0.2, -0.8]])
    relaxed = bm.RelaxedSolution(phase, [[-1.0, 1.0]], [[1.0, 0.0]])
    relaxed.set_composition([1.0], q_initial=[0.1])
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        relaxed.set_state(1e9, 1000.0)
    assert relaxed.molar_fractions[1] == pytest.approx(0.5, abs=1e-6)
    volume = relaxed.molar_volume
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        relaxed.set_state_with_volume(volume, 1000.0)
    assert relaxed.molar_fractions[1] == pytest.approx(0.5, abs=1e-6)
    assert relaxed.molar_volume == pytest.approx(volume, rel=1e-10)
    relaxed.endmember_constraints = [[0.7, -0.3]]
    relaxed.set_state(1e9, 1000.0)
    assert relaxed.molar_fractions[1] == pytest.approx(0.5, abs=1e-6)


def test_anisotropic_relaxation_and_susceptibility_ignore_cap():
    from test_original_anisotropicsolution import make_nonorthotropic_solution

    phase = make_nonorthotropic_solution(two_fos=True, molar_fractions=[0.6, 0.2, 0.2])
    phase.endmember_constraints = [[0.2, -0.8, 0.2]]
    relaxed = bm.RelaxedAnisotropicSolution(
        phase, [[1.0, -1.0, 0.0]], [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    relaxed.set_composition([0.8, 0.2], q_initial=[0.2])
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        relaxed.set_state(20e9, 1000.0)
    assert relaxed.molar_fractions[1] == pytest.approx(0.4, abs=1e-6)
    assert np.linalg.norm(relaxed._d2Fdqdq_fixed_epsT_pinv) > 0.0
    volume = relaxed.molar_volume
    with pytest.warns(UserWarning, match="violates.*endmember_constraints"):
        relaxed.set_state_with_volume(volume, 1000.0)
    assert relaxed.molar_fractions[1] == pytest.approx(0.4, abs=1e-6)


def test_reaction_reduction_and_thermobarometry_respect_active_bounds():
    phase = isochemical_solution([[0.2, -0.8]])
    phase.set_composition([0.8, 0.2])
    assemblage = bm.Assemblage([phase, bm.Mineral(oxide_params("Mg"))], [0.5, 0.5])
    assemblage.set_state(1e9, 1000.0)
    reactions = thermobarometry.get_reaction_matrix(assemblage, small_fraction_tol=1e-8)
    assert reactions.shape == (1, 3)
    assert reactions[0, 1] / reactions[0, 0] == pytest.approx(0.25, abs=1e-8)

    phase.free_compositional_vectors = [[-1.0, 1.0]]
    phase.baseline_composition = phase.molar_fractions
    with pytest.raises(ValueError, match="nonnegative composition bounds"):
        thermobarometry.assemblage_set_state_from_params(assemblage, [1e9, 1000.0, 0.1])
    phase.free_compositional_vectors = None
    phase.set_composition([0.5, 0.5])
    with pytest.raises(ValueError, match="Fixed solution composition"):
        thermobarometry.estimate_conditions(
            assemblage, guessed_conditions=[1e9, 1000.0]
        )
