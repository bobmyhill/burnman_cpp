# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Compositional solutions and nonnegative phase amounts."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import nnls

import burnman_cpp as bm
from burnman_cpp import minerals
from burnman_cpp.optimize.composition_fitting import (
    DummyCompositionSolution,
    fit_composition_to_solution,
    fit_phase_proportions_to_bulk_composition as fit,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "examples/data/input_fitting/Bertka_Fei_1997_mars_mantle.dat"

COMPOSITION_CASES = [
    (["Mg2SiO4", "Fe2SiO4"], ["[Mg]2SiO4", "[Fe]2SiO4"], [0.2, 0.8]),
    (
        [
            "Mg3Al2Si3O12",
            "Fe3Al2Si3O12",
            "Ca3Al2Si3O12",
            "Ca3Fe2Si3O12",
            "Mg3Cr2Si3O12",
        ],
        [
            "[Mg]3[Al]2Si3O12",
            "[Fe]3[Al]2Si3O12",
            "[Ca]3[Al]2Si3O12",
            "[Ca]3[Fef]2Si3O12",
            "[Mg]3[Cr]2Si3O12",
        ],
        [0.2, 0.3, 0.1, 0.15, 0.25],
    ),
    (
        ["MgSiO3", "FeSiO3", "Mg1/2Fe1/2SiO3"],
        ["[Mg][Si]O3", "[Fe][Si]O3", "[Mg1/2Fe1/2][Si]O3"],
        [0.3, 0.1, 0.6],
    ),
    (["CaCO3"], ["Ca[C][O]3"], [1.0]),
    (["MgO", "Mg2O2"], ["[Mg]O", "[Mg]2O2"], [0.3, 0.7]),
]


@pytest.mark.parametrize("normalize", [False, True])
def test_solution_fit_amounts_covariance_and_residual(normalize):
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    assert (
        bm.optimize.composition_fitting.fit_composition_to_solution
        is fit_composition_to_solution
    )
    amounts = np.array([7 / 3, 19 / 3])
    sum_amounts = 26 / 3
    expected = amounts / sum_amounts if normalize else amounts
    # The information from independent Mg, Fe and O observations is [[2,1],[1,2]].
    amount_covariance = np.array([[2, -1], [-1, 2]]) / 3
    projection = np.eye(2) - expected[:, None]
    expected_covariance = projection @ amount_covariance @ projection.T
    if normalize:
        expected_covariance /= sum_amounts**2
    actual = fit_composition_to_solution(
        solution, ["Mg", "Fe", "O"], [2, 6, 9], None, normalize=normalize
    )
    assert_allclose(actual[0], expected, rtol=1e-12)
    assert_allclose(actual[1], expected_covariance, rtol=1e-12)
    assert_allclose(actual[2], 1 / 26 if normalize else 1 / 3, rtol=1e-12)
    assert solution.molar_fractions.size == 0
    assert not solution.has_state()


def test_converted_variables_honor_coefficients_and_definition_order():
    solution = DummyCompositionSolution(
        ["MgFeSiO4", "MgFeSiO4"], ["[Mg][Fe]SiO4", "[Fe][Mg]SiO4"]
    )
    conversions = {
        "twice_Mg": {"Mg_A": 2.0},
        "Mg_equal": {"twice_Mg": 0.5, "Mg_B": -1.0},
    }
    amounts, covariance, residual = fit_composition_to_solution(
        solution, ["Mg_A", "Mg_B", "Mg_equal"], [0.8, 0.2, 0], None, conversions
    )
    assert_allclose(amounts, [0.5, 0.5], atol=1e-12)
    assert_allclose(covariance, np.array([[1, -1], [-1, 1]]) / 6, atol=1e-12)
    assert_allclose(residual, 0.18, rtol=1e-12)
    assert conversions["Mg_equal"] == {"twice_Mg": 0.5, "Mg_B": -1.0}


def test_solution_fit_constrains_sites_and_permits_signed_endmembers():
    solution = DummyCompositionSolution(
        ["Mg1/2Fe1/2O", "FeO"], ["[Mg1/2Fe1/2]O", "[Fe]O"]
    )
    amounts, _, residual = fit_composition_to_solution(
        solution, ["Mg", "Fe"], [1, 0], np.eye(2)
    )
    assert_allclose(amounts, [2, -1], atol=1e-12)
    assert residual < 1e-24
    solution.set_composition(amounts)
    assert_allclose(solution.site_occupancies, [1, 0], atol=1e-12)

    oxides = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    amounts, _, _ = fit_composition_to_solution(oxides, ["Mg", "Fe"], [-0.2, 1], None)
    assert_allclose(amounts, [0, 1], atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_observations_still_enforce_explicit_absence(reverse):
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    observations = [1, 0, 1] if not reverse else [0, 1, 1]
    amounts, _, residual = fit_composition_to_solution(
        solution, ["Mg", "Mg", "Fe"], observations, None
    )
    assert_allclose(amounts, [0, 1], atol=1e-12)
    assert_allclose(residual, 1.0, rtol=1e-12)


def test_zero_total_is_only_returned_without_normalization():
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    with pytest.raises(ValueError, match="nonpositive total"):
        fit_composition_to_solution(solution, ["Mg", "Fe"], [0, 0], None)
    amounts, covariance, residual = fit_composition_to_solution(
        solution, ["Mg", "Fe"], [0, 0], None, normalize=False
    )
    assert_allclose(amounts, [0, 0], atol=0)
    assert_allclose(covariance, np.eye(2), atol=0)
    assert residual == 0


@pytest.mark.parametrize("real_model", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("covariance_kind", ["identity", "correlated"])
def test_optional_solution_fit_python_burnman_parity(
    real_model, normalize, covariance_kind
):
    burnman = pytest.importorskip("burnman")
    pytest.importorskip("cvxpy")
    from burnman.optimize.composition_fitting import (
        DummyCompositionSolution as ReferenceSolution,
        fit_composition_to_solution as reference_fit,
    )

    elements, sites, _ = COMPOSITION_CASES[1]
    actual_solution = (
        minerals.JH15.garnet()
        if real_model
        else DummyCompositionSolution(elements, sites)
    )
    expected_solution = (
        burnman.minerals.JH_2015.garnet()
        if real_model
        else ReferenceSolution(elements, sites)
    )
    # Record a composition and state to check that fitting does not change them.
    actual_solution.set_composition([0.2] * 5)
    actual_solution.set_state(1e9, 1000)
    variables = ["Fe", "Ca", "Mg", "Cr", "Al", "Si", "Fe3+"]
    covariance = np.eye(7) * 0.01**2
    if covariance_kind == "correlated":
        covariance += 2e-5
    observations = np.random.RandomState(100).multivariate_normal(
        [1.1, 2.0, 0, 0, 1.9, 3.0, 0.1], covariance
    )
    conversions = {"Fe3+": {"Fef_B": 1.0}}
    actual = fit_composition_to_solution(
        actual_solution, variables, observations, covariance, conversions, normalize
    )
    expected = reference_fit(
        expected_solution, variables, observations, covariance, conversions, normalize
    )
    assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-9)
    assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-12)
    assert_allclose(actual[2], expected[2], rtol=1e-6, atol=1e-8)
    assert_allclose(actual_solution.molar_fractions, [0.2] * 5)
    assert actual_solution.pressure == 1e9
    assert actual_solution.temperature == 1000
    # The same fitting information is supplied by a composition-only solution.
    equivalent = fit_composition_to_solution(
        DummyCompositionSolution(elements, sites),
        variables,
        observations,
        covariance,
        conversions,
        normalize,
    )
    for value, comparison in zip(actual, equivalent):
        assert_allclose(value, comparison, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "variables,values,covariance",
    [
        ([], [], None),
        (["Mg", "Fe"], [1], None),
        (["Mg", "Fe"], [1, np.nan], None),
        (["Mg", "Fe"], [1, 2], np.eye(3)),
        (["Mg", "Fe"], [1, 2], [[1, 0], [0, -1]]),
        (["Mg", "Fe"], [1, 2], [[1, np.nan], [np.nan, 1]]),
        (["Mg", "unknown"], [1, 2], None),
        (["Mg"], [1], None),
    ],
)
def test_solution_fit_rejects_invalid_observations(variables, values, covariance):
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    with pytest.raises(ValueError):
        fit_composition_to_solution(solution, variables, values, covariance)


@pytest.mark.parametrize(
    "conversions",
    [
        {"Mg": {"Fe": 1.0}},
        {"": {"Fe": 1.0}},
        {"extra": {"unknown": 1.0}},
        {"extra": {"extra": 1.0}},
        {"first": {"second": 1.0}, "second": {"Fe": 1.0}},
        {"extra": {"Fe": np.nan}},
        {"extra": {"Fe": np.inf}},
    ],
)
def test_solution_fit_rejects_undefined_or_invalid_conversions(conversions):
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    with pytest.raises(ValueError):
        fit_composition_to_solution(solution, ["Mg", "Fe"], [1, 2], None, conversions)


def test_dummy_composition_solution_chemistry_and_composition_changes():
    solution = DummyCompositionSolution(
        endmember_element_formulae=["Mg2SiO4", "Fe2SiO4"],
        endmember_site_formulae=["[Mg]2SiO4", "[Fe]2SiO4"],
    )
    assert isinstance(solution, bm.Solution)
    assert (
        bm.optimize.composition_fitting.DummyCompositionSolution
        is DummyCompositionSolution
    )
    assert solution.n_endmembers == 2
    assert solution.endmember_formulae == [
        {"Mg": 2, "Si": 1, "O": 4},
        {"Fe": 2, "Si": 1, "O": 4},
    ]
    assert solution.solution_model.site_formulae == ["[Mg]2SiO4", "[Fe]2SiO4"]
    assert solution.site_names == ["Mg_A", "Fe_A"]
    assert_allclose(solution.endmember_occupancies, np.eye(2))
    assert_allclose(solution.solution_model.endmember_noccupancies, 2 * np.eye(2))
    for request in (
        lambda: solution.formula,
        lambda: solution.site_occupancies,
        solution.site_formula,
    ):
        with pytest.raises((ValueError, RuntimeError)):
            request()
    solution.set_composition([0.2, 0.8])
    assert_allclose(solution.site_occupancies, [0.2, 0.8])
    assert solution.formula == {"Mg": 0.4, "Fe": 1.6, "Si": 1.0, "O": 4.0}
    assert solution.site_formula() == "[Mg0.20Fe0.80]2SiO4"
    assert not solution.has_state()
    # Returned arrays are snapshots, including the site chemistry on the model.
    occupancies = solution.solution_model.endmember_noccupancies
    occupancies[:] = 0
    assert_allclose(solution.solution_model.endmember_noccupancies, 2 * np.eye(2))
    copied = solution.copy()
    assert isinstance(copied, DummyCompositionSolution)
    copied.set_composition([1.0, 0.0])
    assert (
        copied.site_formula(precision=3, print_absent_species=False) == "[Mg1.000]2SiO4"
    )
    assert copied.formula["Mg"] == 2.0
    assert solution.site_formula() == "[Mg0.20Fe0.80]2SiO4"
    solution.set_composition([0.0, 1.0])
    assert solution.site_formula(print_absent_species=False) == "[Fe1.00]2SiO4"
    assert solution.formula["Mg"] == 0.0


def test_site_formula_optional_occupancies_and_no_thermodynamics():
    solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
    assert solution.site_formula(site_occupancies=[0.3, 0.7]) == "[Mg0.30Fe0.70]O"
    assert solution.site_formula(1, False, [-1e-15, 1.0]) == "[Fe1.0]O"
    with pytest.raises(ValueError):
        solution.site_formula(precision=-1, site_occupancies=[0.3, 0.7])
    for occupancies in ([1], [np.nan, 1]):
        with pytest.raises(ValueError):
            solution.site_formula(site_occupancies=occupancies)
    solution.set_composition([0.3, 0.7])
    solution.set_state(1.0e9, 1000.0)
    for property_name in (
        "molar_volume",
        "density",
        "molar_gibbs",
        "activities",
        "gibbs_hessian",
    ):
        with pytest.raises(NotImplementedError):
            getattr(solution, property_name)
    with pytest.raises(NotImplementedError):
        solution.solution_model.compute_activities(1e9, 1000, [0.3, 0.7])


def test_dummy_solution_allows_signed_coordinates_for_physical_sites():
    solution = DummyCompositionSolution(
        ["MgO", "FeO", "Mg1/2Fe1/2O"], ["[Mg]O", "[Fe]O", "[Mg1/2Fe1/2]O"]
    )
    solution.set_composition([-0.2, 0.4, 0.8])
    assert_allclose(solution.site_occupancies, [0.2, 0.8])
    for fractions in (
        [-0.5, 1.5, 0],
        [1, 0],
        [0, 0, 0],
        [np.nan, 0, 1],
        [np.inf, 0, 1],
    ):
        with pytest.raises(ValueError):
            solution.set_composition(fractions)
    assert_allclose(solution.molar_fractions, [-0.2, 0.4, 0.8])


@pytest.mark.parametrize(
    "elements,sites",
    [
        ([], []),
        (["MgO"], ["[Mg]O", "[Fe]O"]),
        (["XxO"], ["[Xx]O"]),
        (["Mg-1O"], ["[Mg]O"]),
        (["MgO"], [""]),
        (["MgO"], ["MgO"]),
        (["MgO"], ["[MgO"]),
        (["MgO"], ["Mg]O"]),
        (["MgO"], ["[]O"]),
        (["MgO"], ["[[Mg]]O"]),
        (["MgO"], ["[Mg0.5]O"]),
        (["MgO", "FeO"], ["[Mg]O", "[Fe][O]"]),
    ],
)
def test_dummy_solution_rejects_invalid_formulae(elements, sites):
    with pytest.raises((ValueError, RuntimeError)):
        DummyCompositionSolution(elements, sites)


@pytest.mark.parametrize("elements,sites,fractions", COMPOSITION_CASES)
def test_optional_dummy_solution_python_burnman_parity(elements, sites, fractions):
    pytest.importorskip("burnman")
    from burnman.optimize.composition_fitting import (
        DummyCompositionSolution as Reference,
    )

    actual = DummyCompositionSolution(elements, sites)
    expected = Reference(elements, sites)
    assert actual.elements == expected.elements
    assert actual.endmember_formulae == expected.endmember_formulae
    assert_allclose(
        actual.stoichiometric_matrix,
        np.array(expected.stoichiometric_matrix, dtype=float),
    )
    model = actual.solution_model
    reference_model = expected.solution_model
    for name in (
        "n_sites",
        "n_occupancies",
        "sites",
        "site_names",
        "site_formulae",
        "empty_formula",
        "general_formula",
    ):
        assert getattr(model, name) == getattr(reference_model, name)
    for name in (
        "endmember_occupancies",
        "endmember_noccupancies",
        "site_multiplicities",
    ):
        assert_allclose(getattr(model, name), getattr(reference_model, name))
    actual.set_composition(fractions)
    expected.set_composition(fractions)
    for element, amount in expected.formula.items():
        assert_allclose(actual.formula[element], amount)
    assert_allclose(actual.site_occupancies, expected.site_occupancies)
    assert_allclose(actual.site_occupancies, actual.get_site_occupancies())
    for precision in (0, 2, 4):
        for print_absent in (False, True):
            assert actual.site_formula(
                precision, print_absent
            ) == expected.site_formula(precision, print_absent)
    # Explicit occupancies need not be the solution's current composition.
    pure = model.endmember_occupancies[0]
    assert actual.site_formula(3, False, pure) == expected.site_formula(3, False, pure)


def test_exact_mixture_and_unnormalized_phase_amounts():
    assert (
        bm.optimize.composition_fitting.fit_phase_proportions_to_bulk_composition is fit
    )
    phases = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    bulk = np.array([0.25, 0.75, 1.0])
    amounts, covariance, residual = fit(phases, bulk)
    assert_allclose(amounts, [0.25, 0.75], rtol=1e-13)
    assert_allclose(phases @ amounts, bulk, rtol=1e-13)
    assert_allclose(covariance, [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], rtol=1e-13)
    assert residual < 1e-25
    scaled_amounts, scaled_covariance, scaled_residual = fit(phases, 4 * bulk)
    assert_allclose(scaled_amounts, [1, 3], rtol=1e-13)
    assert_allclose(scaled_covariance, covariance, rtol=1e-13)
    assert scaled_residual < 1e-25


def test_inexact_bulk_reaches_a_nonnegative_boundary():
    phases = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    bulk = np.array([-1, 2, 0], dtype=float)
    amounts, covariance, residual = fit(phases, bulk)
    assert_allclose(amounts, [0, 1], atol=1e-10)
    assert_allclose(covariance, [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], rtol=1e-13)
    assert_allclose(residual, 3.0, rtol=1e-12)
    zero, _, zero_residual = fit(phases, np.zeros(3))
    assert_allclose(zero, [0, 0], atol=0)
    assert zero_residual == 0


@pytest.mark.parametrize(
    "phases,bulk",
    [
        (np.empty((3, 0)), np.ones(3)),
        (np.eye(2), [1]),
        (np.eye(2), [1, np.nan]),
        ([[1, 0], [np.inf, 1]], [1, 2]),
        ([[1, 1], [2, 2]], [1, 2]),
    ],
)
def test_invalid_or_underdetermined_compositions_raise(phases, bulk):
    with pytest.raises(ValueError):
        fit(phases, bulk)


def test_all_mars_mantle_experiments_match_independent_nonnegative_fits():
    data = np.genfromtxt(DATA_FILE, names=True, dtype=None, encoding="utf8")
    compositions = np.array([data[name] for name in data.dtype.names[:6]])
    bulk = compositions[:, 0]
    for sample in np.unique(data["sample"]):
        if sample == "bulk_composition":
            continue
        phases = compositions[:, data["sample"] == sample]
        amounts, covariance, residual = fit(phases, bulk)
        expected, residual_norm = nnls(phases, bulk)
        assert_allclose(amounts, expected, atol=1e-7, rtol=1e-6, err_msg=sample)
        assert amounts.min() >= -1e-10, sample
        assert_allclose(
            residual, residual_norm**2, atol=1e-10, rtol=1e-9, err_msg=sample
        )
        inverse = np.linalg.pinv(phases)
        assert_allclose(
            covariance, inverse @ inverse.T, rtol=1e-8, atol=1e-9, err_msg=sample
        )


def test_optional_python_burnman_parity_and_data_provenance():
    burnman = pytest.importorskip("burnman")
    pytest.importorskip("cvxpy")
    from burnman.optimize.composition_fitting import (
        fit_phase_proportions_to_bulk_composition,
    )

    source = Path(burnman.__file__).parent / "data/input_fitting" / DATA_FILE.name
    assert DATA_FILE.read_bytes() == source.read_bytes()
    data = np.genfromtxt(DATA_FILE, names=True, dtype=None, encoding="utf8")
    compositions = np.array([data[name] for name in data.dtype.names[:6]])
    bulk = compositions[:, 0]
    for sample in np.unique(data["sample"]):
        if sample == "bulk_composition":
            continue
        phases = compositions[:, data["sample"] == sample]
        actual = fit(phases, bulk)
        expected = fit_phase_proportions_to_bulk_composition(phases, bulk)
        assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-7, err_msg=sample)
        assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-9, err_msg=sample)
        assert_allclose(actual[2], expected[2], rtol=1e-7, atol=1e-10, err_msg=sample)
