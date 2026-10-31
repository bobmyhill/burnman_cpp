# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Subregular cubic interactions, partial molar derivatives and reference parity."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011


def endmembers(n=3, reference=False):
    minerals = (
        pytest.importorskip("burnman.minerals.SLB_2011") if reference else SLB_2011
    )
    return [
        (getattr(minerals, name)(), formula)
        for name, formula in (
            ("pyrope", "[Mg]3[Al]2Si3O12"),
            ("almandine", "[Fe]3[Al]2Si3O12"),
            ("mg_majorite", "[Mg]3[Mg1/2Si1/2]2Si3O12"),
            ("grossular", "[Ca]3[Al]2Si3O12"),
        )[:n]
    ]


def parameters(n=3, ternary=True):
    energy = [
        [[10000.0, -10000.0], [5000.0, 3000.0], [-2000.0, 7000.0]],
        [[-10000.0, 6000.0], [4000.0, -5000.0]],
        [[6000.0, 9000.0]],
    ]
    terms = [[0, 1, 2, 3000.0], [2, 0, 1, 700.0], [0, 1, 2, -200.0]]
    if n == 4:
        terms.append([1, 2, 3, -4000.0])
    if not ternary or n < 3:
        terms = []
    return dict(
        energy_interaction=[energy[i][: n - i - 1] for i in range(n - 1)],
        volume_interaction=[
            [[(i + 1) * 2e-7, -(j + 1) * 1e-7] for j in range(i + 1, n)]
            for i in range(n - 1)
        ],
        entropy_interaction=[
            [[(i + 1) * 0.2, -(j + 1) * 0.3] for j in range(i + 1, n)]
            for i in range(n - 1)
        ],
        energy_ternary_terms=terms,
        volume_ternary_terms=[[i, j, k, v * 3e-11] for i, j, k, v in terms],
        entropy_ternary_terms=[[i, j, k, v * 2e-4] for i, j, k, v in terms],
    )


def make_model(n=3, ternary=True, reference=False):
    models = pytest.importorskip("burnman.classes.solutionmodel") if reference else bm
    return models.SubregularSolution(endmembers(n, reference), **parameters(n, ternary))


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("ternary", [False, True])
@pytest.mark.parametrize("composition", ["interior", "edge", "pure"])
def test_python_reference(n, ternary, composition):
    native = make_model(n, ternary)
    reference = make_model(n, ternary, reference=True)
    for name in ("Wijke", "Wijks", "Wijkv"):
        assert_array_equal(getattr(native, name), getattr(reference, name))
    p = np.arange(1.0, n + 1)
    if composition == "edge":
        p[0] = 0.0
    elif composition == "pure":
        p = np.eye(n)[0]
    p /= p.sum()
    names = {
        "excess_gibbs_free_energy": "excess_gibbs_energy",
        "excess_entropy": "excess_entropy",
        "excess_volume": "excess_volume",
        "excess_enthalpy": "excess_enthalpy",
        "excess_partial_gibbs_free_energies": "excess_partial_gibbs_energies",
        "excess_partial_entropies": "excess_partial_entropies",
        "excess_partial_volumes": "excess_partial_volumes",
        "gibbs_hessian": "gibbs_hessian",
        "entropy_hessian": "entropy_hessian",
        "volume_hessian": "volume_hessian",
        "activities": "activities",
        "activity_coefficients": "activity_coefficients",
    }
    for P, T in ((1e5, 300.0), (8e9, 1600.0)):
        for actual, expected in names.items():
            # Absent-species ideal chemical potentials and curvatures use
            # different existing logarithm cutoffs in the native and Python
            # IdealSolution classes. Compare those quantities in the interior.
            if composition != "interior" and actual in (
                "excess_partial_gibbs_free_energies",
                "excess_partial_entropies",
                "gibbs_hessian",
                "entropy_hessian",
            ):
                continue
            assert_allclose(
                getattr(native, "compute_" + actual)(P, T, p),
                getattr(reference, expected)(P, T, p),
                rtol=2e-10,
                atol=1e-12,
                err_msg=actual,
            )
    # At T=0 the Gibbs quantities isolate the finite cubic E+PV contribution,
    # including at endpoints where ideal chemical potentials are singular.
    for actual, expected in (
        ("excess_partial_gibbs_free_energies", "excess_partial_gibbs_energies"),
        ("gibbs_hessian", "gibbs_hessian"),
    ):
        assert_allclose(
            getattr(native, "compute_" + actual)(8e9, 0.0, p),
            getattr(reference, expected)(8e9, 0.0, p),
            rtol=2e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_binary_parameters_include_induced_multicomponent_terms(n):
    params = parameters(n)
    model = make_model(n)
    p = np.arange(1.0, n + 1)
    p /= p.sum()
    expected = sum(
        p[i] * p[j] * (w0 * (1 + p[j] - p[i]) + w1 * (1 + p[i] - p[j])) / 2
        for i, row in enumerate(params["energy_interaction"])
        for j, (w0, w1) in enumerate(row, start=i + 1)
    ) + sum(v * p[i] * p[j] * p[k] for i, j, k, v in params["energy_ternary_terms"])
    assert_allclose(model.compute_excess_gibbs_free_energy(0.0, 0.0, p), expected)


@pytest.mark.parametrize("property_name", ["gibbs", "entropy", "volume"])
def test_partials_and_hessians_differentiate_endmember_amounts(property_name):
    model = make_model(4)
    p = np.array([0.15, 0.2, 0.3, 0.35])
    scalar_name, partial_name = {
        "gibbs": ("gibbs_free_energy", "gibbs_free_energies"),
        "entropy": ("entropy", "entropies"),
        "volume": ("volume", "volumes"),
    }[property_name]
    scalar = getattr(model, "compute_excess_" + scalar_name)
    partial = getattr(model, "compute_excess_partial_" + partial_name)
    P, T, step = 3e9, 1200.0, 1e-6
    mu = partial(P, T, p)
    h = getattr(model, f"compute_{property_name}_hessian")(P, T, p)
    assert_allclose(h, h.T, rtol=1e-13, atol=1e-12)
    assert np.linalg.norm(h @ p) < 1e-13 * np.linalg.norm(h)
    for i in range(4):
        plus, minus = p.copy(), p.copy()
        plus[i] += step
        minus[i] -= step
        nplus, nminus = plus.sum(), minus.sum()
        plus /= nplus
        minus /= nminus
        assert_allclose(
            (nplus * scalar(P, T, plus) - nminus * scalar(P, T, minus)) / (2 * step),
            mu[i],
            rtol=2e-7,
        )
        assert_allclose(
            (partial(P, T, plus) - partial(P, T, minus)) / (2 * step),
            h[:, i],
            rtol=2e-7,
            atol=np.linalg.norm(h) * 1e-9,
        )


def test_symmetric_limit_and_default_optional_interactions():
    members = endmembers()
    regular = bm.SymmetricRegularSolution(members, [[10000.0, 5000.0], [-10000.0]])
    subregular = bm.SubregularSolution(
        members,
        [[[10000.0, 10000.0], [5000.0, 5000.0]], [[-10000.0, -10000.0]]],
        volume_interaction=None,
        entropy_interaction=None,
    )
    p, P, T = [0.2, 0.3, 0.5], 1e9, 1200.0
    for name in (
        "excess_partial_gibbs_free_energies",
        "excess_partial_entropies",
        "excess_partial_volumes",
        "gibbs_hessian",
        "entropy_hessian",
        "volume_hessian",
        "activities",
    ):
        assert_allclose(
            getattr(subregular, "compute_" + name)(P, T, p),
            getattr(regular, "compute_" + name)(P, T, p),
            atol=1e-12,
        )
    for name in ("Cp_excess", "alphaV_excess", "VoverKT_excess"):
        assert getattr(subregular, "compute_" + name)(P, T, p) == 0.0
    for T in (0.0, 1e-10):
        with pytest.raises(ValueError, match="0 K"):
            subregular.compute_activity_coefficients(P, T, p)


def test_solution_thermodynamics_and_independent_copies():
    from burnman_cpp.tools.eos import check_eos_consistency

    model = make_model()
    solution = bm.Solution(model, [0.2, 0.3, 0.5])
    assert check_eos_consistency(
        solution, 3e9, 1200.0, including_shear_properties=False
    )
    copy = solution.copy()
    clone = model.clone()
    assert isinstance(copy.solution_model, bm.SubregularSolution)
    assert isinstance(clone, bm.SubregularSolution)
    solution.set_state(3e9, 1200.0)
    g = solution.molar_gibbs
    copy.set_state(1e9, 500.0)
    copy.set_composition([0.6, 0.3, 0.1])
    assert solution.molar_gibbs == g
    assert not model.endmembers[0][0].has_state()
    # Tensor properties are copies, like the other cached native arrays.
    tensor = clone.Wijke
    tensor[:] = 0
    assert_array_equal(model.Wijke, clone.Wijke)


def test_reference_solution_adapter():
    burnman = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    reference = burnman.Solution(
        solution_model=make_model(4, reference=True),
        molar_fractions=[0.15, 0.2, 0.3, 0.35],
    )
    native = from_burnman(reference)
    assert isinstance(native.solution_model, bm.SubregularSolution)
    for name in ("Wijke", "Wijks", "Wijkv"):
        assert_array_equal(
            getattr(native.solution_model, name),
            getattr(reference.solution_model, name),
        )
    for phase in (reference, native):
        phase.set_state(3e9, 1200.0)
    for name in (
        "molar_gibbs",
        "molar_entropy",
        "molar_volume",
        "molar_heat_capacity_p",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
        "partial_gibbs",
        "gibbs_hessian",
        "entropy_hessian",
        "volume_hessian",
        "activities",
    ):
        assert_allclose(getattr(native, name), getattr(reference, name), rtol=2e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_interaction": []},
        {"energy_interaction": [[[1.0, 2.0]]]},
        {"energy_interaction": [[[1.0, np.nan], [1.0, 2.0]], [[1.0, 2.0]]]},
        {"volume_interaction": [[[1.0, np.inf], [1.0, 2.0]], [[1.0, 2.0]]]},
        {"entropy_interaction": [[[1.0, 2.0]]]},
        {"energy_ternary_terms": [[0, 1, 3, 1000.0]]},
        {"volume_ternary_terms": [[0, -1, 2, 1e-6]]},
        {"entropy_ternary_terms": [[0, 0.5, 2, 1.0]]},
        {"energy_ternary_terms": [[0, 1, 2, np.inf]]},
    ],
)
def test_invalid_parameters(kwargs):
    params = parameters()
    params.update(kwargs)
    with pytest.raises(ValueError):
        bm.SubregularSolution(endmembers(), **params)


def test_single_endmember_and_repeated_ternary_indices():
    model = bm.SubregularSolution(
        endmembers(1), [], energy_ternary_terms=[[0, 0, 0, 1500.0]]
    )
    assert model.compute_excess_gibbs_free_energy(1e9, 0.0, [1.0]) == 1500.0
    assert_array_equal(model.compute_gibbs_hessian(1e9, 300.0, [1.0]), [[0.0]])
