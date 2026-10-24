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

"""Reaction covariance propagation and estimation of equilibration conditions."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp import minerals
from burnman_cpp.tools import thermobarometry as thermo
from conftest import oxide_params


def polymorphs():
    assemblage = bm.Assemblage([minerals.HP11.sill(), minerals.HP11.ky()], [0.5, 0.5])
    assemblage.set_state(1e8, 800)
    return assemblage


def olivine_wadsleyite():
    ol, wad = minerals.SLB24.olivine(), minerals.SLB24.wadsleyite()
    ol.set_composition([0.9, 0.1])
    wad.set_composition([0.82, 0.18])
    ol.compositional_covariances = np.eye(2) * 1e-10
    wad.compositional_covariances = np.eye(2) * 1e-4
    assemblage = bm.Assemblage([ol, wad], [0.5, 0.5])
    assemblage.set_state(14e9, 1400)
    assemblage.state_priors = [13e9, 1400]
    assemblage.state_inverse_covariances = np.diag([1e-18, 1e-4])
    data = {
        "endmember_names": [
            name for phase in assemblage.phases for name in phase.endmember_names
        ],
        "covariance_matrix": np.eye(4) * 70**2,
    }
    return assemblage, data


def metapelite():
    dataset = minerals.mp50MnNCKFMASHTO
    phases = [
        dataset.mu(),
        dataset.bi(),
        dataset.g(),
        dataset.ilmm(),
        dataset.st(),
        minerals.HP11.q(),
    ]
    compositions = [
        [0.55510672, 0.00393874, 0.00365819, 0.43232719, 0.00339884, 0.00157032],
        [
            0.12024528,
            0.55795845,
            -0.23022757,
            0.2928378,
            0.19457943,
            0.06260454,
            0.00200206,
        ],
        [0.11996805, 0.75282947, 0.10051702, 0.02013999, 0.00654547],
        [-0.05461279, 0.84687573, 0.1062428, 0.05404236, 0.04745189],
        [0.06925327, 0.80706976, 0.01346876, 0.03074615, 0.07946205],
    ]
    deviations = [
        [0.01, 0.001, 0.001, 0.01, 0.001, 0.0005],
        [0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.0005],
        [0.01, 0.01, 0.01, 0.01, 0.001],
        [0.01] * 5,
        [0.01] * 5,
    ]
    for phase, composition, deviation in zip(phases, compositions, deviations):
        phase.set_composition(np.array(composition) / np.sum(composition))
        phase.compositional_covariances = np.diag(np.square(deviation))
    assemblage = bm.Assemblage(phases, [1 / 6] * 6)
    assemblage.set_state(0.4e9, 873)
    return assemblage


def test_reactions_covariance_and_prior_misfit():
    assemblage = polymorphs()
    R = thermo.get_reaction_matrix(assemblage)
    assert_allclose(R, [[-1, 1]], atol=1e-12)
    assert_allclose(R @ assemblage.stoichiometric_matrix, 0, atol=1e-12)
    data = minerals.HP11.cov()
    indices = [
        data["endmember_names"].index(name) for name in assemblage.endmember_names
    ]
    expected = R @ data["covariance_matrix"][np.ix_(indices, indices)] @ R.T
    actual = thermo.assemblage_affinity_covariance_matrix(assemblage, R, data)
    assert_allclose(actual, expected, rtol=1e-12)
    affinity = R @ assemblage.endmember_partial_gibbs
    assert_allclose(
        thermo.assemblage_affinity_misfit(assemblage, R, data),
        affinity @ np.linalg.pinv(actual) @ affinity,
    )
    assemblage.state_covariances = np.diag([1e16, 100])
    jacobian = np.array(
        [assemblage.endmember_partial_volumes, -assemblage.endmember_partial_entropies]
    ).T
    with_state = thermo.assemblage_affinity_covariance_matrix(assemblage, R, data, True)
    assert_allclose(
        with_state,
        actual + R @ jacobian @ assemblage.state_covariances @ jacobian.T @ R.T,
        rtol=1e-12,
    )
    assemblage.state_priors = [2e8, 810]
    assemblage.state_inverse_covariances = np.diag([1e-16, 0.01])
    assert_allclose(thermo.assemblage_state_misfit(assemblage), 2)


def test_optional_attributes_and_independent_copies():
    assemblage, _ = olivine_wadsleyite()
    phase = assemblage.phases[0]
    assert not hasattr(phase, "free_compositional_vectors")
    assert not hasattr(assemblage, "state_covariances")
    phase.free_compositional_vectors = [[-1, 1]]
    phase.baseline_composition = phase.molar_fractions
    view = phase.compositional_covariances
    view[:] = 0
    assert_allclose(phase.compositional_covariances, np.eye(2) * 1e-10)
    copied = assemblage.copy()
    copied.phases[0].compositional_covariances = np.eye(2)
    assert_allclose(phase.compositional_covariances, np.eye(2) * 1e-10)
    thermo.assemblage_set_state_from_params(assemblage, [13e9, 1200, 0.01])
    assert_allclose(phase.molar_fractions, [0.89, 0.11])
    thermo.assemblage_set_state_from_params(assemblage, [12e9, 1100, 0.02])
    assert_allclose(phase.molar_fractions, [0.88, 0.12])
    phase.free_compositional_vectors = None
    assert not hasattr(phase, "free_compositional_vectors")


def test_polymorph_temperature_at_fixed_pressure():
    assemblage = polymorphs()
    with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
        result = thermo.estimate_conditions(
            assemblage,
            minerals.HP11.cov(),
            guessed_conditions=[0.5e9, 500],
            pressure_bounds=[1e8, 1e8],
        )
    assert result.success, result.message
    assert result.x[0] == 1e8
    assert 600 < result.x[1] < 900
    assert result.fun < 1e-9
    assert result.n_params == result.n_reactions == 1
    assert result.degrees_of_freedom == 0
    assert result.xcov.shape == (2, 2)
    assert result.dadx.shape == (1, 2)
    assert not hasattr(result, "reduced_chisqr")
    assert not hasattr(result, "fit")
    assert_allclose([assemblage.pressure, assemblage.temperature], result.x)
    assert_allclose(assemblage.reaction_matrix_for_optimization, [[-1, 1]])


def test_estimate_olivine_wadsleyite_with_priors():
    assemblage, data = olivine_wadsleyite()
    result = thermo.estimate_conditions(assemblage, data, True, [14e9, 1400])
    assert result.success, result.message
    assert 12e9 < result.x[0] < 16e9
    assert 1000 < result.x[1] < 1800
    assert result.n_params == result.n_reactions == result.n_constraints == 2
    assert result.degrees_of_freedom == 2
    assert_allclose(result.fit**2, result.reduced_chisqr)
    assert_allclose(
        result.fun,
        thermo.assemblage_affinity_misfit(
            assemblage, thermo.get_reaction_matrix(assemblage), data
        )
        + thermo.assemblage_state_misfit(assemblage),
    )


def test_free_compositional_coordinate_keeps_pt_only_uncertainties():
    endmembers = []
    for name, shift, site in [("a", 0, "[Mg][Fe]O2"), ("b", 1000, "[Fe][Mg]O2")]:
        params = oxide_params(energy_shift=shift)
        params.update(
            name=name,
            formula={"Mg": 1.0, "Fe": 1.0, "O": 2.0},
            molar_mass=0.1121488,
            n=4,
        )
        endmembers.append((bm.Mineral(params), site))
    phase = bm.Solution(bm.IdealSolution(endmembers), [0.65, 0.35])
    phase.compositional_covariances = np.eye(2) * 1e-4
    phase.free_compositional_vectors = [[-1, 1]]
    assemblage = bm.Assemblage([phase], [1])
    with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
        result = thermo.estimate_conditions(
            assemblage,
            guessed_conditions=[1e9, 1000],
            pressure_bounds=[1e9, 1e9],
            temperature_bounds=[1000, 1000],
            max_it=300,
        )
    assert result.success, result.message
    expected_b = 1 / (1 + np.exp(1000 / (2 * bm.constants.gas_constant * 1000)))
    assert_allclose(phase.molar_fractions, [1 - expected_b, expected_b], atol=1e-6)
    assert_allclose(result.x[2], expected_b - 0.35, atol=1e-6)
    assert result.x.shape == (3,)
    assert result.xcov.shape == (2, 2)
    assert result.dadx.shape == (1, 2)
    assert result.var.shape == (2,)
    assert result.n_params == 1
    assert_allclose(phase.baseline_composition, [0.65, 0.35])


@pytest.mark.parametrize(
    "tolerance,reactions,expected",
    [
        (0, 18, [410.965848, 866.918348]),
        (0.01, 11, [425.624983, 876.969462]),
    ],
)
def test_metapelite_reaction_basis_and_conditions(tolerance, reactions, expected):
    assemblage = metapelite()
    R = thermo.get_reaction_matrix(assemblage, tolerance)
    assert R.shape[0] == reactions
    assert_allclose(R @ assemblage.stoichiometric_matrix, 0, atol=1e-10)
    result = thermo.estimate_conditions(
        assemblage,
        minerals.HP11.cov(),
        guessed_conditions=[0.5e9, 500],
        small_fraction_tol=tolerance,
        max_it=300,
    )
    assert result.success, result.message
    assert result.n_reactions == reactions
    # Expected estimates are from the pinned Python BurnMan calculation.
    assert_allclose([result.x[0] / 1e6, result.x[1]], expected, atol=0.2)


@pytest.mark.parametrize("with_state", [False, True])
def test_optional_python_covariance_and_misfit_parity(with_state):
    burnman = pytest.importorskip("burnman")
    from burnman.tools import thermobarometry as reference

    actual = metapelite()
    reference_phases = [
        burnman.minerals.mp50MnNCKFMASHTO.mu(),
        burnman.minerals.mp50MnNCKFMASHTO.bi(),
        burnman.minerals.mp50MnNCKFMASHTO.g(),
        burnman.minerals.mp50MnNCKFMASHTO.ilmm(),
        burnman.minerals.mp50MnNCKFMASHTO.st(),
        burnman.minerals.HP_2011_ds62.q(),
    ]
    for native, python in zip(actual.phases[:-1], reference_phases[:-1]):
        python.set_composition(native.molar_fractions)
        python.compositional_covariances = native.compositional_covariances
    expected = burnman.Composite(reference_phases, [1 / 6] * 6)
    expected.set_state(actual.pressure, actual.temperature)
    if with_state:
        actual.state_covariances = np.diag([1e16, 100])
        expected.state_covariances = actual.state_covariances
    # Supply the same independent reactions to compare covariance entries directly.
    R = thermo.get_reaction_matrix(actual)
    data = minerals.HP11.cov()
    assert_allclose(
        thermo.assemblage_affinity_covariance_matrix(actual, R, data, with_state),
        reference.assemblage_affinity_covariance_matrix(expected, R, data, with_state),
        rtol=1e-6,
        atol=1e-5,
    )
    assert_allclose(
        thermo.assemblage_affinity_misfit(actual, R, data, with_state),
        reference.assemblage_affinity_misfit(expected, R, data, with_state),
        rtol=1e-5,
        atol=1e-12,
    )


@pytest.mark.parametrize("pressure_scaling", [1e6, 1e7])
def test_optional_python_estimated_conditions_and_statistics(pressure_scaling):
    burnman = pytest.importorskip("burnman")
    from scipy.linalg import sqrtm
    from burnman.tools import thermobarometry as reference

    actual, data = olivine_wadsleyite()
    ol, wad = (
        burnman.minerals.SLB_2024.olivine(),
        burnman.minerals.SLB_2024.wadsleyite(),
    )
    for native, phase in zip(actual.phases, [ol, wad]):
        phase.set_composition(native.molar_fractions)
        phase.compositional_covariances = native.compositional_covariances
    expected = burnman.Composite([ol, wad], [0.5, 0.5])
    expected.state_priors = actual.state_priors
    expected.state_inverse_covariances = actual.state_inverse_covariances
    native_result = thermo.estimate_conditions(
        actual, data, True, [14e9, 1400], P_scaling=pressure_scaling
    )
    python_result = reference.estimate_conditions(
        expected, data, True, [14e9, 1400], P_scaling=pressure_scaling
    )
    assert native_result.success and python_result.success
    # Different finite-difference gradients and stopping criteria shift the
    # NLopt and SciPy estimates slightly along the correlated P-T uncertainty.
    # Allow 2% of the joint one-sigma uncertainty, retaining the independent
    # misfit and covariance checks below.
    condition_error = np.linalg.solve(
        np.linalg.cholesky(python_result.xcov), native_result.x - python_result.x
    )
    assert np.linalg.norm(condition_error) < 0.02

    # Compare state-dependent statistics at identical conditions; small shifts
    # in fitted P and T can change individual reaction residuals appreciably.
    expected.set_state(*native_result.x)
    R = actual.reaction_matrix_for_optimization
    assert_allclose(
        native_result.fun,
        reference.assemblage_affinity_misfit(expected, R, data)
        + reference.assemblage_state_misfit(expected),
        rtol=1e-8,
        atol=1e-10,
    )
    covariance = reference.assemblage_affinity_covariance_matrix(expected, R, data)
    dadx = (
        R
        @ np.array(
            [expected.endmember_partial_volumes, -expected.endmember_partial_entropies]
        ).T
    )
    xcov = np.linalg.pinv(
        dadx.T @ np.linalg.pinv(covariance) @ dadx + expected.state_inverse_covariances
    )
    var = np.sqrt(np.diag(xcov))
    affinities = R @ expected.endmember_partial_gibbs
    statistics = {
        "acov": covariance,
        "dadx": dadx,
        "xcov": xcov,
        "var": var,
        "xcorr": xcov / np.outer(var, var),
        "affinities": affinities,
        "weighted_affinities": np.linalg.pinv(sqrtm(covariance)) @ affinities,
    }
    for field, value in statistics.items():
        assert_allclose(
            getattr(native_result, field), value, rtol=1e-6, atol=1e-10, err_msg=field
        )

    for field in ["fun", "reduced_chisqr", "fit"]:
        assert_allclose(
            getattr(native_result, field),
            getattr(python_result, field),
            rtol=5e-3,
            atol=1e-8,
            err_msg=field,
        )


def test_optional_python_metapelite_free_composition_covariance_parity():
    burnman = pytest.importorskip("burnman")
    from burnman.tools import thermobarometry as reference

    actual = metapelite()
    dataset = burnman.minerals.mp50MnNCKFMASHTO
    phases = [
        dataset.mu(),
        dataset.bi(),
        dataset.g(),
        dataset.ilmm(),
        dataset.st(),
        burnman.minerals.HP_2011_ds62.q(),
    ]
    for native, phase in zip(actual.phases[:-1], phases[:-1]):
        phase.set_composition(native.molar_fractions)
        phase.compositional_covariances = native.compositional_covariances
        if len(native.reaction_basis):
            native.free_compositional_vectors = native.reaction_basis
            phase.free_compositional_vectors = native.reaction_basis
            phase.baseline_composition = native.molar_fractions.copy()
    expected = burnman.Composite(phases, [1 / 6] * 6)
    expected.set_state(0.4e9, 873)
    data = minerals.HP11.cov()
    native_result = thermo.estimate_conditions(
        actual, data, guessed_conditions=[0.5e9, 500]
    )
    assert native_result.success
    assert native_result.x.shape == (4,)
    assert native_result.n_params == 4
    assert native_result.xcov.shape == (2, 2)
    assert native_result.dadx.shape[1] == 2
    for phase in actual.phases[:-1]:
        assert phase.site_occupancies.min() >= -1e-10
    # Evaluate both uncertainty calculations at the same physical compositions.
    reference.assemblage_set_state_from_params(expected, native_result.x)
    R = actual.reaction_matrix_for_optimization
    covariance = reference.assemblage_affinity_covariance_matrix(expected, R, data)
    dadx = (
        R
        @ np.array(
            [expected.endmember_partial_volumes, -expected.endmember_partial_entropies]
        ).T
    )
    xcov = np.linalg.pinv(dadx.T @ np.linalg.pinv(covariance) @ dadx)
    assert_allclose(
        native_result.fun,
        reference.assemblage_affinity_misfit(expected, R, data),
        rtol=1e-5,
    )
    assert_allclose(native_result.acov, covariance, rtol=1e-6, atol=1e-5)
    assert_allclose(native_result.dadx, dadx, rtol=1e-6, atol=1e-8)
    assert_allclose(native_result.xcov, xcov, rtol=5e-3, atol=1e-8)


def test_invalid_data_and_unsupported_inversions_raise():
    assemblage = polymorphs()
    R = thermo.get_reaction_matrix(assemblage)
    with pytest.raises(ValueError):
        thermo.estimate_conditions(assemblage)
    with pytest.raises(ValueError):
        thermo.estimate_conditions(
            assemblage, pressure_bounds=[1e8, 1e8], temperature_bounds=[800, 800]
        )
    with pytest.raises(ValueError):
        thermo.assemblage_affinity_covariance_matrix(
            assemblage, R, include_state_uncertainties=True
        )
    with pytest.raises(ValueError):
        thermo.assemblage_state_misfit(assemblage)
    for data in [
        {"endmember_names": ["missing"], "covariance_matrix": [[1]]},
        {
            "endmember_names": assemblage.endmember_names,
            "covariance_matrix": [[1, 2], [2, 1]],
        },
    ]:
        with pytest.raises(ValueError):
            thermo.assemblage_affinity_covariance_matrix(assemblage, R, data)
    phase = minerals.SLB24.olivine()
    mixture = bm.Assemblage([phase, minerals.SLB24.wadsleyite()], [0.5, 0.5])
    mixture.set_state(13e9, 1400)
    with pytest.raises(ValueError, match="compositional_covariances"):
        thermo.assemblage_affinity_covariance_matrix(
            mixture, thermo.get_reaction_matrix(mixture)
        )
    phase.free_compositional_vectors = [[-1, 1]]
    phase.baseline_composition = phase.molar_fractions
    before = phase.molar_fractions
    with pytest.raises(ValueError):
        thermo.assemblage_set_state_from_params(mixture, [13e9, 1400, 2])
    assert_allclose(phase.molar_fractions, before)
