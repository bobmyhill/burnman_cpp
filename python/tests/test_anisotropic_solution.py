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

"""Anisotropic mixing, reference parity and composition/strain derivatives."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011
from burnman_cpp.classes.anisotropicsolution import AnisotropicSolution
from test_anisotropic_mineral import model_parameters, python_reference


def endmember_data(system):
    reference, cell, c = model_parameters(system)
    second = SLB_2011.fayalite()
    second_cell = cell.copy()
    second_cell[:3] *= [1.05, 1.1, 0.95]
    second_cell[3:] = {
        "orthorhombic": [90.0, 90.0, 90.0],
        "monoclinic": [90.0, 104.0, 90.0],
        "triclinic": [88.0, 83.0, 91.0],
    }[system]
    cosine = np.cos(np.deg2rad(second_cell[3:]))
    volume = np.prod(second_cell[:3]) * np.sqrt(
        1.0 + 2.0 * np.prod(cosine) - np.sum(cosine**2)
    )
    second_cell[:3] *= np.cbrt(second.params.V_0 / volume)
    second_c = c.copy()
    second_c[:3, :3, 1, 0] += np.diag([0.03, -0.02, -0.01])
    second_c[3:, 3:, 1, 0] *= 1.08
    return [
        (reference, cell, c, "[Mg]2SiO4"),
        (second, second_cell, second_c, "[Fe]2SiO4"),
    ]


def excess_psi(lnV, Pth, fractions, params):
    direction = np.zeros((6, 6))
    direction[0, 0], direction[1, 1] = 1.0, -1.0
    coefficient = params.get("coefficient", 0.0)
    amplitude = coefficient * (lnV - np.log(4.5e-5))
    fraction = fractions[0] * fractions[1]
    gradient = np.zeros((3, 3, len(fractions)))
    gradient[:, :, 0] = np.diag([1.0, -1.0, 0.0]) * amplitude * fractions[1]
    gradient[:, :, 1] = np.diag([1.0, -1.0, 0.0]) * amplitude * fractions[0]
    return (
        fraction * amplitude * direction,
        fraction * coefficient * direction,
        np.zeros((6, 6)),
        gradient,
    )


def make_solution(
    system="triclinic", excess=True, fractions=(0.8, 0.2), kind="symmetric"
):
    data = endmember_data(system)
    endmembers = [
        (bm.AnisotropicMineral(reference, cell, c, [0, 1, 2]), formula)
        for reference, cell, c, formula in data
    ]
    if kind == "ideal":
        model = bm.IdealSolution(endmembers)
    elif kind == "asymmetric":
        model = bm.AsymmetricRegularSolution(
            endmembers, [1.0, 1.3], [[10000.0]], [[1.0e-6]]
        )
    else:
        model = bm.SymmetricRegularSolution(endmembers, [[10000.0]], [[1.0e-6]])
    return AnisotropicSolution(
        "olivine",
        model,
        excess_psi,
        {"coefficient": 0.025 if excess else 0.0},
        fractions,
    )


@pytest.mark.parametrize("system", ["orthorhombic", "monoclinic", "triclinic"])
@pytest.mark.parametrize("excess", [False, True])
def test_python_reference(system, excess, python_reference):
    module = pytest.importorskip("burnman.classes.anisotropicsolution")
    minerals = pytest.importorskip("burnman.minerals.SLB_2011")
    models = pytest.importorskip("burnman.classes.solutionmodel")
    data = endmember_data(system)
    original_endmembers = []
    for reference, (_, cell, c, formula) in zip(
        [minerals.forsterite(), minerals.fayalite()], data
    ):
        original_endmembers.append(
            (
                python_reference.AnisotropicMineral(reference, cell, c, [0, 1, 2]),
                formula,
            )
        )
    original_model = models.SymmetricRegularSolution(
        original_endmembers, [[10000.0]], volume_interaction=[[1.0e-6]]
    )
    original = module.AnisotropicSolution(
        "olivine",
        original_model,
        excess_psi,
        {"coefficient": 0.025 if excess else 0.0},
    )
    # The pinned Python constructor dispatches composition before initialization.
    original.set_composition([0.8, 0.2])
    native = make_solution(system, excess)
    properties = [
        "molar_volume",
        "molar_gibbs",
        "molar_entropy",
        "molar_heat_capacity_p",
        "density",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "cell_vectors_0",
        "deformation_gradient_tensor",
        "unrotated_cell_vectors",
        "rotation_matrix",
        "cell_vectors",
        "cell_parameters",
        "thermal_expansivity_tensor",
        "isothermal_compliance_tensor",
        "isentropic_compliance_tensor",
        "isothermal_stiffness_tensor",
        "isentropic_stiffness_tensor",
        "grueneisen_tensor",
        "thermal_stress_tensor",
        "molar_isometric_heat_capacity",
        "partial_gibbs",
        "partial_volumes",
        "gibbs_hessian",
        "_dM0dp_fixed_VT",
        "_unrotated_dFdp_fixed_VT",
        "_unrotated_dMdn_fixed_VT",
        "_dMdn_fixed_VT",
        "depsdn_fixed_VT",
        "_PsiI_mbr",
        "_dPsiIdp_xs",
    ]
    for pressure, temperature in [(1.0e9, 300.0), (20.0e9, 1000.0)]:
        native.set_state(pressure, temperature)
        original.set_state(pressure, temperature)
        for prop in properties:
            actual, expected = getattr(native, prop), getattr(original, prop)
            scale = max(float(np.max(np.abs(expected))), 1.0e-30)
            assert_allclose(
                np.asarray(actual) / scale,
                np.asarray(expected) / scale,
                rtol=3.0e-8,
                atol=2.0e-10,
                err_msg=prop,
            )
        assert_allclose(native.Pth, original.Pth, rtol=1.0e-8, atol=0.1)
        assert_allclose(native.dPthdf, original.dPthdf, rtol=1.0e-8, atol=0.1)
        assert_allclose(
            native.wave_velocities([1.0, 2.0, 3.0])[0],
            original.wave_velocities([1.0, 2.0, 3.0])[0],
            rtol=3.0e-8,
        )


@pytest.mark.parametrize("kind", ["ideal", "symmetric", "asymmetric"])
def test_model_endmembers_scalar_eos_and_pure_limits(kind):
    solution = make_solution(kind=kind)
    assert isinstance(solution, bm.Solution)
    assert isinstance(solution, bm._core.AnisotropicMineral)
    assert isinstance(solution, bm.AnisotropicMaterial)
    assert all(
        isinstance(m, bm._core.AnisotropicMineral)
        for m, _ in solution.solution_model.endmembers
    )
    scalar = bm.Solution(solution.solution_model, [0.8, 0.2])
    for material in [solution, scalar]:
        material.set_state(20.0e9, 1000.0)
    for prop in [
        "molar_volume",
        "molar_gibbs",
        "molar_entropy",
        "molar_mass",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "partial_volumes",
    ]:
        assert_allclose(getattr(solution, prop), getattr(scalar, prop))
    for i in range(2):
        fractions = np.eye(2)[i]
        solution.set_composition(fractions)
        endmember = solution.endmembers[i][0]
        assert_allclose(
            solution.cell_vectors, endmember.cell_vectors, rtol=1.0e-12, atol=1.0e-17
        )
        assert_allclose(
            solution.isentropic_stiffness_tensor,
            endmember.isentropic_stiffness_tensor,
            rtol=1.0e-10,
            atol=0.01,
        )


def test_composition_derivatives_and_frozen_crystal_frame():
    solution = make_solution()
    fractions = np.array([0.8, 0.2])
    solution.set_state(20.0e9, 1000.0)
    volume = solution.molar_volume
    rotation = solution.rotation_matrix
    M = solution.cell_vectors.T
    dM0 = solution._dM0dp_fixed_VT
    dF = solution._unrotated_dFdp_fixed_VT
    dMdn = solution._dMdn_fixed_VT
    depsdn = solution.depsdn_fixed_VT
    direction = np.array([1.0, -1.0])
    step = 1.0e-4
    values = []
    for sign in [1.0, -1.0]:
        solution.set_composition(fractions + sign * step * direction)
        solution.set_state_with_volume(volume, 1000.0)
        values.append((solution.cell_vectors_0.T, solution.deformation_gradient_tensor))
    for derivative, index in [(dM0, 0), (dF, 1)]:
        assert_allclose(
            np.einsum("ijk,k->ij", derivative, direction),
            (values[0][index] - values[1][index]) / (2 * step),
            rtol=1.0e-7,
            atol=1.0e-12,
        )
    step = 1.0e-5
    for direction in [np.array([1.0, -1.0]), np.array([1.0, 0.0])]:
        cells = []
        for sign in [1.0, -1.0]:
            amounts = fractions + sign * step * direction
            n = amounts.sum()
            solution.set_composition(amounts / n)
            solution.set_state_with_volume(volume / n, 1000.0)
            cells.append(rotation @ solution.unrotated_cell_vectors.T * np.cbrt(n))
        derivative = (cells[0] - cells[1]) / (2 * step)
        assert_allclose(
            np.einsum("ijk,k->ij", dMdn, direction),
            derivative,
            rtol=1.0e-6,
            atol=1.0e-11,
        )
        L = derivative @ np.linalg.inv(M)
        assert_allclose(
            np.einsum("ijk,k->ij", depsdn, direction),
            0.5 * (L + L.T),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
    assert_allclose(np.trace(depsdn, axis1=0, axis2=1), 0.0, atol=1.0e-12)


@pytest.mark.parametrize("system", ["orthorhombic", "monoclinic", "triclinic"])
def test_pressure_temperature_consistency(system):
    solution = make_solution(system)
    pressure, temperature, dp, dt = 20.0e9, 1000.0, 10000.0, 0.01
    tensors = []
    for p, t in [
        (pressure - dp, temperature),
        (pressure + dp, temperature),
        (pressure, temperature - dt),
        (pressure, temperature + dt),
    ]:
        solution.set_state(p, t)
        tensors.append(solution.deformation_gradient_tensor)
    solution.set_state(pressure, temperature)
    invF = np.linalg.inv(solution.deformation_gradient_tensor)
    r = solution.rotation_matrix
    lp = (tensors[1] - tensors[0]) / (2 * dp) @ invF
    lt = (tensors[3] - tensors[2]) / (2 * dt) @ invF
    assert_allclose(
        r @ (-0.5 * (lp + lp.T)) @ r.T,
        solution.isothermal_compressibility_tensor,
        rtol=1.0e-6,
        atol=1.0e-20,
    )
    assert_allclose(
        r @ (0.5 * (lt + lt.T)) @ r.T,
        solution.thermal_expansivity_tensor,
        rtol=1.0e-6,
        atol=1.0e-14,
    )
    assert_allclose(
        np.linalg.det(solution.cell_vectors), solution.molar_volume, rtol=1.0e-12
    )
    assert_allclose(
        np.trace(solution.thermal_expansivity_tensor),
        solution.thermal_expansivity,
        rtol=1.0e-12,
    )
    assert_allclose(
        solution.isothermal_compliance_tensor[:3, :3].sum(),
        solution.isothermal_compressibility_reuss,
        rtol=1.0e-12,
    )
    assert_allclose(
        solution.isentropic_compliance_tensor[:3, :3].sum(),
        solution.isentropic_compressibility_reuss,
        rtol=1.0e-12,
    )


def test_states_copies_evaluation_and_validation():
    solution = make_solution(fractions=None)
    with pytest.raises(RuntimeError, match="set_composition"):
        solution.set_state(1.0e9, 300.0)
    solution.set_composition([0.8, 0.2])
    assert not solution.has_state()
    expected = np.prod(
        [
            np.linalg.det(m.cell_vectors_0) ** x
            for (m, _), x in zip(solution.endmembers, [0.8, 0.2])
        ]
    )
    assert_allclose(np.linalg.det(solution.cell_vectors_0), expected, rtol=1.0e-12)
    values = solution.evaluate(
        ["molar_volume", "cell_vectors"],
        [1.0e9, 2.0e9],
        [300.0, 1000.0],
        [[0.8, 0.2], [0.3, 0.7]],
    )
    assert values[0].shape == (2,)
    assert values[1].shape == (2, 3, 3)
    assert not solution.has_state()
    assert_array_equal(solution.molar_fractions, [0.8, 0.2])
    solution.set_state(20.0e9, 1000.0)
    copy = solution.copy()
    assert copy.endmembers[0][0] is not solution.endmembers[0][0]
    copy.set_composition([0.3, 0.7])
    copy.set_state(1.0e9, 300.0)
    assert solution.pressure == 20.0e9
    assert solution.endmembers[0][0].pressure == 20.0e9
    stiffness = solution.isentropic_stiffness_tensor
    solution.reset()
    assert_allclose(solution.isentropic_stiffness_tensor, stiffness)
    solution.depsdn_fixed_VT[:] = 0.0
    assert np.any(solution.depsdn_fixed_VT)
    for fractions in [[0.2, 0.2], [-0.2, 1.2], [np.nan, 0.0], [1.0]]:
        with pytest.raises(ValueError):
            solution.set_composition(fractions)
    with pytest.raises(ValueError):
        solution.set_state_with_volume(-1.0, 300.0)
    normal = bm.IdealSolution([(SLB_2011.forsterite(), "[Mg]2SiO4")])
    with pytest.raises(ValueError, match="anisotropic endmembers"):
        bm.AnisotropicSolution("invalid", normal, excess_psi, {})
    with pytest.raises(ValueError, match="shape"):
        bad = bm.AnisotropicSolution(
            "invalid",
            solution.solution_model,
            lambda *args: (np.zeros((6, 6)),) * 3 + (np.zeros((2, 3, 3)),),
            {},
            [0.8, 0.2],
        )
        bad.set_state(1.0e9, 300.0)


def test_three_endmembers_and_duplicate_endmember_derivatives():
    source = make_solution()
    a, b = source.endmembers
    model = bm.SymmetricRegularSolution(
        [a, a, b], [[-1000.0, 10000.0], [10000.0]], [[0.0, 1.0e-6], [1.0e-6]]
    )
    solution = bm.AnisotropicSolution(
        "three endmembers", model, excess_psi, {"coefficient": 0.025}, [0.4, 0.4, 0.2]
    )
    solution.set_state(20.0e9, 1000.0)
    assert solution.depsdn_fixed_VT.shape == (3, 3, 3)
    assert_allclose(
        solution.depsdn_fixed_VT[:, :, 0], solution.depsdn_fixed_VT[:, :, 1]
    )
    assert_allclose(
        np.linalg.det(solution.cell_vectors), solution.molar_volume, rtol=1.0e-12
    )
    assert_array_equal(solution.ones, np.ones(3))
    assert_array_equal(solution.eye, np.eye(3))
    assert solution.endmembers[0][0] is not solution.endmembers[1][0]
