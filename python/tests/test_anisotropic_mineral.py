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

"""Reference parity, tensor rotations and thermodynamic derivative checks."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011
from burnman_cpp.classes.anisotropicmineral import (
    AnisotropicMineral,
    convert_f_Pth_to_f_T_derivatives,
    deformation_gradient_alpha_and_compliance,
)


@pytest.fixture
def python_reference():
    """Use the unmodified pinned reference, which includes the shear-index fix."""
    return pytest.importorskip("burnman.classes.anisotropicmineral")


def model_parameters(system="orthorhombic", thermal=False):
    reference = SLB_2011.forsterite()
    angles = {
        "orthorhombic": [90.0, 90.0, 90.0],
        "monoclinic": [90.0, 100.0, 90.0],
        "triclinic": [85.0, 80.0, 87.0],
    }[system]
    cosines = np.cos(np.deg2rad(angles))
    volume_factor = np.sqrt(1.0 + 2.0 * np.prod(cosines) - np.sum(cosines**2))
    lengths = np.array([4.7646, 10.2296, 5.9942])
    lengths *= np.cbrt(reference.params.V_0 / (np.prod(lengths) * volume_factor))
    cell = np.r_[lengths, angles]
    c = np.zeros((6, 6, 3, 2 if thermal else 1))
    c[:, :, 1, 0] = [
        [0.44, -0.12, -0.1, 0.0, 0.0, 0.0],
        [-0.12, 0.78, -0.22, 0.0, 0.0, 0.0],
        [-0.1, -0.22, 0.66, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.97, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.61, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.55],
    ]
    c[:3, :3, 2, 0] = [[0.24, -0.12, -0.1], [-0.12, 0.38, -0.22], [-0.1, -0.22, 0.26]]
    if system == "monoclinic":
        for i, j, value in [(0, 4, -1.0), (1, 4, 0.4), (2, 4, 0.2), (3, 5, 0.3)]:
            c[i, j, 1, 0] = c[j, i, 1, 0] = 0.1 * value
        c[:3, 4, 2, 0] = c[4, :3, 2, 0] = [-0.1, 0.04, 0.02]
    elif system == "triclinic":
        for i, j, value in [(0, 3, 0.4), (0, 4, -1.0), (0, 5, -0.6), (3, 4, 1.97)]:
            c[i, j, 1, 0] = c[j, i, 1, 0] = 0.1 * value
        c[0, 3:, 2, 0] = c[3:, 0, 2, 0] = [0.04, -0.1, -0.06]
    if thermal:
        c[:3, :3, 0, 1] = np.diag([1.0, -2.0, 1.0]) * 1.0e-11
        c[:3, :3, 1, 1] = np.diag([0.03, -0.02, -0.01]) * 1.0e-10
    return reference, cell, c


def make_model(system="orthorhombic", thermal=False):
    reference, cell, c = model_parameters(system, thermal)
    return bm.AnisotropicMineral(reference, cell, c, [0, 1, 2])


@pytest.mark.parametrize("system", ["orthorhombic", "monoclinic", "triclinic"])
@pytest.mark.parametrize("thermal", [False, True])
def test_python_reference(system, thermal, python_reference):
    reference_module = python_reference
    reference_dataset = pytest.importorskip("burnman.minerals.SLB_2011")
    reference, cell, c = model_parameters(system, thermal)
    native = AnisotropicMineral(reference, cell, c, [0, 1, 2])
    original = reference_module.AnisotropicMineral(
        reference_dataset.forsterite(), cell, c, [0, 1, 2]
    )
    properties = [
        "molar_volume",
        "density",
        "molar_gibbs",
        "molar_entropy",
        "molar_heat_capacity_p",
        "molar_heat_capacity_v",
        "molar_isometric_heat_capacity",
        "grueneisen_parameter",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "isothermal_bulk_modulus_voigt",
        "isentropic_bulk_modulus_voigt",
        "isentropic_shear_modulus_reuss",
        "isentropic_shear_modulus_voigt",
        "cell_parameters",
        "cell_vectors",
        "unrotated_cell_vectors",
        "rotation_matrix",
        "deformation_gradient_tensor",
        "thermal_expansivity_tensor",
        "thermal_stress_tensor",
        "grueneisen_tensor",
        "isothermal_compressibility_tensor",
        "isentropic_compressibility_tensor",
        "isothermal_compliance_tensor",
        "isentropic_compliance_tensor",
        "isothermal_stiffness_tensor",
        "isentropic_stiffness_tensor",
        "full_isothermal_compliance_tensor",
        "full_isentropic_compliance_tensor",
        "full_isothermal_stiffness_tensor",
        "full_isentropic_stiffness_tensor",
        "Pth",
        "dPthdf",
    ]
    for pressure, temperature in [(1.0e9, 300.0), (20.0e9, 1000.0), (5.0e9, 1600.0)]:
        native.set_state(pressure, temperature)
        original.set_state(pressure, temperature)
        assert np.all(np.linalg.eigvalsh(native.isentropic_stiffness_tensor) > 0.0)
        for prop in properties:
            actual, expected = getattr(native, prop), getattr(original, prop)
            if prop in ("Pth", "dPthdf"):
                # Both vanish at T_0; volume-root roundoff is measured in Pa.
                assert_allclose(actual, expected, rtol=2.0e-8, atol=0.1, err_msg=prop)
                continue
            scale = max(float(np.max(np.abs(expected))), 1.0e-30)
            assert_allclose(
                np.asarray(actual) / scale,
                np.asarray(expected) / scale,
                rtol=2.0e-8,
                atol=2.0e-11,
                err_msg=prop,
            )
        assert_allclose(
            native.wave_velocities([1.0, 2.0, 3.0])[0],
            original.wave_velocities([1.0, 2.0, 3.0])[0],
            rtol=2.0e-8,
        )


@pytest.mark.parametrize("system", ["orthorhombic", "monoclinic", "triclinic"])
def test_consistency_and_rotations(system):
    mineral = make_model(system, thermal=True)
    assert isinstance(mineral, bm.Material)
    assert isinstance(mineral, bm.AnisotropicMaterial)
    assert isinstance(mineral, bm._core.Mineral)
    assert mineral.orthotropic == (system == "orthorhombic")
    pressure, temperature = 20.0e9, 1000.0
    dp, dt = 10000.0, 0.01
    mineral.set_state(pressure - dp, temperature)
    fmP = mineral.deformation_gradient_tensor
    mineral.set_state(pressure + dp, temperature)
    fpP = mineral.deformation_gradient_tensor
    mineral.set_state(pressure, temperature - dt)
    fmT = mineral.deformation_gradient_tensor
    mineral.set_state(pressure, temperature + dt)
    fpT = mineral.deformation_gradient_tensor
    mineral.set_state(pressure, temperature)
    invF = np.linalg.inv(mineral.deformation_gradient_tensor)
    lp = (fpP - fmP) / (2.0 * dp) @ invF
    lt = (fpT - fmT) / (2.0 * dt) @ invF
    r = mineral.rotation_matrix
    assert_allclose(
        r @ (-0.5 * (lp + lp.T)) @ r.T,
        mineral.isothermal_compressibility_tensor,
        rtol=1.0e-6,
        atol=1.0e-20,
    )
    assert_allclose(
        r @ (0.5 * (lt + lt.T)) @ r.T,
        mineral.thermal_expansivity_tensor,
        rtol=1.0e-6,
        atol=1.0e-14,
    )
    assert_allclose(
        np.linalg.det(mineral.cell_vectors), mineral.molar_volume, rtol=1.0e-12
    )
    assert_allclose(
        np.trace(mineral.thermal_expansivity_tensor),
        mineral.thermal_expansivity,
        rtol=1.0e-12,
    )
    assert_allclose(
        mineral.isothermal_compliance_tensor[:3, :3].sum(),
        mineral.isothermal_compressibility_reuss,
        rtol=1.0e-12,
    )
    assert_allclose(
        mineral.isentropic_compliance_tensor[:3, :3].sum(),
        mineral.isentropic_compressibility_reuss,
        rtol=1.0e-12,
    )
    assert_allclose(
        mineral.isothermal_compliance_tensor @ mineral.isothermal_stiffness_tensor,
        np.eye(6),
        atol=1.0e-13,
    )
    # Infer the frame from cell vectors, independently of rotation_matrix.
    independent_r = mineral.cell_vectors.T @ np.linalg.inv(
        mineral.unrotated_cell_vectors.T
    )
    assert_allclose(r, independent_r, atol=1.0e-14)
    full = bm.AnisotropicMaterial(
        1.0, mineral._unrotated_isothermal_stiffness_tensor
    ).full_isentropic_stiffness_tensor
    rotated = np.einsum("ip,jq,kr,ls,pqrs->ijkl", r, r, r, r, full)
    assert_allclose(
        rotated, mineral.full_isothermal_stiffness_tensor, atol=0.01, rtol=1.0e-11
    )
    c = mineral.isentropic_stiffness_tensor
    mineral.reset()
    assert_allclose(mineral.isentropic_stiffness_tensor, c)
    # Rotation preserves quadratic elastic energy.
    strain = (
        np.array([[1.0, 0.2, -0.3], [0.2, -0.4, 0.15], [-0.3, 0.15, -0.6]]) * 1.0e-3
    )
    rotated_strain = r @ strain @ r.T
    assert_allclose(
        np.einsum("ij,ijkl,kl", strain, full, strain),
        np.einsum("ij,ijkl,kl", rotated_strain, rotated, rotated_strain),
        rtol=1.0e-12,
    )


def test_custom_psi_and_polynomial_derivatives():
    reference, cell, c = model_parameters("triclinic", thermal=True)
    params = {"c": c}
    custom = bm.AnisotropicMineral(
        reference,
        cell,
        params,
        [0, 1, 2],
        AnisotropicMineral.standard_psi_function,
        False,
    )
    standard = make_model("triclinic", thermal=True)
    f, pth, df, dp = -0.1, 2.0e9, 1.0e-6, 1000.0
    psi, dpsi_df, dpsi_dp = custom.evaluate_psi(f, pth)
    assert_allclose(
        (custom.evaluate_psi(f + df, pth)[0] - custom.evaluate_psi(f - df, pth)[0])
        / (2 * df),
        dpsi_df,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    assert_allclose(
        (custom.evaluate_psi(f, pth + dp)[0] - custom.evaluate_psi(f, pth - dp)[0])
        / (2 * dp),
        dpsi_dp,
        rtol=1.0e-8,
        atol=1.0e-20,
    )
    custom.set_state(20.0e9, 1000.0)
    standard.set_state(20.0e9, 1000.0)
    assert_allclose(
        custom.isentropic_stiffness_tensor, standard.isentropic_stiffness_tensor
    )
    assert_allclose(custom.cell_vectors, standard.cell_vectors)
    for mineral in [custom, standard]:
        copy = mineral.copy()
        assert copy.isotropic_mineral is not mineral.isotropic_mineral
        copy.set_state(1.0e9, 300.0)
        assert mineral.pressure == 20.0e9
        assert mineral.isotropic_mineral.pressure == 20.0e9


def test_frechet_and_derivative_helpers(python_reference):
    module = python_reference
    psi = np.diag([-0.1, -0.2, 0.15])
    df = np.array([[0.2, 0.1, -0.04], [0.1, 0.3, 0.02], [-0.04, 0.02, 0.5]])
    dt = np.array([[0.4, -0.2, 0.1], [-0.2, 0.2, 0.3], [0.1, 0.3, -0.6]]) * 1.0e-5
    inputs = (2.0e-5, 1.0e-11, psi, np.eye(6), df, dt)
    for actual, expected in zip(
        deformation_gradient_alpha_and_compliance(*inputs),
        module.deformation_gradient_alpha_and_compliance(*inputs),
    ):
        assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-18)
    inputs = (np.eye(6), np.eye(6) * 1.0e-11, 1.0e6, 2.0e9)
    for actual, expected in zip(
        convert_f_Pth_to_f_T_derivatives(*inputs),
        module.convert_f_Pth_to_f_T_derivatives(*inputs),
    ):
        assert_allclose(actual, expected, rtol=1.0e-14)


def test_evaluation_volume_states_and_snapshots():
    mineral = make_model()
    assert not mineral.has_state()
    for prop in [
        "density",
        "isentropic_bulk_modulus_reuss",
        "isentropic_stiffness_tensor",
        "full_isentropic_stiffness_tensor",
    ]:
        with pytest.raises(RuntimeError):
            getattr(mineral, prop)
    result = mineral.evaluate(
        ["density", "cell_vectors"], [[1.0e9, 2.0e9]], [[300.0, 1000.0]]
    )
    assert result[0].shape == (1, 2)
    assert result[1].shape == (1, 2, 3, 3)
    assert not mineral.has_state()
    assert not mineral.isotropic_mineral.has_state()
    mineral.set_state(20.0e9, 1000.0)
    volume = mineral.molar_volume
    c = mineral.isentropic_stiffness_tensor
    mineral.isentropic_stiffness_tensor[:] = 0.0
    assert_array_equal(mineral.isentropic_stiffness_tensor, c)
    mineral.evaluate(["V", "rho"], [1.0e9], [300.0])
    assert mineral.pressure == 20.0e9
    assert mineral.isotropic_mineral.pressure == 20.0e9
    mineral.set_state_with_volume(volume, 1000.0)
    assert_allclose(mineral.pressure, 20.0e9, rtol=1.0e-10)
    assert_allclose(mineral.isentropic_stiffness_tensor, c)
    for prop in [
        "K_T",
        "K_S",
        "beta_T",
        "beta_S",
        "shear_modulus",
        "p_wave_velocity",
        "shear_wave_velocity",
    ]:
        with pytest.raises(NotImplementedError):
            getattr(mineral, prop)


def test_invalid_parameters():
    reference, cell, c = model_parameters()
    for bad in [
        np.zeros((6, 6, 1, 1)),
        c[:5],
        np.full(c.shape, np.nan),
        np.zeros(c.shape),
    ]:
        with pytest.raises(ValueError):
            bm.AnisotropicMineral(reference, cell, bad)
    for index, value in [((0, 0, 0, 0), 1.0), ((0, 0, 1, 0), 0.8), ((0, 1, 2, 0), 0.2)]:
        bad = c.copy()
        bad[index] = value
        with pytest.raises(ValueError):
            bm.AnisotropicMineral(reference, cell, bad)
    with pytest.raises(ValueError):
        bm.AnisotropicMineral(reference, cell * 2.0, c)
    with pytest.raises(ValueError):
        bm.AnisotropicMineral(
            reference, cell, {}, psi_function=lambda f, p, params: (np.eye(6),) * 3
        )
    with pytest.raises(ValueError, match="vanish"):
        bm.AnisotropicMineral(
            reference,
            cell,
            {},
            psi_function=lambda f, p, params: (np.eye(6),) * 3,
            orthotropic=True,
        )
    reference, cell, c = model_parameters("triclinic")
    with pytest.raises(ValueError, match="frame convention"):
        bm.AnisotropicMineral(reference, cell, c)
    with pytest.raises(ValueError):
        bm.AnisotropicMineral(reference, cell, c, [0, 0, 1])
