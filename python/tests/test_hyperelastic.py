# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Finite-strain reference parity, objectivity and Helmholtz derivatives."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import expm

import burnman_cpp as bm
from burnman_cpp.utils.anisotropy import (
    contract_compliances,
    contract_stiffnesses,
    contract_strains,
    contract_stresses,
    expand_strains,
    expand_stresses,
    polar_decomposition,
    voigt_notation_to_compliance_tensor,
    voigt_notation_to_stiffness_tensor,
)
from burnman_cpp.tools.eos import check_hyperelastic_eos_consistency
from test_anisotropic_mineral import model_parameters, python_reference


def make_model(system="orthorhombic", thermal=False):
    scalar, cell, c = model_parameters(system, thermal)
    if thermal:
        c[0, 3, 0, 1] = c[3, 0, 0, 1] = 1e-12
    return bm.HyperelasticMineral(bm.AnisotropicMineral(scalar, cell, c, [0, 1, 2]))


def set_finite_state(m, temperature=1000.0):
    H = np.array(
        [[0.010, 0.004, -0.002], [0.004, -0.006, 0.003], [-0.002, 0.003, -0.004]]
    )
    m.hydrostatic.set_state(20e9, temperature)
    cells = (expm(H) @ m.hydrostatic.unrotated_cell_vectors.T).T
    m.set_state_with_molar_cell_vectors(cells, temperature)
    return H, cells


@pytest.mark.parametrize("system", ["orthorhombic", "triclinic"])
@pytest.mark.parametrize("frame", ["current", "crystal"])
def test_python_reference(system, frame, python_reference):
    reference_module = pytest.importorskip("burnman.classes.hyperelastic")
    minerals = pytest.importorskip("burnman.minerals.SLB_2011")
    scalar, cell, c = model_parameters(system, True)
    c[0, 3, 0, 1] = c[3, 0, 0, 1] = 1e-12
    native = bm.HyperelasticMineral(bm.AnisotropicMineral(scalar, cell, c, [0, 1, 2]))
    original = reference_module.HyperelasticMineral(
        python_reference.AnisotropicMineral(minerals.forsterite(), cell, c, [0, 1, 2])
    )
    _, cells = set_finite_state(native)
    original.set_state_with_molar_cell_vectors(cells, 1000.0)
    for m in (native, original):
        m.set_output_frame(frame)
    for name in [
        "cell_vectors",
        "cell_parameters",
        "rotation_matrix",
        "hencky_strain",
        "unrotated_cell_vectors",
        "deformation_gradient_tensor",
        "pressure",
        "_helmholtz_elastic",
        "helmholtz",
        "entropy",
        "internal_energy",
        "cauchy_stress",
        "_cauchy_stress_unrotated",
        "thermal_stress",
        "_thermal_stress_unrotated",
        "isothermal_stiffness_tensor",
        "isentropic_stiffness_tensor",
        "full_isothermal_compliance_tensor",
        "full_isentropic_compliance_tensor",
        "thermal_expansivity_tensor",
        "isothermal_compressibility_reuss",
        "isentropic_compressibility_reuss",
        "molar_isometric_heat_capacity",
        "molar_isostress_heat_capacity",
    ]:
        expected = np.asarray(getattr(original, name))
        scale = max(np.max(np.abs(expected)), 1e-30)
        assert_allclose(
            np.asarray(getattr(native, name)) / scale,
            expected / scale,
            rtol=3e-5,
            atol=2e-6,
            err_msg=name,
        )


@pytest.mark.parametrize("system", ["orthorhombic", "triclinic"])
def test_hydrostatic_limit_and_geometry(system):
    m = make_model(system)
    m.hydrostatic.set_state(20e9, 1000.0)
    cells = m.hydrostatic.cell_vectors
    m.set_state_with_molar_cell_vectors(cells, 1000.0)
    q = m.rotation_matrix @ m.hydrostatic.rotation_matrix.T
    assert_allclose(m.hencky_strain, 0.0, atol=1e-12)
    assert_allclose(m.cell_vectors, cells, rtol=1e-13)
    assert_allclose(m.molar_volume, np.linalg.det(cells), rtol=1e-13)
    assert_allclose(m.cauchy_stress / 1e9, -20 * np.eye(3), atol=2e-5)
    expected = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        q,
        q,
        q,
        q,
        m.hydrostatic.full_isothermal_stiffness_tensor,
    )
    assert_allclose(
        m.full_isothermal_stiffness_tensor / 1e9, expected / 1e9, rtol=2e-6, atol=1e-5
    )
    assert_allclose(
        m.molar_isostress_heat_capacity, m.hydrostatic.molar_heat_capacity_p, rtol=2e-6
    )
    directions = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    assert_allclose(
        m.christoffel_tensor(directions) / 1e9,
        q @ m.hydrostatic.christoffel_tensor(directions @ q) @ q.T / 1e9,
        rtol=2e-6,
        atol=2e-5,
    )


def test_finite_strain_thermal_identities_and_objectivity():
    m = make_model("triclinic", True)
    H, cells = set_finite_state(m)
    assert_allclose(m.hencky_strain, H, rtol=1e-11, atol=1e-13)
    stress, C, S = (
        m.cauchy_stress,
        m.isothermal_stiffness_tensor,
        m.isothermal_compliance_tensor,
    )
    pi = contract_stresses(m.thermal_stress)
    assert_allclose(S @ C, np.eye(6), atol=1e-13)
    assert_allclose(
        m.isentropic_stiffness_tensor - C,
        m.V * m.T / m.molar_isometric_heat_capacity * np.outer(pi, pi),
        rtol=1e-9,
        atol=0.01,
    )
    assert_allclose(m.molar_internal_energy, m.molar_helmholtz + m.T * m.molar_entropy)
    assert_allclose(m.pressure, -np.trace(stress) / 3)
    assert np.max(np.abs(C - C.T)) > 1e7
    rotation = expm(
        np.array([[0.0, -0.37, -0.23], [0.37, 0.0, -0.19], [0.23, 0.19, 0.0]])
    )
    m2 = make_model("triclinic", True)
    m2.set_state_with_molar_cell_vectors((rotation @ cells.T).T, m.T)
    assert_allclose(m2.molar_helmholtz, m.molar_helmholtz, rtol=1e-12)
    assert_allclose(
        m2.cauchy_stress / 1e9, rotation @ stress @ rotation.T / 1e9, atol=1e-7
    )
    expected = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        rotation,
        rotation,
        rotation,
        rotation,
        m.full_isothermal_stiffness_tensor,
    )
    assert_allclose(
        m2.full_isothermal_stiffness_tensor / 1e9, expected / 1e9, atol=2e-5
    )


@pytest.mark.parametrize("frame", ["current", "crystal"])
def test_helmholtz_and_maxwell_checker(frame):
    m = make_model("triclinic", True)
    _, cells = set_finite_state(m)
    m.set_output_frame(frame)
    assert check_hyperelastic_eos_consistency(m, T=1000.0, verbose=True)
    assert_array_equal(m._cell_vectors_current, cells)
    assert m.output_frame == frame


def test_frames_copies_validation_and_independent_snapshots():
    m = make_model("triclinic")
    assert m.output_frame == "current"
    m.set_output_frame(bm.OutputFrame.crystal)
    assert m.output_frame == "crystal"
    with pytest.raises(RuntimeError):
        _ = m.hencky_strain
    _, cells = set_finite_state(m)
    m.set_output_frame("current")
    stress, C = m.cauchy_stress, m.full_isothermal_stiffness_tensor
    Q, R = m.rotation_to_crystallographic_frame, m.rotation_matrix
    m.set_output_frame("crystal")
    assert_allclose(m.rotation_matrix, Q @ R, atol=1e-13)
    assert_allclose(m.cauchy_stress, Q @ stress @ Q.T, rtol=1e-12, atol=0.01)
    assert_allclose(
        m.full_isothermal_stiffness_tensor,
        np.einsum("ia,jb,kc,ld,abcd->ijkl", Q, Q, Q, Q, C),
        rtol=1e-10,
        atol=0.1,
    )
    copy = m.copy()
    assert copy.hydrostatic is not m.hydrostatic
    copy.set_state_with_molar_cell_vectors(cells * 1.001, 900.0)
    assert m.T == 1000.0
    assert m.hydrostatic.temperature == 1000.0
    m._cell_vectors_current[:] = 0.0
    assert_array_equal(m._cell_vectors_current, cells)
    with pytest.raises(ValueError):
        m.set_output_frame("invalid")
    assert m.output_frame == "crystal"
    for cell, t in [
        (np.zeros((3, 3)), 300.0),
        (np.diag([-1.0, 1.0, 1.0]), 300.0),
        (cells, np.nan),
        (cells, -1.0),
    ]:
        with pytest.raises(ValueError):
            m.set_state_with_molar_cell_vectors(cell, t)
    assert_array_equal(m._cell_vectors_current, cells)
    with pytest.raises(NotImplementedError):
        m.set_state(1e9, 300.0)
    with pytest.raises(NotImplementedError):
        m.set_state_with_volume(1e-5, 300.0)


def test_tensor_utilities_preserve_nonsymmetric_major_indices():
    C = np.arange(36.0).reshape(6, 6) + np.eye(6) * 100
    assert_array_equal(contract_stiffnesses(voigt_notation_to_stiffness_tensor(C)), C)
    assert_array_equal(contract_compliances(voigt_notation_to_compliance_tensor(C)), C)
    strain = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert_array_equal(expand_strains(contract_strains(strain)), strain)
    assert_array_equal(expand_stresses(contract_stresses(strain)), strain)
    F = np.array([[1.01, 0.04, 0.02], [0.0, 0.99, 0.01], [0.01, 0.0, 1.02]])
    for measure in ("U", "V", "H"):
        R, S = polar_decomposition(F, measure)
        expected = (
            R @ expm(S) if measure == "H" else (R @ S if measure == "U" else S @ R)
        )
        assert_allclose(F, expected, atol=1e-14)
    with pytest.raises(ValueError):
        polar_decomposition(F, "invalid")
