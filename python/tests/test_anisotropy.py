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

"""Elastic limits, tensor contractions, broadcasting and Python parity."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.classes import anisotropy

CASES = [
    ("IsotropicMaterial", [40, 24]),
    ("CubicMaterial", [220, 80, 65]),
    ("HexagonalMaterial", [220, 80, 70, 250, 65]),
    ("TetragonalMaterial", [220, 80, 70, 250, 65, 75]),
    ("TetragonalMaterial", [220, 80, 70, 8, 250, 65, 75]),
    ("RhombohedralMaterial", [220, 80, 70, 8, 250, 65, 75]),
    ("RhombohedralMaterial", [220, 80, 70, 8, 6, 250, 65, 75]),
    ("OrthorhombicMaterial", [320, 68, 72, 201, 77, 233, 64, 78, 80]),
    ("MonoclinicMaterial", [220, 80, 70, 8, 230, 75, 6, 250, 7, 65, 5, 70, 75]),
    (
        "TriclinicMaterial",
        [220, 80, 70, 8, 6, 5, 230, 75, 4, 3, 2, 250, 7, 6, 5, 65, 4, 3, 70, 2, 75],
    ),
]
BOUNDS = [
    "isentropic_bulk_modulus_voigt",
    "isentropic_bulk_modulus_reuss",
    "isentropic_bulk_modulus_vrh",
    "isentropic_shear_modulus_voigt",
    "isentropic_shear_modulus_reuss",
    "isentropic_shear_modulus_vrh",
    "isentropic_universal_elastic_anisotropy",
    "isentropic_isotropic_poisson_ratio",
]


def test_isotropic_limits_and_direction_shapes():
    material = anisotropy.IsotropicMaterial(rho=2735.0, cijs=[40.0e9, 24.0e9])
    assert isinstance(material, bm.Material)
    assert material.density == 2735.0
    assert not material.has_state()
    bulk, shear = 56.0e9, 24.0e9
    assert_allclose(
        [getattr(material, p) for p in BOUNDS[:6]], [bulk] * 3 + [shear] * 3
    )
    assert abs(material.isentropic_universal_elastic_anisotropy) < 1.0e-13
    directions = np.array(
        [[[1.0, 0.0, 0.0], [-1.0, 2.0, 3.0]], [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]]]
    )
    n = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    expected = shear * np.eye(3) + 64.0e9 * np.einsum("...i,...k->...ik", n, n)
    assert_allclose(material.christoffel_tensor(directions), expected, atol=1.0e-4)
    assert_allclose(
        material.isentropic_linear_compressibility(directions), 1.0 / (3.0 * bulk)
    )
    assert_allclose(material.isentropic_youngs_modulus(directions), 63.0e9)
    assert_allclose(
        material.isentropic_shear_modulus(
            [2.0, 0.0, 0.0], [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        ),
        shear,
    )
    assert_allclose(
        material.isentropic_poissons_ratio(
            [2.0, 0.0, 0.0], [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        ),
        0.3125,
    )
    speeds, polarizations = material.wave_velocities(directions)
    assert speeds.shape == (2, 2, 3)
    assert polarizations.shape == (2, 2, 3, 3)
    assert_allclose(
        speeds,
        np.broadcast_to(
            np.sqrt(np.array([88.0e9, shear, shear]) / 2735.0), speeds.shape
        ),
    )
    assert material.christoffel_tensor(np.empty((0, 3))).shape == (0, 3, 3)
    assert material.wave_velocities(np.empty((0, 3)))[0].shape == (0, 3)
    assert material.isentropic_youngs_modulus(np.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("name,cijs", CASES)
def test_symmetries_tensors_and_polarizations(name, cijs):
    material = getattr(anisotropy, name)(3300.0, np.array(cijs) * 1.0e9)
    assert getattr(bm, name) is getattr(anisotropy, name)
    copied = material.copy()
    assert type(copied) is type(material)
    c = material.isentropic_stiffness_tensor
    s = material.isentropic_compliance_tensor
    assert_allclose(c @ s, np.eye(6), atol=1.0e-14)
    full_c = material.full_isentropic_stiffness_tensor
    full_s = material.full_isentropic_compliance_tensor
    identity = np.eye(3)
    symmetric_identity = 0.5 * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    )
    assert_allclose(
        np.einsum("ijmn,mnkl->ijkl", full_c, full_s), symmetric_identity, atol=1.0e-14
    )
    assert_allclose(full_s[1, 2, 1, 2], s[3, 3] / 4.0)
    directions = np.array([[1.0, 2.0, -3.0], [2.0, -3.0, 1.0]])
    q = material.christoffel_tensor(directions)
    speeds, p = material.wave_velocities(directions)
    assert np.all(np.diff(speeds, axis=-1) <= 0.0)
    assert_allclose(
        q @ p, p * (speeds**2 * material.density)[..., None, :], atol=1.0e-3
    )
    assert_allclose(
        p.swapaxes(-1, -2) @ p, np.broadcast_to(identity, p.shape), atol=1.0e-14
    )
    material.set_state(1.0e9, 300.0)
    material.reset()
    assert_allclose(material.isentropic_compliance_tensor, s)
    c[:] = 0.0
    assert np.any(material.isentropic_stiffness_tensor)
    assert_allclose(copied.isentropic_compliance_tensor, s)


@pytest.mark.parametrize("name,cijs", CASES)
def test_python_reference(name, cijs):
    reference = pytest.importorskip("burnman.classes.anisotropy")
    cijs = np.array(cijs) * 1.0e9
    native = getattr(anisotropy, name)(3300.0, cijs)
    original = getattr(reference, name)(3300.0, cijs)
    for prop in BOUNDS + [
        "isentropic_stiffness_tensor",
        "isentropic_compliance_tensor",
        "full_isentropic_stiffness_tensor",
        "full_isentropic_compliance_tensor",
    ]:
        assert_allclose(getattr(native, prop), getattr(original, prop), atol=1.0e-14)
    for n in [np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, -3.0]) / np.sqrt(14.0)]:
        lateral = np.cross(n, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral)
        for method, args in [
            ("christoffel_tensor", (n,)),
            ("isentropic_linear_compressibility", (n,)),
            ("isentropic_youngs_modulus", (n,)),
            ("isentropic_shear_modulus", (n, lateral)),
            ("isentropic_poissons_ratio", (n, lateral)),
        ]:
            assert_allclose(
                getattr(native, method)(*args),
                getattr(original, method)(*args),
                atol=1.0e-4,
            )
        assert_allclose(native.wave_velocities(n)[0], original.wave_velocities(n)[0])


def test_invalid_inputs_and_snapshots():
    for rho in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            bm.IsotropicMaterial(rho, [1.0, 1.0])
    for name, cijs in CASES:
        with pytest.raises(ValueError):
            getattr(anisotropy, name)(3000.0, [])
    c = np.eye(6)
    material = bm.AnisotropicMaterial(3000.0, c)
    c[:] = 0.0
    assert_array_equal(material.isentropic_stiffness_tensor, np.eye(6))
    material.full_isentropic_stiffness_tensor[:] = 0.0
    assert material.full_isentropic_stiffness_tensor[0, 0, 0, 0] == 1.0
    with pytest.raises(RuntimeError, match="singular"):
        bm.AnisotropicMaterial(3000.0, c).isentropic_compliance_tensor
    c[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        bm.AnisotropicMaterial(3000.0, c)
    for direction in [[], [1.0, 0.0], np.ones((3, 2)), [np.nan, 0.0, 0.0]]:
        with pytest.raises(ValueError):
            material.wave_velocities(direction)
    for method in [
        material.isentropic_shear_modulus,
        material.isentropic_poissons_ratio,
    ]:
        with pytest.raises(ValueError, match="orthogonal"):
            method([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
