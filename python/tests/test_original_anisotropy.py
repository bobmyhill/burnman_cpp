# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_anisotropy.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
import numpy as np
from scipy.linalg import logm
from burnman_cpp.classes import anisotropy
from burnman_cpp.utils.unitcell import (
    cell_parameters_to_vectors,
    cell_vectors_to_parameters,
)
from burnman_cpp.utils.anisotropy import polar_decomposition


class test_anisotropy(BurnManTest):

    def test_isotropic(self):
        l, G = (40000000000.0, 24000000000.0)
        elastic_constants = [l, G]
        rho = 2735.0
        m = anisotropy.IsotropicMaterial(rho, elastic_constants)
        E = G * (3.0 * l + 2.0 * G) / (l + G)
        Vp = np.sqrt(
            (
                m.isentropic_bulk_modulus_reuss
                + 4.0 / 3.0 * m.isentropic_shear_modulus_reuss
            )
            / rho
        )
        Vs = np.sqrt(m.isentropic_shear_modulus_reuss / rho)
        array_1 = [
            m.isentropic_bulk_modulus_reuss,
            m.isentropic_bulk_modulus_vrh,
            m.isentropic_bulk_modulus_voigt,
            m.isentropic_shear_modulus_reuss,
            m.isentropic_shear_modulus_vrh,
            m.isentropic_shear_modulus_voigt,
            m.isentropic_universal_elastic_anisotropy + 6.0,
            m.isentropic_isotropic_poisson_ratio,
            E,
            Vp,
            Vs,
            Vs,
        ]
        d1 = [1.0, 0.0, 0.0]
        d2 = [0.0, 1.0, 0.0]
        ds = np.array([d1, d2])
        beta_100 = m.isentropic_linear_compressibility(direction=ds)[0]
        E_100 = m.isentropic_youngs_modulus(direction=ds)[0]
        G_100_010 = m.isentropic_shear_modulus(plane_normal=d1, shear_direction=d2)
        nu_100_010 = m.isentropic_poissons_ratio(
            axial_direction=d1, lateral_direction=d2
        )
        wave_speeds, _ = m.wave_velocities(propagation_direction=d1)
        Vp, Vs1, Vs2 = wave_speeds
        array_2 = [
            1.0 / 3.0 / beta_100,
            1.0 / 3.0 / beta_100,
            1.0 / 3.0 / beta_100,
            G_100_010,
            G_100_010,
            G,
            6.0,
            nu_100_010,
            E_100,
            Vp,
            Vs1,
            Vs2,
        ]
        self.assertArraysAlmostEqual(array_1, array_2)

    def test_christoffel_under_hydrostatic_stress(self):
        """
        Hydrostatic stress preserves the isotropic acoustic tensor in every direction.
        """
        lam, G = (40000000000.0, 24000000000.0)
        m = anisotropy.IsotropicMaterial(2735.0, [lam, G])
        directions = np.array(
            [[[1.0, 0.0, 0.0], [-1.0, 2.0, 3.0]], [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]]]
        )
        n = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
        expected = G * np.eye(3) + (lam + G) * np.einsum("...i,...k->...ik", n, n)
        actual = m.christoffel_tensor(directions)
        self.assertEqual(actual.shape, expected.shape)
        self.assertArraysAlmostEqual(
            (actual / G).flatten(), (expected / G).flatten(), tol=1e-12, tol_zero=1e-12
        )

    def test_crystal_systems(self):
        m = anisotropy.IsotropicMaterial(3000.0, [1.0, 1.0])
        m = anisotropy.CubicMaterial(3000.0, [1.0, 1.0, 1])
        m = anisotropy.HexagonalMaterial(3000.0, [1.0, 1.0, 1.0, 1.0, 1.0])
        m = anisotropy.TetragonalMaterial(3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        m = anisotropy.TetragonalMaterial(3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        m = anisotropy.RhombohedralMaterial(3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        m = anisotropy.RhombohedralMaterial(
            3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        m = anisotropy.OrthorhombicMaterial(
            3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        m = anisotropy.MonoclinicMaterial(
            3000.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        m = anisotropy.TriclinicMaterial(
            3000.0,
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        )

    def test_tetragonal_ii(self):
        m = anisotropy.TetragonalMaterial(3000.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertFloatEqual(
            m.isentropic_stiffness_tensor[0][5], -m.isentropic_stiffness_tensor[1][5]
        )

    def test_rhombohedral_i(self):
        m = anisotropy.RhombohedralMaterial(3000.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertFloatEqual(
            m.isentropic_stiffness_tensor[0][3], -m.isentropic_stiffness_tensor[1][3]
        )

    def test_rhombohedral_ii(self):
        m = anisotropy.RhombohedralMaterial(
            3000.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        self.assertArraysAlmostEqual(
            [m.isentropic_stiffness_tensor[0][3], m.isentropic_stiffness_tensor[0][4]],
            [
                -m.isentropic_stiffness_tensor[1][3],
                -m.isentropic_stiffness_tensor[1][4],
            ],
        )

    def test_cell_parameters_conversions(self):
        cell_parameters = np.array([1.0, 4.2, 2.2, 80.0, 85.0, 88.0])
        lengths_orig = cell_parameters[:3]
        cos_a = np.cos(np.deg2rad(cell_parameters[3:]))
        for convention in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
            [2, 0, 1],
        ]:
            v = cell_parameters_to_vectors(cell_parameters, convention)
            p = cell_vectors_to_parameters(v, convention)
            self.assertArraysAlmostEqual(cell_parameters, p)
            lengths = np.linalg.norm(v, axis=1)
            cos_a2 = np.array(
                [
                    np.dot(v[1], v[2]) / lengths[1] / lengths[2],
                    np.dot(v[0], v[2]) / lengths[0] / lengths[2],
                    np.dot(v[0], v[1]) / lengths[0] / lengths[1],
                ]
            )
            self.assertArraysAlmostEqual(cos_a, cos_a2)
            self.assertArraysAlmostEqual(lengths_orig, lengths)

    def test_cell_parameters_conversions_with_rotation(self):
        cell_parameters = np.array([1.0, 4.2, 2.2, 80.0, 85.0, 88.0])
        lengths_orig = cell_parameters[:3]
        cos_a = np.cos(np.deg2rad(cell_parameters[3:]))
        for convention in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 1, 0],
            [2, 0, 1],
        ]:
            v = cell_parameters_to_vectors(cell_parameters, convention)
            theta = 0.34
            c, s = (np.cos(theta), np.sin(theta))
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            v = np.dot(v, R)
            p = cell_vectors_to_parameters(v, convention, rotate=True)
            self.assertArraysAlmostEqual(cell_parameters, p)
            lengths = np.linalg.norm(v, axis=1)
            cos_a2 = np.array(
                [
                    np.dot(v[1], v[2]) / lengths[1] / lengths[2],
                    np.dot(v[0], v[2]) / lengths[0] / lengths[2],
                    np.dot(v[0], v[1]) / lengths[0] / lengths[1],
                ]
            )
            self.assertArraysAlmostEqual(cos_a, cos_a2)
            self.assertArraysAlmostEqual(lengths_orig, lengths)

    def test_polar_decomposition(self):
        F = np.array([[1.0, 0.2, 0.3], [0.0, 1.0, 0.4], [0.0, 0.0, 1.0]])
        R1, U = polar_decomposition(F)
        R2, U = polar_decomposition(F, "U")
        R3, V = polar_decomposition(F, "V")
        R4, H = polar_decomposition(F, "H")
        self.assertArraysAlmostEqual(
            np.dot(R1.T, R1).flatten(), np.eye(3).flatten(), tol_zero=1e-12
        )
        self.assertArraysAlmostEqual(R1.flatten(), R2.flatten(), tol_zero=1e-12)
        self.assertArraysAlmostEqual(R1.flatten(), R3.flatten(), tol_zero=1e-12)
        self.assertArraysAlmostEqual(R1.flatten(), R4.flatten(), tol_zero=1e-12)
        self.assertArraysAlmostEqual(
            np.dot(R1, U).flatten(), np.dot(V, R1).flatten(), tol_zero=1e-12
        )
        self.assertArraysAlmostEqual(logm(U).flatten(), H.flatten(), tol_zero=1e-12)
