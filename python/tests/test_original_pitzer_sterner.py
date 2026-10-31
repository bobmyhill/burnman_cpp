# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_pitzer_sterner.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

import unittest
import numpy as np
from burnman_cpp.minerals import HP_2011_fluids, Pitzer_Sterner_1994
from burnman_cpp.tools.eos import check_eos_consistency


class PitzerSternerTests(unittest.TestCase):

    def test_thermocalc_absolute_water_energies(self):
        water = HP_2011_fluids.H2O()
        for p, t, gibbs, enthalpy in (
            (250000000.0, 873.15, -367307.85, -240824.1),
            (1000000000.0, 873.15, -351096.4, -234778.7),
            (100000000.0, 873.15, -372392.59, -236340.4),
            (1000000000.0, 773.15, -338135.81, -240869.3),
        ):
            with self.subTest(P=p, T=t):
                water.set_state(p, t)
                self.assertAlmostEqual(water.molar_gibbs, gibbs, delta=10.0)
                self.assertAlmostEqual(water.molar_enthalpy, enthalpy, delta=10.0)

    def test_thermodynamic_consistency(self):
        for constructor in (HP_2011_fluids.H2O, Pitzer_Sterner_1994.H2O_Pitzer_Sterner):
            with self.subTest(reference=constructor.__name__):
                self.assertTrue(
                    check_eos_consistency(
                        constructor(),
                        P=1000000000.0,
                        T=1000.0,
                        tol=1e-05,
                        verbose=False,
                        including_shear_properties=False,
                    )
                )

    def test_water_domain(self):
        water = HP_2011_fluids.H2O()
        for pressure in (0.0, -1.0, np.nextafter(5000000000.0, np.inf), np.inf, np.nan):
            with self.subTest(P=pressure):
                with self.assertRaises(ValueError):
                    water.set_state(pressure, 1000.0)
                    _ = water.molar_volume
        for temperature in (
            np.nextafter(500.0, 0.0),
            np.nextafter(1700.0, np.inf),
            -1.0,
            np.inf,
            np.nan,
        ):
            with self.subTest(T=temperature):
                with self.assertRaises(ValueError):
                    water.set_state(100000000.0, temperature)
                    _ = water.molar_gibbs
        for temperature in (500.0, 1700.0):
            water.set_state(5000000000.0, temperature)
            self.assertTrue(np.isfinite(water.molar_gibbs))
            self.assertGreater(water.isothermal_bulk_modulus_reuss, 0.0)
