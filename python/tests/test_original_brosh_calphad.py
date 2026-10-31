# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_brosh_calphad.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict, mineral_eos, eos_call
import numpy as np
from scipy.optimize import brentq
import burnman_cpp as burnman


class BroshCalphad(BurnManTest):

    def setUp(self):
        self.phases = [
            getattr(burnman.minerals.SE_2015, name + "_iron")()
            for name in ("bcc", "fcc", "hcp", "liquid")
        ]

    def test_zero_temperature_limit(self):
        for mineral in self.phases:
            eos, p = (mineral_eos(mineral), parameter_dict(mineral.params))
            P = 300000000000.0
            with (
                self.subTest(phase=p["name"]),
                np.errstate(over="raise", invalid="raise", divide="raise"),
            ):
                V = eos_call(eos, "volume", P, 0.0, p)
                self.assertTrue(
                    np.isfinite(eos_call(eos, "gibbs_energy", P, 0.0, V, p))
                )
                self.assertTrue(np.isfinite(eos_call(eos, "entropy", P, 0.0, V, p)))
                self.assertGreater(
                    eos_call(eos, "isothermal_bulk_modulus_reuss", P, 0.0, V, p), 0.0
                )
                self.assertEqual(
                    eos_call(eos, "molar_heat_capacity_p", P, 0.0, V, p), 0.0
                )
                self.assertEqual(
                    eos_call(eos, "thermal_expansivity", P, 0.0, V, p), 0.0
                )
                self.assertTrue(np.isfinite(eos_call(eos, "entropy", P, 0.01, V, p)))

    def test_derivatives_at_sgte_joins(self):
        for mineral in self.phases:
            eos, p = (mineral_eos(mineral), parameter_dict(mineral.params))
            for T in (1811.0, 6000.0, 12000.0):
                with self.subTest(phase=p["name"], T=T):
                    P, dT = (300000000000.0, 0.001)
                    V = eos_call(eos, "volume", P, T, p)
                    dS = eos_call(eos, "entropy", P, T + dT, V, p) - eos_call(
                        eos, "entropy", P, T, V, p
                    )
                    dV = eos_call(eos, "volume", P, T + dT, p) - V
                    self.assertFloatEqual(
                        T * dS / dT,
                        eos_call(eos, "molar_heat_capacity_p", P, T, V, p),
                        tol=1e-06,
                    )
                    self.assertFloatEqual(
                        dV / dT / V,
                        eos_call(eos, "thermal_expansivity", P, T, V, p),
                        tol=1e-06,
                    )

    def test_se2015_core_melting(self):
        hcp, liquid = self.phases[2:]
        P = 363850000000.0

        def delta_g(T):
            hcp.set_state(P, T)
            liquid.set_state(P, T)
            return hcp.molar_gibbs - liquid.molar_gibbs

        T = brentq(delta_g, 6000.0, 7000.0)
        self.assertLess(abs(T - 6453.0), 5.0)
        hcp.set_state(P, T)
        self.assertLess(abs(hcp.density / 1000.0 - 13.71), 0.01)
