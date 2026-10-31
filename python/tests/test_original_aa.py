# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_aa.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    parameter_dict,
    native_copy,
    mineral_eos,
    eos_call,
    eos_function,
)
import numpy as np
import burnman_cpp as burnman
from burnman_cpp.tools.eos import check_eos_consistency


class AndersonAhrens(BurnManTest):

    def setUp(self):
        self.iron = burnman.minerals.other.liquid_iron()
        self.params = parameter_dict(self.iron.params)
        self.eos = mineral_eos(self.iron)

    def test_reference_state(self):
        p, eos = (parameter_dict(self.params), self.eos)
        T, V, P = (p["T_0"], p["V_0"], p["P_0"])
        self.assertEqual(eos_call(eos, "pressure", T, V, p), P)
        self.assertEqual(eos_call(eos, "entropy", P, T, V, p), p["S_0"])
        self.assertEqual(eos_call(eos, "molar_internal_energy", P, T, V, p), p["E_0"])
        self.assertEqual(
            eos_call(eos, "helmholtz_energy", P, T, V, p), p["E_0"] - T * p["S_0"]
        )
        self.assertFloatEqual(
            eos_call(eos, "grueneisen_parameter", P, T, V, p),
            p["grueneisen_0"],
            tol=2e-15,
        )
        self.assertFloatEqual(
            eos_call(eos, "isentropic_bulk_modulus_reuss", P, T, V, p),
            p["K_S"],
            tol=2e-15,
        )
        calculated_volume = eos_call(eos, "volume", P, T, p)
        self.assertFloatEqual(calculated_volume, V, tol=2e-14)
        self.assertLess(
            abs(eos_call(eos, "pressure", T, calculated_volume, p) - P), 0.01
        )

    def test_heat_capacity_and_figure5_expression_are_unchanged(self):
        p, eos = (parameter_dict(self.params), self.eos)
        for rho in [5000.0, 7019.0, 10000.0, 15000.0]:
            for T in [1000.0, 1811.0, 5000.0, 15000.0]:
                with self.subTest(rho=rho, T=T):
                    V = p["molar_mass"] / rho
                    r = p["V_0"] / V
                    A = p["a"][0] + p["a"][1] / r
                    B = p["b"][0] + p["b"][1] / r**2
                    Theta = p["Theta"][0] * r ** p["Theta"][1]
                    xi = p["xi_0"] * r ** (-0.6)
                    logistic = 1.0 / (1.0 + np.exp((r - p["F"][0]) / p["F"][1]))
                    lmda = logistic * (
                        p["lmda"][0] + p["lmda"][1] * r + p["lmda"][2] * r**0.4
                    )
                    expected = (
                        1.5 * p["n"] * burnman.constants.gas_constant
                        + A * T * T / (Theta * Theta + T * T)
                        + B * T**0.6
                        + (lmda * T + xi * p["theta"]) / (p["theta"] + T)
                    )
                    self.assertFloatEqual(
                        eos_call(eos, "molar_heat_capacity_v", 0.0, T, V, p),
                        expected,
                        tol=2e-15,
                    )

    def test_derivatives_with_perturbed_coefficients(self):
        p = native_copy(parameter_dict(self.params))
        p.update(
            a=[10.0, 5.0],
            b=[0.03, -0.02],
            Theta=[2300.0, 0.7],
            lmda=[-2.0, 4.0, 3.0],
            xi_0=20.0,
            F=[1.2, 0.3],
            theta=4000.0,
            grueneisen_n=-0.8,
        )
        T, V = (6500.0, 0.7 * p["V_0"])
        dV = 1e-05 * V
        eos = self.eos
        derivative = -(
            eos_call(eos, "helmholtz_energy", 0.0, T, V + dV, p)
            - eos_call(eos, "helmholtz_energy", 0.0, T, V - dV, p)
        ) / (2 * dV)
        self.assertFloatEqual(eos_call(eos, "pressure", T, V, p), derivative, tol=1e-08)
        derivative = (
            -V
            * (
                eos_call(eos, "pressure", T, V + dV, p)
                - eos_call(eos, "pressure", T, V - dV, p)
            )
            / (2 * dV)
        )
        self.assertFloatEqual(
            eos_call(eos, "isothermal_bulk_modulus_reuss", 0.0, T, V, p),
            derivative,
            tol=1e-08,
        )

    def test_volume_accuracy_is_independent_of_formula_unit_size(self):
        for scale in [0.001, 1.0, 1000.0]:
            p = native_copy(parameter_dict(self.params))
            for key in ["V_0", "E_0", "S_0", "molar_mass", "n", "xi_0"]:
                p[key] *= scale
            for key in ["a", "b", "lmda"]:
                p[key] = [value * scale for value in p[key]]
            p["grueneisen_prime"] /= scale
            for rho, T in [(7019.0, 1811.0), (10000.0, 5000.0), (13000.0, 7000.0)]:
                with self.subTest(scale=scale, rho=rho, T=T):
                    V = p["molar_mass"] / rho
                    P = eos_call(self.eos, "pressure", T, V, p)
                    calculated = eos_call(self.eos, "volume", P, T, p)
                    self.assertFloatEqual(calculated, V, tol=2e-14)
                    self.assertLess(
                        abs(eos_call(self.eos, "pressure", T, calculated, p) - P), 0.02
                    )

    def test_gibbs_uses_supplied_volume(self):
        p = parameter_dict(self.params)
        P, T, V = (130000000000.0, 5000.0, 0.8 * p["V_0"])
        self.assertFloatEqual(
            eos_call(self.eos, "gibbs_energy", P, T, V, p),
            eos_call(self.eos, "helmholtz_energy", 0.0, T, V, p) + P * V,
            tol=1e-14,
        )

    def test_all_public_properties_agree_with_mineral(self):
        eos, p = (self.eos, parameter_dict(self.params))
        P, T = (130000000000.0, 5000.0)
        self.iron.set_state(P, T)
        V = self.iron.molar_volume
        properties = {
            "helmholtz_energy": "molar_helmholtz",
            "gibbs_energy": "molar_gibbs",
            "molar_internal_energy": "molar_internal_energy",
            "enthalpy": "molar_enthalpy",
            "entropy": "molar_entropy",
            "molar_heat_capacity_v": "molar_heat_capacity_v",
            "molar_heat_capacity_p": "molar_heat_capacity_p",
            "grueneisen_parameter": "grueneisen_parameter",
            "isothermal_bulk_modulus_reuss": "isothermal_bulk_modulus_reuss",
            "isentropic_bulk_modulus_reuss": "isentropic_bulk_modulus_reuss",
            "thermal_expansivity": "thermal_expansivity",
            "shear_modulus": "shear_modulus",
        }
        for method, attribute in properties.items():
            with self.subTest(method=method):
                self.assertFloatEqual(
                    eos_function(eos, method)(P, T, V, p),
                    getattr(self.iron, attribute),
                    tol=2e-14,
                    tol_zero=1e-08,
                )

    def test_preserves_heat_capacity_based_grueneisen_trend(self):
        p = parameter_dict(self.params)
        T, V, dt = (p["T_0"], p["V_0"], 0.1)
        slope = (
            eos_call(self.eos, "grueneisen_parameter", 0.0, T + dt, V, p)
            - eos_call(self.eos, "grueneisen_parameter", 0.0, T - dt, V, p)
        ) / (2 * dt)
        self.assertFloatEqual(slope, 0.00010773006668962637, tol=1e-08)
        self.assertGreater(slope, 0.0)

    def test_burnman_consistency_checker(self):
        for P, T in [
            (100000.0, 1811.0),
            (130000000000.0, 5000.0),
            (320000000000.0, 7000.0),
        ]:
            with self.subTest(P=P, T=T):
                self.assertTrue(
                    check_eos_consistency(
                        self.iron, P=P, T=T, tol=1e-05, including_shear_properties=False
                    )
                )
