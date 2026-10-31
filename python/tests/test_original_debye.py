# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_debye.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict
import burnman_cpp as burnman


class mypericlase(burnman.Mineral):
    """
    Stixrude & Lithgow-Bertelloni 2005 and references therein
    """

    def __init__(self):
        params = {
            "equation_of_state": "slb3",
            "V_0": 1.124e-05,
            "K_0": 161000000000.0,
            "Kprime_0": 3.8,
            "G_0": 131000000000.0,
            "Gprime_0": 2.1,
            "molar_mass": 0.0403,
            "n": 2,
            "Debye_0": 773.0,
            "grueneisen_0": 1.5,
            "q_0": 1.5,
            "eta_s_0": 2.8,
        }
        params.setdefault("formula", {"Mg": 1.0, "O": 1.0})
        params.setdefault("name", "mypericlase")
        burnman.Mineral.__init__(self, params)


class Debye(BurnManTest):

    def test_same_debye(self):
        x = 300.0
        test_debye = burnman.eos.debye.debye_fn(x)
        test_debye_cheb = burnman.eos.debye.debye_fn_cheb(x)
        self.assertFloatEqual(test_debye, test_debye_cheb)

    def test_return_zero(self):
        rock = mypericlase()
        x = 0.0
        test_helmholtz = burnman.eos.debye.helmholtz_energy(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_helmholtz, 0.0)
        test_heat_capacity_v = burnman.eos.debye.molar_heat_capacity_v(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_heat_capacity_v, 0.0)
        test_thermal_energy = burnman.eos.debye.thermal_energy(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_thermal_energy, 0.0)

    def test_small(self):
        rock = mypericlase()
        x = 1e-16
        test_helmholtz = burnman.eos.debye.helmholtz_energy(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_helmholtz, 0.0)
        test_heat_capacity_v = burnman.eos.debye.molar_heat_capacity_v(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_heat_capacity_v, 0.0)
        test_thermal_energy = burnman.eos.debye.thermal_energy(
            x, parameter_dict(rock.params)["Debye_0"], parameter_dict(rock.params)["n"]
        )
        self.assertFloatEqual(test_thermal_energy, 0.0)

    def test_theta_derivatives(self):
        rock = mypericlase()
        T = 300.0
        debye_T = parameter_dict(rock.params)["Debye_0"]
        n = parameter_dict(rock.params)["n"]
        delta = 1e-06
        dF_dTheta_num = (
            burnman.eos.debye.helmholtz_energy(T, debye_T + delta, n)
            - burnman.eos.debye.helmholtz_energy(T, debye_T - delta, n)
        ) / (2 * delta)
        d2F_dTheta2_num = (
            burnman.eos.debye.dhelmholtz_dTheta(T, debye_T + delta, n)
            - burnman.eos.debye.dhelmholtz_dTheta(T, debye_T - delta, n)
        ) / (2 * delta)
        dS_dTheta_num = (
            burnman.eos.debye.entropy(T, debye_T + delta, n)
            - burnman.eos.debye.entropy(T, debye_T - delta, n)
        ) / (2 * delta)
        dF_dTheta_analytical = burnman.eos.debye.dhelmholtz_dTheta(T, debye_T, n)
        d2F_dTheta2_analytical = burnman.eos.debye.d2helmholtz_dTheta2(T, debye_T, n)
        dS_dTheta_analytical = burnman.eos.debye.dentropy_dTheta(T, debye_T, n)
        self.assertFloatEqual(dF_dTheta_num, dF_dTheta_analytical)
        self.assertFloatEqual(d2F_dTheta2_num, d2F_dTheta2_analytical)
        self.assertFloatEqual(dS_dTheta_num, dS_dTheta_analytical)
