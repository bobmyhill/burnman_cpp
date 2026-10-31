# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_anharmonic.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, thermal_call
from burnman_cpp.eos.anharmonic_debye import AnharmonicDebye
from burnman_cpp.eos import anharmonic_thermal_models
from burnman_cpp.eos import debye_temperature_models
from burnman_cpp.eos import anharmonic_prefactor_models
from burnman_cpp.eos import debye
from burnman_cpp.constants import gas_constant

params = {
    "a_anh": 1.5,
    "V_0": 2e-06,
    "m_anh": 3.0,
    "Debye_0": 1000.0,
    "grueneisen_0": 1.2,
    "q_0": 1.1,
    "debye_temperature_model": debye_temperature_models.SLB(),
    "T_0": 300.0,
    "mu_anh": 0.2,
    "sigma_anh": 1.2,
}


class Anharmonic(BurnManTest):

    def test_pade_derivatives(self):
        pade = anharmonic_thermal_models.Pade()
        T = 2000.0
        S = thermal_call(pade, "nondimensional_entropy", T, params["Debye_0"])
        C_v = thermal_call(pade, "nondimensional_heat_capacity", T, params["Debye_0"])
        dFdTheta_ana = thermal_call(
            pade, "nondimensional_dhelmholtz_dTheta", T, params["Debye_0"]
        )
        dSdTheta_ana = thermal_call(
            pade, "nondimensional_dentropy_dTheta", T, params["Debye_0"]
        )
        d2FdTheta2_ana = thermal_call(
            pade, "nondimensional_d2helmholtz_dTheta2", T, params["Debye_0"]
        )
        dT = 0.01
        dTheta = 0.01
        S_num = -(
            thermal_call(
                pade, "nondimensional_helmholtz_energy", T + dT, params["Debye_0"]
            )
            - thermal_call(
                pade, "nondimensional_helmholtz_energy", T - dT, params["Debye_0"]
            )
        ) / (2.0 * dT)
        C_v_num = (
            T
            * (
                thermal_call(pade, "nondimensional_entropy", T + dT, params["Debye_0"])
                - thermal_call(
                    pade, "nondimensional_entropy", T - dT, params["Debye_0"]
                )
            )
            / (2.0 * dT)
        )
        dFdTheta_num = (
            thermal_call(
                pade, "nondimensional_helmholtz_energy", T, params["Debye_0"] + dTheta
            )
            - thermal_call(
                pade, "nondimensional_helmholtz_energy", T, params["Debye_0"] - dTheta
            )
        ) / (2.0 * dTheta)
        dSdTheta_num = (
            thermal_call(pade, "nondimensional_entropy", T, params["Debye_0"] + dTheta)
            - thermal_call(
                pade, "nondimensional_entropy", T, params["Debye_0"] - dTheta
            )
        ) / (2.0 * dTheta)
        d2FdTheta2_num = (
            thermal_call(
                pade, "nondimensional_helmholtz_energy", T, params["Debye_0"] + dTheta
            )
            - 2
            * thermal_call(
                pade, "nondimensional_helmholtz_energy", T, params["Debye_0"]
            )
            + thermal_call(
                pade, "nondimensional_helmholtz_energy", T, params["Debye_0"] - dTheta
            )
        ) / dTheta**2
        self.assertAlmostEqual(S, S_num, places=6)
        self.assertAlmostEqual(C_v, C_v_num, places=6)
        self.assertAlmostEqual(dFdTheta_ana, dFdTheta_num, places=6)
        self.assertAlmostEqual(dSdTheta_ana, dSdTheta_num, places=6)
        self.assertAlmostEqual(d2FdTheta2_ana, d2FdTheta2_num, places=6)

    def test_lognormal_derivatives(self):
        lognormal = anharmonic_thermal_models.LogNormal()
        T = 1500.0
        S = thermal_call(
            lognormal, "nondimensional_entropy", T, params["Debye_0"], params
        )
        C_v = thermal_call(
            lognormal, "nondimensional_heat_capacity", T, params["Debye_0"], params
        )
        dFdTheta_ana = thermal_call(
            lognormal, "nondimensional_dhelmholtz_dTheta", T, params["Debye_0"], params
        )
        dSdTheta_ana = thermal_call(
            lognormal, "nondimensional_dentropy_dTheta", T, params["Debye_0"], params
        )
        d2FdTheta2_ana = thermal_call(
            lognormal,
            "nondimensional_d2helmholtz_dTheta2",
            T,
            params["Debye_0"],
            params,
        )
        dT = 0.01
        dTheta = 0.01
        S_num = -(
            thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T + dT,
                params["Debye_0"],
                params,
            )
            - thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T - dT,
                params["Debye_0"],
                params,
            )
        ) / (2.0 * dT)
        C_v_num = (
            T
            * (
                thermal_call(
                    lognormal,
                    "nondimensional_entropy",
                    T + dT,
                    params["Debye_0"],
                    params,
                )
                - thermal_call(
                    lognormal,
                    "nondimensional_entropy",
                    T - dT,
                    params["Debye_0"],
                    params,
                )
            )
            / (2.0 * dT)
        )
        dFdTheta_num = (
            thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T,
                params["Debye_0"] + dTheta,
                params,
            )
            - thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T,
                params["Debye_0"] - dTheta,
                params,
            )
        ) / (2.0 * dTheta)
        dSdTheta_num = (
            thermal_call(
                lognormal,
                "nondimensional_entropy",
                T,
                params["Debye_0"] + dTheta,
                params,
            )
            - thermal_call(
                lognormal,
                "nondimensional_entropy",
                T,
                params["Debye_0"] - dTheta,
                params,
            )
        ) / (2.0 * dTheta)
        d2FdTheta2_num = (
            thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T,
                params["Debye_0"] + dTheta,
                params,
            )
            - 2
            * thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T,
                params["Debye_0"],
                params,
            )
            + thermal_call(
                lognormal,
                "nondimensional_helmholtz_energy",
                T,
                params["Debye_0"] - dTheta,
                params,
            )
        ) / dTheta**2
        self.assertAlmostEqual(S, S_num, places=6)
        self.assertAlmostEqual(C_v, C_v_num, places=6)
        self.assertAlmostEqual(dFdTheta_ana, dFdTheta_num, places=6)
        self.assertAlmostEqual(dSdTheta_ana, dSdTheta_num, places=6)
        self.assertAlmostEqual(d2FdTheta2_ana, d2FdTheta2_num, places=6)

    def test_derivatives_lognormal(self):
        thermal_model = anharmonic_thermal_models.LogNormal()
        prefactor_model = anharmonic_prefactor_models.PowerLaw()
        params["anharmonic_thermal_model"] = thermal_model
        params["anharmonic_prefactor_model"] = prefactor_model
        V_0 = params["V_0"]
        V = V_0 * 0.9
        T = 1500.0
        S = AnharmonicDebye.entropy(T, V, params)
        C_v = AnharmonicDebye.heat_capacity_v(T, V, params)
        P = AnharmonicDebye.pressure(T, V, params)
        K_T = AnharmonicDebye.isothermal_bulk_modulus(T, V, params)
        alphaK_T = AnharmonicDebye.dSdV(T, V, params)
        dV = 1e-11
        dT = 0.01
        dFdV = (
            AnharmonicDebye.helmholtz_energy(T, V + dV, params)
            - AnharmonicDebye.helmholtz_energy(T, V - dV, params)
        ) / (2.0 * dV)
        dFdT = (
            AnharmonicDebye.helmholtz_energy(T + dT, V, params)
            - AnharmonicDebye.helmholtz_energy(T - dT, V, params)
        ) / (2.0 * dT)
        dSdV = (
            AnharmonicDebye.entropy(T, V + dV, params)
            - AnharmonicDebye.entropy(T, V - dV, params)
        ) / (2.0 * dV)
        dSdT = (
            AnharmonicDebye.entropy(T + dT, V, params)
            - AnharmonicDebye.entropy(T - dT, V, params)
        ) / (2.0 * dT)
        dPdV = (
            AnharmonicDebye.pressure(T, V + dV, params)
            - AnharmonicDebye.pressure(T, V - dV, params)
        ) / (2.0 * dV)
        dPdT = (
            AnharmonicDebye.pressure(T + dT, V, params)
            - AnharmonicDebye.pressure(T - dT, V, params)
        ) / (2.0 * dT)
        self.assertAlmostEqual(S, -dFdT, places=6)
        self.assertAlmostEqual(C_v, T * dSdT, places=6)
        self.assertAlmostEqual(P / 100000000.0, -dFdV / 100000000.0, places=6)
        self.assertAlmostEqual(K_T / 100000000.0, -V * dPdV / 100000000.0, places=6)
        self.assertAlmostEqual(alphaK_T / 1000.0, dSdV / 1000.0, places=6)
        self.assertAlmostEqual(alphaK_T / 1000.0, dPdT / 1000.0, places=6)

    def test_derivatives_pade(self):
        thermal_model = anharmonic_thermal_models.Pade()
        prefactor_model = anharmonic_prefactor_models.PowerLaw()
        params["anharmonic_thermal_model"] = thermal_model
        params["anharmonic_prefactor_model"] = prefactor_model
        V_0 = params["V_0"]
        V = V_0 * 0.9
        T = 1500.0
        S = AnharmonicDebye.entropy(T, V, params)
        C_v = AnharmonicDebye.heat_capacity_v(T, V, params)
        P = AnharmonicDebye.pressure(T, V, params)
        K_T = AnharmonicDebye.isothermal_bulk_modulus(T, V, params)
        alphaK_T = AnharmonicDebye.dSdV(T, V, params)
        dV = 1e-11
        dT = 0.01
        dFdV = (
            AnharmonicDebye.helmholtz_energy(T, V + dV, params)
            - AnharmonicDebye.helmholtz_energy(T, V - dV, params)
        ) / (2.0 * dV)
        dFdT = (
            AnharmonicDebye.helmholtz_energy(T + dT, V, params)
            - AnharmonicDebye.helmholtz_energy(T - dT, V, params)
        ) / (2.0 * dT)
        dSdV = (
            AnharmonicDebye.entropy(T, V + dV, params)
            - AnharmonicDebye.entropy(T, V - dV, params)
        ) / (2.0 * dV)
        dSdT = (
            AnharmonicDebye.entropy(T + dT, V, params)
            - AnharmonicDebye.entropy(T - dT, V, params)
        ) / (2.0 * dT)
        dPdV = (
            AnharmonicDebye.pressure(T, V + dV, params)
            - AnharmonicDebye.pressure(T, V - dV, params)
        ) / (2.0 * dV)
        dPdT = (
            AnharmonicDebye.pressure(T + dT, V, params)
            - AnharmonicDebye.pressure(T - dT, V, params)
        ) / (2.0 * dT)
        self.assertAlmostEqual(S, -dFdT, places=6)
        self.assertAlmostEqual(C_v, T * dSdT, places=6)
        self.assertAlmostEqual(P / 100000000.0, -dFdV / 100000000.0, places=6)
        self.assertAlmostEqual(K_T / 100000000.0, -V * dPdV / 100000000.0, places=6)
        self.assertAlmostEqual(alphaK_T / 1000.0, dSdV / 1000.0, places=6)
        self.assertAlmostEqual(alphaK_T / 1000.0, dPdT / 1000.0, places=6)

    def test_standard_entropy_pade(self):
        thermal_model = anharmonic_thermal_models.Pade()
        prefactor_model = anharmonic_prefactor_models.PowerLaw()
        params["anharmonic_thermal_model"] = thermal_model
        params["anharmonic_prefactor_model"] = prefactor_model
        V = params["V_0"]
        T = thermal_call(params["debye_temperature_model"], "value", 1.0, params)
        self.assertAlmostEqual(T, params["Debye_0"], places=6)
        S = AnharmonicDebye.entropy(T, V, params) / params["a_anh"]
        S1 = -debye.thermal_energy(1.0, 1.0, 1.0) / gas_constant / 3.0
        self.assertAlmostEqual(S, S1, places=4)
        self.assertAlmostEqual(S, -0.6744, places=4)

    def test_prefactor_power_law(self):
        prefactor_model = anharmonic_prefactor_models.PowerLaw()
        model_params = {"a_anh": 1.1, "m_anh": 5.0}
        Vrel = 0.8
        dVrel = 1e-06
        dAdVrel_num = (
            thermal_call(prefactor_model, "value", Vrel + dVrel, model_params)
            - thermal_call(prefactor_model, "value", Vrel - dVrel, model_params)
        ) / (2.0 * dVrel)
        dAdVrel_analytical = thermal_call(prefactor_model, "dVrel", Vrel, model_params)
        self.assertAlmostEqual(dAdVrel_num, dAdVrel_analytical, places=6)
        d2AdVrel2_num = (
            thermal_call(prefactor_model, "dVrel", Vrel + dVrel, model_params)
            - thermal_call(prefactor_model, "dVrel", Vrel - dVrel, model_params)
        ) / (2.0 * dVrel)
        d2AdVrel2_analytical = thermal_call(
            prefactor_model, "dVrel2", Vrel, model_params
        )
        self.assertAlmostEqual(d2AdVrel2_num, d2AdVrel2_analytical, places=6)

    def test_prefactor_sigmoid(self):
        prefactor_model = anharmonic_prefactor_models.Sigmoid()
        model_params = {"a_anh": 1.1, "b_anh": 1.2, "c_anh": 1.3}
        Vrel = 0.8
        dVrel = 1e-06
        dAdVrel_num = (
            thermal_call(prefactor_model, "value", Vrel + dVrel, model_params)
            - thermal_call(prefactor_model, "value", Vrel - dVrel, model_params)
        ) / (2.0 * dVrel)
        dAdVrel_analytical = thermal_call(prefactor_model, "dVrel", Vrel, model_params)
        self.assertAlmostEqual(dAdVrel_num, dAdVrel_analytical, places=6)
        d2AdVrel2_num = (
            thermal_call(prefactor_model, "dVrel", Vrel + dVrel, model_params)
            - thermal_call(prefactor_model, "dVrel", Vrel - dVrel, model_params)
        ) / (2.0 * dVrel)
        d2AdVrel2_analytical = thermal_call(
            prefactor_model, "dVrel2", Vrel, model_params
        )
        self.assertAlmostEqual(d2AdVrel2_num, d2AdVrel2_analytical, places=6)
