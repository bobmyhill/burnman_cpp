# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_eos.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict, eos_call, mineral_eos
import warnings
import numpy as np
import burnman_cpp as burnman
from burnman_cpp import minerals
from burnman_cpp.eos import debye
from burnman_cpp.tools.eos import check_eos_consistency
from burnman_cpp.utils.chemistry import dictionarize_formula, formula_mass
import logging

logging.basicConfig(level=logging.CRITICAL)


class Fe_Dewaele(burnman.Mineral):
    """
    Dewaele et al., 2006, Physical Review Letters
    """

    def __init__(self):
        params = {
            "equation_of_state": "vinet",
            "P_0": 0.0,
            "V_0": 6.75e-06,
            "K_0": 163400000000.0,
            "Kprime_0": 5.38,
            "molar_mass": 0.055845,
            "n": 1,
        }
        params.setdefault("name", "Fe_Dewaele")
        burnman.Mineral.__init__(self, params)


class Liquid_Fe_Anderson(burnman.Mineral):
    """
    Anderson & Ahrens, 1994 JGR
    """

    def __init__(self):
        params = {
            "equation_of_state": "bm4",
            "P_0": 0.0,
            "V_0": 7.95626e-06,
            "K_0": 109700000000.0,
            "Kprime_0": 4.66,
            "Kprime_prime_0": -4.3e-11,
            "molar_mass": 0.055845,
        }
        params.setdefault("name", "Liquid_Fe_Anderson")
        burnman.Mineral.__init__(self, params)


class outer_core_rkprime(burnman.Mineral):
    """
    Stacey and Davis, 2004 PEPI (Table 5)
    """

    def __init__(self):
        params = {
            "equation_of_state": "rkprime",
            "P_0": 0.0,
            "V_0": 0.055845 / 6562.54,
            "K_0": 124553000000.0,
            "Kprime_0": 4.9599,
            "Kprime_inf": 3.0,
            "molar_mass": 0.055845,
        }
        params.setdefault("name", "outer_core_rkprime")
        burnman.Mineral.__init__(self, params)


class periclase_morse(burnman.Mineral):
    """
    Periclase parameters from SLB dataset (which uses BM3)
    """

    def __init__(self):
        formula = "MgO"
        formula = dictionarize_formula(formula)
        params = {
            "name": "Periclase",
            "formula": formula,
            "equation_of_state": "morse",
            "P_0": 0.0,
            "V_0": 1.1244e-05,
            "K_0": 161383600000.0,
            "Kprime_0": 3.84045,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        params.setdefault("name", "periclase_morse")
        burnman.Mineral.__init__(self, params)


class alpha_quartz(burnman.Mineral):

    def __init__(self):
        formula = "SiO2"
        formula = dictionarize_formula(formula)
        params = {
            "name": "alpha quartz",
            "formula": formula,
            "equation_of_state": "slb3",
            "F_0": -2313.86317911,
            "V_0": 2.2761615699999998e-05,
            "K_0": 74803223700.0,
            "Kprime_0": 7.01334832,
            "Debye_0": 1069.02139,
            "grueneisen_0": -0.0615759576,
            "q_0": 10.940017,
            "G_0": 44856170000.0,
            "Gprime_0": 0.95315,
            "eta_s_0": 2.36469,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        property_modifiers = [
            ["einstein", {"Theta_0": 510.158911, "Cv_inf": 39.9366924855}],
            ["einstein", {"Theta_0": 801.41137578, "Cv_inf": -52.9869089}],
            ["einstein", {"Theta_0": 2109.4459429999997, "Cv_inf": 13.0502164145}],
        ]
        params.setdefault("name", "alpha_quartz")
        burnman.Mineral.__init__(self, params)
        self.set_property_modifiers(property_modifiers)


class eos(BurnManTest):

    def test_reference_values_vinet(self):
        rock = Fe_Dewaele()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.make_eos("vinet")
        Volume_test = eos_call(
            eos, "volume", pressure, temperature, parameter_dict(rock.params)
        )
        self.assertFloatEqual(Volume_test, parameter_dict(rock.params)["V_0"])
        Kt_test = eos_call(
            eos,
            "isothermal_bulk_modulus_reuss",
            pressure,
            300.0,
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(Kt_test, parameter_dict(rock.params)["K_0"])
        Density_test = eos_call(
            eos,
            "density",
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(
            Density_test,
            parameter_dict(rock.params)["molar_mass"]
            / parameter_dict(rock.params)["V_0"],
        )

    def test_reference_values_bm4(self):
        rock = Liquid_Fe_Anderson()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.make_eos("bm4")
        Volume_test = eos_call(
            eos, "volume", pressure, temperature, parameter_dict(rock.params)
        )
        self.assertFloatEqual(Volume_test, parameter_dict(rock.params)["V_0"])
        Kt_test = eos_call(
            eos,
            "isothermal_bulk_modulus_reuss",
            pressure,
            300.0,
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(Kt_test, parameter_dict(rock.params)["K_0"])
        Density_test = eos_call(
            eos,
            "density",
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(
            Density_test,
            parameter_dict(rock.params)["molar_mass"]
            / parameter_dict(rock.params)["V_0"],
        )

    def test_reference_values_morse(self):
        rock = periclase_morse()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.make_eos("morse")
        Volume_test = eos_call(
            eos, "volume", pressure, temperature, parameter_dict(rock.params)
        )
        self.assertFloatEqual(Volume_test, parameter_dict(rock.params)["V_0"])
        Kt_test = eos_call(
            eos,
            "isothermal_bulk_modulus_reuss",
            pressure,
            300.0,
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(Kt_test, parameter_dict(rock.params)["K_0"])
        Density_test = eos_call(
            eos,
            "density",
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(
            Density_test,
            parameter_dict(rock.params)["molar_mass"]
            / parameter_dict(rock.params)["V_0"],
        )

    def test_reference_values_rkprime(self):
        rock = outer_core_rkprime()
        pressure = 0.0
        temperature = 300.0
        eos = burnman.make_eos("rkprime")
        Volume_test = eos_call(
            eos, "volume", pressure, temperature, parameter_dict(rock.params)
        )
        self.assertFloatEqual(Volume_test, parameter_dict(rock.params)["V_0"])
        Kt_test = eos_call(
            eos,
            "isothermal_bulk_modulus_reuss",
            pressure,
            300.0,
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(Kt_test, parameter_dict(rock.params)["K_0"])
        Density_test = eos_call(
            eos,
            "density",
            parameter_dict(rock.params)["V_0"],
            parameter_dict(rock.params),
        )
        self.assertFloatEqual(
            Density_test,
            parameter_dict(rock.params)["molar_mass"]
            / parameter_dict(rock.params)["V_0"],
        )

    def test_reference_values_aa(self):
        m = minerals.other.liquid_iron()
        pressure = parameter_dict(m.params)["P_0"]
        temperature = parameter_dict(m.params)["T_0"]
        rho0 = parameter_dict(m.params)["molar_mass"] / parameter_dict(m.params)["V_0"]
        m.set_state(pressure, temperature)
        self.assertFloatEqual(m.molar_volume, parameter_dict(m.params)["V_0"])
        Ks = m.isentropic_bulk_modulus_reuss
        self.assertFloatEqual(Ks, parameter_dict(m.params)["K_S"])
        self.assertFloatEqual(m.density, rho0)


class test_eos_validation(BurnManTest):

    def _cold_mineral(self, method, reference=False):
        params = {
            "equation_of_state": method,
            # The original loop carries these defaults from its initial BM3.
            "F_0": 0.0,
            "P_0": 0.0,
            "V_0": 7.95626e-6,
            "K_0": 109.7e9,
            "Kprime_0": 4.66,
            "Kprime_inf": 3.00,
            "Kprime_prime_0": -0.043e-9,
            "Kdprime_0": -4.66 / 100.0e9,
            "molar_mass": 0.055845,
        }
        # Upstream BM4 consumes Kprime_prime_0; MT and SPOCK consume Kdprime_0.
        # The native representation has one field, so select the model's value.
        if method == "bm4":
            params["Kdprime_0"] = params["Kprime_prime_0"]
        del params["Kprime_prime_0"]
        if reference:
            params.update(F_0=1000.0, T_0=100.0, P_0=1.0e10)
        return burnman.Mineral(params)

    def test_reference_energies(self):
        eoses = ["bm3", "bm4", "vinet", "mt", "morse", "rkprime"]
        energies = []
        for eos in eoses:
            m = self._cold_mineral(eos, reference=True)
            m.set_state(m.params.P_0, m.params.T_0)
            energies.append(m.molar_helmholtz)
        self.assertArraysAlmostEqual(energies, [m.params.F_0] * len(energies))

    def test_energy_derivatives(self):
        eoses = [
            "bm3",
            "bm4",
            "vinet",
            "mt",
            "morse",
            "rkprime",
            "macaw",
            "spock",
            "murnaghan",
        ]
        calculated = []
        derivative = []
        for eos in eoses:
            m = self._cold_mineral(eos)
            P_0 = 10.0e9
            dP = 10000.0
            pressures = np.array([P_0 - 0.5 * dP, P_0, P_0 + 0.5 * dP])
            temperatures = np.zeros(3)
            E, G, H, A, V, KT = m.evaluate(
                ["molar_internal_energy", "gibbs", "H", "helmholtz", "V", "K_T"],
                pressures,
                temperatures,
            )
            calculated.append(P_0)
            derivative.append(-(E[2] - E[0]) / (V[2] - V[0]))
            calculated.append(V[1])
            derivative.append((G[2] - G[0]) / dP)
            calculated.append(-V[1] / KT[1])
            derivative.append((V[2] - V[0]) / dP)
        self.assertArraysAlmostEqual(calculated, derivative)

    def test_no_shear_error(self):

        class mymineralwithoutshear(burnman.Mineral):

            def __init__(self):
                params = {
                    "equation_of_state": "slb3",
                    "V_0": 1.124e-05,
                    "K_0": 161000000000.0,
                    "Kprime_0": 3.8,
                    "molar_mass": 0.0403,
                    "n": 2,
                    "Debye_0": 773.0,
                    "grueneisen_0": 1.5,
                    "q_0": 1.5,
                    "eta_s_0": 2.8,
                }
                params.setdefault("name", "mymineralwithoutshear")
                burnman.Mineral.__init__(self, params)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shearless = mymineralwithoutshear()
            if len(w) != 0:
                self.fail("Caught unexpected warning: " + str(w[-1]))
            try:
                x = parameter_dict(shearless.params)["G_0"]
                y = parameter_dict(shearless.params)["Gprime_0"]
                z = parameter_dict(shearless.params)["F_0"]
                z = parameter_dict(shearless.params)["eta_s_0"]
            except KeyError:
                self.fail("Parameter padding failed in validation")
                pass

    def test_debye(self):
        T = 500.0
        debye_T = 200.0
        n = 3.0
        dT = 0.001
        Cv = debye.molar_heat_capacity_v(T, debye_T, n)
        S = debye.entropy(T, debye_T, n)
        Cv2 = (
            debye.thermal_energy(T + dT / 2.0, debye_T, n)
            - debye.thermal_energy(T - dT / 2.0, debye_T, n)
        ) / dT
        S2 = (
            -(
                debye.helmholtz_energy(T + dT / 2.0, debye_T, n)
                - debye.helmholtz_energy(T - dT / 2.0, debye_T, n)
            )
            / dT
        )
        self.assertAlmostEqual(Cv, Cv2)
        self.assertAlmostEqual(S, S2)

    def test_pressure_finding_SLB(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = alpha_quartz()
            P = 100000.0
            T = 1000.0
            m.set_state(P, T)
            V = m.molar_volume
            m.set_state_with_volume(V, T)
        self.assertAlmostEqual(m.pressure / 100000.0, P / 100000.0, places=3)

    def test_pressure_finding_HP(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = minerals.HP_2011_ds62.mt()
            P = 1000000000.0
            T = 1000.0
            m.set_state(P, T)
            V = m.molar_volume
            m.set_state(P * 1.01, T)
            m.set_state_with_volume(V, T)
        self.assertAlmostEqual(m.pressure / 100000.0, P / 100000.0, places=3)


class test_cold_eos_reference_pressure(BurnManTest):
    def setUp(self):
        self.eoses = ["bm3shear2", "bm3", "bm4", "vinet", "morse", "spock"]
        self.params = {
            "F_0": 1000.0,
            "T_0": 100.0,
            "V_0": 7.95626e-6,
            "K_0": 109.7e9,
            "Kprime_0": 4.66,
            "Kprime_inf": 3.0,
            "Kprime_prime_0": -0.043e-9,
            "Kdprime_0": -4.66 / 100.0e9,
            "molar_mass": 0.055845,
        }

    def _mineral(self, eos, reference_pressure):
        # Native BM2 uses the BM3 bulk curve with second-order shear.
        params = dict(
            self.params,
            equation_of_state="bm2" if eos == "bm3shear2" else eos,
            P_0=reference_pressure,
        )
        if eos == "bm4":
            params["Kdprime_0"] = params["Kprime_prime_0"]
        del params["Kprime_prime_0"]
        return burnman.Mineral(params)

    def _helmholtz(self, m, volume):
        params, eos = m.params, mineral_eos(m)
        T = params.T_0
        P = eos.compute_pressure(T, volume, params)
        return eos.compute_gibbs_free_energy(P, T, volume, params) - P * volume

    def test_reference_pressure_work(self):
        for eos in self.eoses:
            zero = self._mineral(eos, 0.0)
            for P_0 in [1.0e5, 2.5e9, 1.0e10]:
                m = self._mineral(eos, P_0)
                for ratio in [0.8, 1.0, 1.1]:
                    with self.subTest(eos=eos, P_0=P_0, ratio=ratio):
                        V = ratio * self.params["V_0"]
                        self.assertFloatEqual(
                            self._helmholtz(m, V) - self._helmholtz(zero, V),
                            P_0 * (self.params["V_0"] - V),
                            tol=1.0e-10,
                            tol_zero=1.0e-8,
                        )

    def test_reference_energies(self):
        for eos in self.eoses:
            for P_0 in [0.0, 1.0e5, 2.5e9, 1.0e10]:
                with self.subTest(eos=eos, P_0=P_0):
                    m = self._mineral(eos, P_0)
                    self.assertFloatEqual(
                        self._helmholtz(m, self.params["V_0"]),
                        self.params["F_0"],
                        tol=1.0e-12,
                        tol_zero=1.0e-8,
                    )
                    m.set_state(P_0, self.params["T_0"])
                    self.assertFloatEqual(m.molar_helmholtz, self.params["F_0"])

    def test_eos_consistency(self):
        for eos in self.eoses:
            for P_0 in [1.0e5, 2.5e9, 1.0e10]:
                m = self._mineral(eos, P_0)
                for ratio in [0.8, 0.9, 1.0, 1.1]:
                    with self.subTest(eos=eos, P_0=P_0, ratio=ratio):
                        V = ratio * self.params["V_0"]
                        P = mineral_eos(m).compute_pressure(
                            self.params["T_0"], V, m.params
                        )
                        self.assertTrue(
                            check_eos_consistency(
                                m,
                                P=P,
                                T=self.params["T_0"],
                                including_shear_properties=False,
                            )
                        )

    def test_volume_resolution_at_reference_pressure(self):
        for eos in self.eoses:
            for P_0 in [0.0, 2.5e9, 1.0e10]:
                for scale in [1.0e-3, 1.0, 1.0e3]:
                    with self.subTest(eos=eos, P_0=P_0, scale=scale):
                        m = self._mineral(eos, P_0)
                        params, model = m.params, mineral_eos(m)
                        params.V_0 *= scale
                        V_0, K_0 = params.V_0, params.K_0
                        volumes = []
                        for dP in [-1000.0, 0.0, 1000.0]:
                            V = model.compute_volume(
                                P_0 + dP, self.params["T_0"], params
                            )
                            P = model.compute_pressure(self.params["T_0"], V, params)
                            self.assertFloatEqual(P, P_0 + dP, tol=0.0, tol_zero=1.0e-3)
                            self.assertFloatEqual(V / V_0, 1.0 - dP / K_0, tol=1.0e-14)
                            volumes.append(V)
                        self.assertGreater(volumes[0], volumes[1])
                        self.assertGreater(volumes[1], volumes[2])
                        self.assertFloatEqual(
                            -V_0 * 2000.0 / (volumes[2] - volumes[0]), K_0, tol=2.0e-7
                        )

    def test_spock_pressure_at_small_strain(self):
        for scale in [1.0e-3, 1.0, 1.0e3]:
            m = self._mineral("spock", 0.0)
            params, model = m.params, mineral_eos(m)
            params.V_0 *= scale
            for ratio in [1.0 - 1.0e-8, 1.0 + 1.0e-8, 1.0 - 1.0e-12, 1.0 + 1.0e-12]:
                with self.subTest(scale=scale, ratio=ratio):
                    V = params.V_0 * ratio
                    strain = V / params.V_0 - 1.0
                    P = model.compute_pressure(self.params["T_0"], V, params)
                    self.assertFloatEqual(
                        P, -params.K_0 * strain, tol=1.0e-7, tol_zero=0.0
                    )
