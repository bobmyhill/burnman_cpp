# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_eos.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict, eos_call
import warnings
import burnman_cpp as burnman
from burnman_cpp import minerals
from burnman_cpp.eos import debye
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
