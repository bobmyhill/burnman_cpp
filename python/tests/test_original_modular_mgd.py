# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_modular_mgd.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    parameter_dict,
    make_assemblage,
    thermal_call,
)
from burnman_cpp import Mineral
from burnman_cpp.minerals.SLB_2011 import forsterite as forsterite_slb
from burnman_cpp.minerals.SLB_2024 import fea as alpha_iron_slb
from burnman_cpp.utils.chemistry import dictionarize_formula, formula_mass
from burnman_cpp import make_eos as create
from burnman_cpp.tools.eos import check_eos_consistency
from burnman_cpp.eos import debye_temperature_models
from burnman_cpp.eos import anharmonic_thermal_models
from burnman_cpp.eos import anharmonic_prefactor_models

formula = "Mg2SiO4"
formula = dictionarize_formula(formula)
mineral_params = {
    "name": "Forsterite",
    "formula": formula,
    "equation_of_state": "modular_mgd",
    "F_0": -2055403.0,
    "V_0": 4.3603e-05,
    "K_0": 127955500000.0,
    "Kprime_0": 4.21796,
    "T_0": 298.15,
    "n": sum(formula.values()),
    "molar_mass": formula_mass(formula),
}


class ModularMGD(BurnManTest):

    def test_setup(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        _ = Mineral(params)

    def test_SLB_derivatives(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        t = debye_temperature_models.SLB()
        dVrel = 1e-06
        Vrel = 0.7
        dtdVrel_numerical = (
            thermal_call(t, "value", Vrel + dVrel, params)
            - thermal_call(t, "value", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        dtdVrel_analytical = thermal_call(t, "dVrel", Vrel, params)
        d2tdVrel2_numerical = (
            thermal_call(t, "dVrel", Vrel + dVrel, params)
            - thermal_call(t, "dVrel", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        d2tdVrel2_analytical = thermal_call(t, "dVrel2", Vrel, params)
        self.assertAlmostEqual(dtdVrel_numerical, dtdVrel_analytical, places=6)
        self.assertAlmostEqual(d2tdVrel2_numerical, d2tdVrel2_analytical, places=6)

    def test_PLG_derivatives(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.PowerLawGamma()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.0
        params["c_1"] = 0.2
        params["c_2"] = 0.1
        params["q_1"] = 1.0
        params["q_2"] = 2.0
        t = debye_temperature_models.PowerLawGamma()
        dVrel = 1e-06
        Vrel = 0.7
        dtdVrel_numerical = (
            thermal_call(t, "value", Vrel + dVrel, params)
            - thermal_call(t, "value", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        dtdVrel_analytical = thermal_call(t, "dVrel", Vrel, params)
        d2tdVrel2_numerical = (
            thermal_call(t, "dVrel", Vrel + dVrel, params)
            - thermal_call(t, "dVrel", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        d2tdVrel2_analytical = thermal_call(t, "dVrel2", Vrel, params)
        self.assertAlmostEqual(dtdVrel_numerical, dtdVrel_analytical, places=6)
        self.assertAlmostEqual(d2tdVrel2_numerical, d2tdVrel2_analytical, places=6)

    def test_PLGS_derivatives(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = (
            debye_temperature_models.PowerLawGammaSimple()
        )
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        t = debye_temperature_models.PowerLawGammaSimple()
        dVrel = 1e-06
        Vrel = 0.7
        dtdVrel_numerical = (
            thermal_call(t, "value", Vrel + dVrel, params)
            - thermal_call(t, "value", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        dtdVrel_analytical = thermal_call(t, "dVrel", Vrel, params)
        d2tdVrel2_numerical = (
            thermal_call(t, "dVrel", Vrel + dVrel, params)
            - thermal_call(t, "dVrel", Vrel - dVrel, params)
        ) / (2.0 * dVrel)
        d2tdVrel2_analytical = thermal_call(t, "dVrel2", Vrel, params)
        self.assertAlmostEqual(dtdVrel_numerical, dtdVrel_analytical, places=6)
        self.assertAlmostEqual(d2tdVrel2_numerical, d2tdVrel2_analytical, places=6)

    def test_check_SLB_consistency(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        m = Mineral(params)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)

    def test_check_PLG_consistency(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.PowerLawGamma()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.0
        params["c_1"] = 0.2
        params["c_2"] = 0.1
        params["q_1"] = 1.0
        params["q_2"] = 2.0
        m = Mineral(params)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)

    def test_check_PLGS_consistency(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = (
            debye_temperature_models.PowerLawGammaSimple()
        )
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.0
        params["q_0"] = 1.0
        m = Mineral(params)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)

    def test_SLB_grueneisen(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m = Mineral(params)
        m.set_state(100000.0, 298.15)
        self.assertAlmostEqual(m.grueneisen_parameter, params["grueneisen_0"], places=6)

    def test_PLG_grueneisen(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.PowerLawGamma()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.0
        params["c_1"] = 0.2
        params["c_2"] = 0.1
        params["q_1"] = 1.0
        params["q_2"] = 2.0
        params["P_0"] = 100000.0
        m = Mineral(params)
        m.set_state(100000.0, 298.15)
        self.assertAlmostEqual(m.grueneisen_parameter, params["grueneisen_0"], places=6)

    def test_PLGS_grueneisen(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = (
            debye_temperature_models.PowerLawGammaSimple()
        )
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m = Mineral(params)
        m.set_state(100000.0, 298.15)
        self.assertAlmostEqual(m.grueneisen_parameter, params["grueneisen_0"], places=6)

    def test_SLB_electronic_contribution(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m_without_component = Mineral(params)
        params2 = params.copy()
        params2["bel_0"] = 0.005
        params2["gel"] = 1.5
        m = Mineral(params2)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)
        m.set_state(2000000000.0, 2000.0)
        m_without_component.set_state(2000000000.0, 2000.0)
        self.assertFalse(m_without_component.molar_entropy == m.molar_entropy)

    def test_SLB_pade_anharmonic_contribution(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m_without_component = Mineral(params)
        params2 = params.copy()
        params2["a_anh"] = 1000.0
        params2["m_anh"] = 3.0
        params2["anharmonic_prefactor_model"] = anharmonic_prefactor_models.PowerLaw()
        params2["anharmonic_thermal_model"] = anharmonic_thermal_models.Pade()
        m = Mineral(params2)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)
        m.set_state(2000000000.0, 2000.0)
        m_without_component.set_state(2000000000.0, 2000.0)
        self.assertFalse(m_without_component.molar_entropy == m.molar_entropy)

    def test_SLB_lognormal_anharmonic_contribution(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m_without_component = Mineral(params)
        params2 = params.copy()
        params2["a_anh"] = 1.0
        params2["m_anh"] = 3.0
        params2["sigma_anh"] = 1.2
        params2["mu_anh"] = 0.2
        params2["anharmonic_prefactor_model"] = anharmonic_prefactor_models.PowerLaw()
        params2["anharmonic_thermal_model"] = anharmonic_thermal_models.LogNormal()
        m = Mineral(params2)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)
        m.set_state(2000000000.0, 2000.0)
        m_without_component.set_state(2000000000.0, 2000.0)
        self.assertFalse(m_without_component.molar_entropy == m.molar_entropy)

    def test_SLB_inbuilt_anharmonic_contribution(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        m_without_component = Mineral(params)
        params2 = params.copy()
        params2["equation_of_state"] = "modular_mgd_with_anharmonicity"
        params2["c_anh"] = 0.05
        m = Mineral(params2)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)
        m.set_state(2000000000.0, 2000.0)
        m_without_component.set_state(2000000000.0, 2000.0)
        self.assertFalse(m_without_component.molar_entropy == m.molar_entropy)

    def test_SPOCK_isothermal_contribution(self):
        params = mineral_params.copy()
        params["reference_eos"] = create("spock")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        params["Debye_0"] = 1000.0
        params["grueneisen_0"] = 1.2
        params["q_0"] = 1.1
        params["P_0"] = 100000.0
        params["Kprime_inf"] = 2.0
        params["Kprime_0"] = 4.21796
        params["Kdprime_0"] = -4 / 100000000000.0
        m = Mineral(params)
        consistent = check_eos_consistency(
            m, 2000000000.0, 2000.0, including_shear_properties=False, tol=0.0001
        )
        self.assertTrue(consistent)

    def test_SLB_is_same_as_MGD_non_conductive(self):
        fo_slb = forsterite_slb()
        params = parameter_dict(fo_slb.params).copy()
        params["equation_of_state"] = "modular_mgd"
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        fo_mgd = Mineral(params)
        minerals = make_assemblage([fo_slb, fo_mgd])
        minerals.set_state(100000.0, 298.15)
        self.assertAlmostEqual(fo_slb.molar_helmholtz, fo_mgd.molar_helmholtz, places=3)
        minerals.set_state(1000000000.0, 2000.0)
        self.assertAlmostEqual(fo_slb.molar_helmholtz, fo_mgd.molar_helmholtz, places=3)

    def test_SLB_is_same_as_MGD_conductive(self):
        fea_slb = alpha_iron_slb()
        fea_slb.set_property_modifiers([])
        params = parameter_dict(fea_slb.params).copy()
        params["equation_of_state"] = "modular_mgd"
        params["reference_eos"] = create("bm3")
        params["debye_temperature_model"] = debye_temperature_models.SLB()
        fea_mgd = Mineral(params)
        self.assertTrue(params["bel_0"] != 0)
        minerals = make_assemblage([fea_slb, fea_mgd])
        minerals.set_state(params["P_0"], params["T_0"])
        self.assertAlmostEqual(fea_mgd.molar_helmholtz, params["F_0"])
        self.assertAlmostEqual(fea_mgd.molar_volume, params["V_0"], places=16)
        minerals.set_state(params["P_0"], params["T_0"])
        self.assertAlmostEqual(fea_slb.molar_helmholtz, params["F_0"])
        self.assertAlmostEqual(fea_slb.molar_volume, params["V_0"], places=16)
        minerals.set_state(1000000000.0, 2000.0)
        self.assertAlmostEqual(
            fea_slb.molar_helmholtz, fea_mgd.molar_helmholtz, places=3
        )
