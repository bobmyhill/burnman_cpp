# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_endmembers.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    parameter_dict,
    set_parameter,
    native_copy,
    mineral_eos,
    eos_call,
)
import burnman_cpp as burnman
from burnman_cpp import Mineral, CombinedMineral
from burnman_cpp.utils.chemistry import dictionarize_formula, formula_mass
from burnman_cpp.tools.eos import check_eos_consistency


class forsterite(Mineral):

    def __init__(self):
        formula = "Mg2.0Si1.0O4.0"
        formula = dictionarize_formula(formula)
        params = {
            "name": "fo",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -2172590.0,
            "S_0": 95.1,
            "V_0": 4.366e-05,
            "Cp": [233.3, 0.001494, -603800.0, -1869.7],
            "a_0": 2.85e-05,
            "K_0": 128500000000.0,
            "Kprime_0": 3.84,
            "Kdprime_0": -3e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        params.setdefault("name", "forsterite")
        burnman.Mineral.__init__(self, params)


class test_endmembers(BurnManTest):

    def test0_T_ref(self):
        fo = forsterite()
        set_parameter(fo, "P_0", 99999.999999)
        fo.set_state(100000.0, 500.0)
        data = [fo.molar_enthalpy, fo.molar_entropy]
        T_0 = 2000.0
        fo.set_state(100000.0, T_0)
        fo2 = native_copy(fo)
        set_parameter(fo2, "T_0", T_0)
        set_parameter(fo2, "H_0", fo.molar_enthalpy)
        set_parameter(fo2, "S_0", fo.molar_entropy)
        set_parameter(fo2, "V_0", fo.molar_volume)
        set_parameter(fo2, "K_0", fo.isothermal_bulk_modulus_reuss)
        set_parameter(
            fo2,
            "Kdprime_0",
            -parameter_dict(fo.params)["Kprime_0"] / parameter_dict(fo2.params)["K_0"],
        )
        set_parameter(fo2, "a_0", fo.thermal_expansivity)
        fo2.set_state(100000.0, 500.0)
        data2 = [fo2.molar_enthalpy, fo2.molar_entropy]
        self.assertArraysAlmostEqual(data, data2)

    def test_mgd(self):
        bdg = burnman.minerals.Matas_etal_2007.mg_perovskite()
        bdg.set_state(100000.0, 300.0)
        self.assertFloatEqual(
            bdg.grueneisen_parameter, parameter_dict(bdg.params)["grueneisen_0"]
        )
        self.assertFloatEqual(
            bdg.isothermal_bulk_modulus_reuss, parameter_dict(bdg.params)["K_0"]
        )
        self.assertFloatEqual(bdg.molar_volume, parameter_dict(bdg.params)["V_0"])
        self.assertFloatEqual(bdg.shear_modulus, parameter_dict(bdg.params)["G_0"])
        bdg.set_state(100000.0, 0.0)
        self.assertFloatEqual(bdg.molar_heat_capacity_v, 0.0)
        self.assertFloatEqual(bdg.molar_heat_capacity_p, 0.0)
        self.assertFloatEqual(bdg.thermal_expansivity, 0.0)
        self.assertFloatEqual(
            bdg.isothermal_bulk_modulus_reuss, bdg.isentropic_bulk_modulus_reuss
        )

    def test_make_mbr(self):
        bdg = burnman.minerals.SLB_2011.mg_perovskite()
        dS = 1.0
        made_bdg = CombinedMineral([bdg, bdg], [2.0, -1.0], [0.0, dS, 0.0])
        bdg.set_state(100000.0, 1000.0)
        made_bdg.set_state(100000.0, 1000.0)
        self.assertFloatEqual(bdg.molar_entropy + dS, made_bdg.molar_entropy)

    def test_make_mbr2(self):
        per = burnman.minerals.SLB_2011.periclase()
        stv = burnman.minerals.SLB_2011.stishovite()
        dE = -15000.0
        made_bdg = CombinedMineral(
            [
                burnman.minerals.SLB_2011.periclase(),
                burnman.minerals.SLB_2011.stishovite(),
            ],
            [1.0, 1.0],
            [dE, 0.0, 0.0],
        )
        per.set_state(100000.0, 1000.0)
        stv.set_state(100000.0, 1000.0)
        made_bdg.set_state(100000.0, 1000.0)
        self.assertFloatEqual(
            made_bdg.molar_volume, per.molar_volume + stv.molar_volume
        )

    def test_brosh_pressure(self):
        fcc = burnman.minerals.SE_2015.fcc_iron()
        fcc.set_state(1000000000.0, 500.0)
        self.assertTrue(
            abs(
                fcc.pressure
                - eos_call(
                    mineral_eos(fcc),
                    "pressure",
                    500.0,
                    fcc.molar_volume,
                    parameter_dict(fcc.params),
                )
            )
            < 1000.0
        )
        self.assertTrue(check_eos_consistency(fcc, including_shear_properties=False))
