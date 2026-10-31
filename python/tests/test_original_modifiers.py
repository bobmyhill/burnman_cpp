# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_modifiers.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
from burnman_cpp.minerals import SLB_2011, SLB_2024, HP_2011_ds62
from burnman_cpp.tools.eos import check_eos_consistency

linear_params = {"delta_E": 1200.0, "delta_S": 5.0, "delta_V": 1e-07}
landau_params = {"Tc_0": 800.0, "S_D": 5.0, "V_D": 1e-07}
landau_params_2 = {"Tc_0": 1200.0, "S_D": 5.0, "V_D": 1e-07}
landau_hp_params = {
    "P_0": 100000.0,
    "T_0": 298.15,
    "Tc_0": 800.0,
    "S_D": 5.0,
    "V_D": 1e-07,
}
landau_hp_params_2 = {
    "P_0": 100000.0,
    "T_0": 298.15,
    "Tc_0": 1200.0,
    "S_D": 5.0,
    "V_D": 1e-07,
}
bragg_williams_params = {
    "n": 1.0,
    "factor": 0.8,
    "Wh": 1000.0,
    "Wv": 1e-07,
    "deltaH": 1000.0,
    "deltaV": 1e-07,
}
magnetic_params = {
    "structural_parameter": 0.4,
    "curie_temperature": [800.0, 1e-08],
    "magnetic_moment": [2.2, 1e-10],
}
magnetic_params_2 = {
    "structural_parameter": 0.4,
    "curie_temperature": [1200.0, 1e-08],
    "magnetic_moment": [2.2, 1e-10],
}
thermal_params = {"Theta_0": 1200.0, "Cv_inf": 1.0}
thermal_delta_params = {"Theta_0": 1200.0, "S_inf": 1.0}
modifier_params = [
    ["linear", linear_params],
    ["landau", landau_params],
    ["landau", landau_params_2],
    ["landau_hp", landau_hp_params],
    ["landau_hp", landau_hp_params_2],
    ["bragg_williams", bragg_williams_params],
    ["magnetic_chs", magnetic_params],
    ["magnetic_chs", magnetic_params_2],
    ["debye", thermal_params],
    ["debye_delta", thermal_delta_params],
    ["einstein", thermal_params],
    ["einstein_delta", thermal_delta_params],
]


class Modifiers(BurnManTest):

    def test_landau_slb_2022_mineral_responses(self):
        """
        Check full EoS consistency for minerals
        with pressure-dependent order.
        """
        for phase in (SLB_2024.qtz(), SLB_2024.neph()):
            with self.subTest(mineral=phase):
                self.assertTrue(
                    check_eos_consistency(
                        phase, 2000000000.0, 300.0, including_shear_properties=False
                    )
                )

    def test_modifier_with_pvt_eos(self):
        qtz = SLB_2011.quartz()
        assert check_eos_consistency(qtz, 100000.0, 600.0)
        assert check_eos_consistency(qtz, 100000.0, 1100.0)
        assert check_eos_consistency(qtz, 1000000000.0, 1100.0)

    def test_modifier_with_vpt_eos(self):
        qtz = HP_2011_ds62.q()
        assert check_eos_consistency(
            qtz, 100000.0, 600.0, including_shear_properties=False
        )
        assert check_eos_consistency(
            qtz, 100000.0, 1100.0, including_shear_properties=False
        )
        assert check_eos_consistency(
            qtz, 1000000000.0, 1100.0, including_shear_properties=False
        )

    def test_consistency(self):
        per = SLB_2011.periclase()
        for name, params in modifier_params:
            per.set_property_modifiers([[name, params]])
            if name != "bragg_williams":
                assert check_eos_consistency(
                    per, 1000000000.0, 200.0, including_shear_properties=False
                )
                assert check_eos_consistency(
                    per, 1000000000.0, 1100.0, including_shear_properties=False
                )
