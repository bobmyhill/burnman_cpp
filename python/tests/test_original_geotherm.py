# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_geotherm.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, native_method
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


class geotherm(BurnManTest):

    def test_brown_shankland(self):
        T = burnman.geotherm.BrownShankland().temperatures([600000.0])
        self.assertArraysAlmostEqual(T, [1825.0])

    def test_anderson(self):
        T = burnman.geotherm.Anderson().temperatures([621000.0])
        self.assertArraysAlmostEqual(T, [1805.0])

    def test_stacey_continental(self):
        T = burnman.geotherm.StaceyContinental().temperatures([11000.0, 620000.0])
        self.assertArraysAlmostEqual(T, [540.0, 2225])

    def test_stacey_oceanic(self):
        T = burnman.geotherm.StaceyOceanic().temperatures([11000.0, 620000.0])
        self.assertArraysAlmostEqual(T, [550.0, 2225])

    def test_katsura_geotherm(self):
        T = burnman.geotherm.Katsura2022().temperatures([100000.0, 400000.0])
        self.assertArraysAlmostEqual(T, [1672, 1796])

    def test_plesa_geotherm(self):
        T = burnman.geotherm.Plesa2022Mars6cm3().temperatures([173000.0, 320000.0])
        self.assertArraysAlmostEqual(T, [1028, 1489])

    def test_anzellini_prem(self):
        T = burnman.geotherm.Anzellini2013().temperatures([100000.0, 700000.0])
        self.assertArraysAlmostEqual(T, [1603.017, 1952.184])

    def test_adiabat(self):
        rock = mypericlase()
        pressure = [100000000000.0, 150000000000.0]
        rock.set_method(native_method("slb3"))
        T0 = 1500.0
        test_K_adiabat = burnman.geotherm.adiabatic_profile(pressure, rock, T0)
        self.assertArraysAlmostEqual(test_K_adiabat, [1500, 1650.22034002])

    def test_adiabat_with_different_anchor_pressure(self):
        rock = mypericlase()
        pressure = [100000000000.0, 50000000000.0]
        rock.set_method(native_method("slb3"))
        T0 = 1500.0
        T_at_50GPa = burnman.geotherm.adiabatic_profile(pressure, rock, T0)[-1]
        pressure = [100000000000.0, 150000000000.0]
        test_K_adiabat = burnman.geotherm.adiabatic_profile(
            pressure, rock, T_at_50GPa, P_0=50000000000.0
        )
        self.assertArraysAlmostEqual(test_K_adiabat, [1500, 1650.22034002])

    def test_adiabatic_geotherm(self):
        rock = mypericlase()
        rock.set_method(native_method("slb3"))
        T0 = 1500.0
        P0 = 100000000000.0
        adiabat = burnman.geotherm.AdiabaticGeotherm(rock, T0, P0)
        pressure = [100000000000.0, 150000000000.0]
        test_K_adiabat = adiabat.temperatures_from_pressures(pressure)
        self.assertArraysAlmostEqual(test_K_adiabat, [1500, 1650.22034002])

    def test_adiabatic_geotherm_with_prem(self):
        rock = mypericlase()
        rock.set_method(native_method("slb3"))
        prem = burnman.seismic.prem_model
        T0 = 1500.0
        P0 = 100000000000.0
        adiabat = burnman.geotherm.AdiabaticGeotherm(rock, T0, P0, prem)
        pressure = [100000000000.0, 150000000000.0]
        depths = prem.depth(pressure)
        test_K_adiabat = adiabat.temperatures(depths)
        self.assertArraysAlmostEqual(test_K_adiabat, [1500, 1650.22034002])
