# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_material.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
import numpy as np
import burnman_cpp as burnman


class test_material_name(BurnManTest):
    """test Material.name and that we can edit and override it in Mineral"""

    class min_no_name(burnman.Mineral):
        """
        Stixrude & Lithgow-Bertelloni 2005 and references therein
        """

        def __init__(self):
            params = {
                "equation_of_state": "slb3",
                "T_0": 300.0,
                "P_0": 0.0,
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
            params.setdefault("name", "min_no_name")
            burnman.Mineral.__init__(self, params)

    class min_with_name(burnman.Mineral):
        """
        Stixrude & Lithgow-Bertelloni 2005 and references therein
        """

        def __init__(self):
            params = {
                "name": "name set in params",
                "equation_of_state": "slb3",
                "T_0": 300.0,
                "P_0": 0.0,
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
            params.setdefault("name", "min_with_name")
            burnman.Mineral.__init__(self, params)

    class min_with_name_manually(burnman.Mineral):
        """
        Stixrude & Lithgow-Bertelloni 2005 and references therein
        """

        def __init__(self):
            params = {
                "equation_of_state": "slb3",
                "T_0": 300.0,
                "P_0": 0.0,
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
            name = "manually set"
            params.setdefault("name", "min_with_name_manually")
            burnman.Mineral.__init__(self, params)
            self.set_name(name)

    def test_mineral_without_name(self):
        m = self.min_no_name()
        self.assertEqual(m.name, "min_no_name")
        m.name = "bla"
        self.assertEqual(m.name, "bla")

    def test_mineral_with_name(self):
        m = self.min_with_name()
        self.assertEqual(m.name, "name set in params")
        m.name = "bla"
        self.assertEqual(m.name, "bla")

    def test_set_name_before_init(self):
        m = self.min_with_name_manually()
        self.assertEqual(m.name, "manually set")
        m.name = "bla"
        self.assertEqual(m.name, "bla")

    def test_evaluate_list(self):
        m = self.min_with_name()
        Ss = m.evaluate(["molar_entropy"], [100000.0, 100000.0], [300.0, 300.0])
        self.assertEqual(Ss[0].shape, (2,))

    def test_evaluate_1Darray(self):
        m = self.min_with_name()
        Ss = m.evaluate(
            ["molar_entropy"], np.array([100000.0, 100000.0]), np.array([300.0, 300.0])
        )
        self.assertEqual(Ss[0].shape, (2,))

    def test_evaluate_2Darray(self):
        m = self.min_with_name()
        Ss = m.evaluate(
            ["molar_entropy"],
            np.array([[100000.0, 100000.0, 100000.0], [100000.0, 100000.0, 100000.0]]),
            np.array([[300.0, 300.0, 300], [300.0, 300.0, 300]]),
        )
        self.assertEqual(Ss[0].shape, (2, 3))

    def test_set_state_with_volume(self):
        m = self.min_with_name()
        P0 = 6000000000.0
        T0 = 1000.0
        T1 = 298.15
        m.set_state(P0, T0)
        V = m.molar_volume
        m.set_state_with_volume(V, T1)
        P1 = m.pressure
        m.set_state(P0, T0)
        m.set_state(P1, T1)
        self.assertFloatEqual(V, m.molar_volume)

    def test_unset_state_raises(self):
        m = self.min_with_name()
        with self.assertRaises(RuntimeError):
            _ = m.pressure
        with self.assertRaises(RuntimeError):
            _ = m.temperature
        with self.assertRaises(RuntimeError):
            _ = m.molar_volume
