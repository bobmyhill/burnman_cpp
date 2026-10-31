# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_layer.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, native_method
import numpy as np
import burnman_cpp as burnman
from burnman_cpp import Layer, BoundaryLayerPerturbation


class min1(burnman.Mineral):

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
        params.setdefault("name", "min1")
        burnman.Mineral.__init__(self, params)


class test_layer(BurnManTest):

    def layer1(self):
        rock = min1()
        rock.set_method(native_method("slb3"))
        depths = np.linspace(660000.0, 2890000.0, 10)
        temperatures = np.linspace(2000.0, 2000.0, 10)
        layer = Layer("m1", depths)
        layer.set_material(rock)
        layer.set_temperature_mode("user-defined", temperatures=temperatures)
        layer.set_pressure_mode(
            "self-consistent", pressure_top=22500000000.0, gravity_bottom=10.0
        )
        layer.make()
        return layer

    def test_tbl(self):
        p = BoundaryLayerPerturbation(
            radius_bottom=3480000.0,
            radius_top=5711000.0,
            rayleigh_number=10000000.0,
            temperature_change=840.0,
            boundary_layer_ratio=0.0,
        )
        Ts = p.temperature(np.array([3480000.0, 5711000.0]))
        self.assertArraysAlmostEqual(Ts, [840.0, 0.0])
        p = BoundaryLayerPerturbation(
            radius_bottom=3480000.0,
            radius_top=5711000.0,
            rayleigh_number=10000000.0,
            temperature_change=1000.0,
            boundary_layer_ratio=0.25,
        )
        Ts = p.temperature(np.array([3480000.0, 5711000.0]))
        self.assertArraysAlmostEqual(Ts, [800.0, -200.0])
        p.set_model_thermal_gradients(-18.0 / 1000.0, -0.6 / 1000.0)
        dTdrs = p.dTdr(np.array([3480000.0, 5711000.0]))
        self.assertArraysAlmostEqual(dTdrs, [-18.0 / 1000.0, -0.6 / 1000.0])
