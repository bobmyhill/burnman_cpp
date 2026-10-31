# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_minerals.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
from burnman_cpp import minerals


class VolumeStateModifiers(BurnManTest):

    def test_volume_state_does_not_add_modifier_volume_twice(self):
        mineral = minerals.SLB_2011.forsterite()
        mineral.set_property_modifiers(
            [["linear", {"delta_V": 2e-07, "delta_S": 0.5, "delta_E": 100.0}]]
        )
        mineral.set_state(3000000000.0, 700.0)
        volume = mineral.molar_volume
        gibbs, helmholtz = (mineral.molar_gibbs, mineral.molar_helmholtz)
        mineral.set_state_with_volume(volume, 700.0)
        self.assertAlmostEqual(mineral.molar_volume / volume, 1.0, places=12)
        self.assertAlmostEqual(mineral.pressure / 3000000000.0, 1.0, places=10)
        self.assertAlmostEqual(mineral.molar_gibbs / gibbs, 1.0, places=12)
        self.assertAlmostEqual(mineral.molar_helmholtz / helmholtz, 1.0, places=12)
