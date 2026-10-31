# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_partitioning.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
import burnman_cpp as burnman
from burnman_cpp import minerals


class test(BurnManTest):

    def test_simple(self):
        bulk_composition = burnman.Composition(
            {"Mg": 0.213, "Fe": 0.0626, "Si": 0.242, "Ca": 0.0, "Al": 0.0},
            unit_type="weight",
        )
        bdg = minerals.SLB_2011.mg_fe_bridgmanite()
        per = minerals.SLB_2011.ferropericlase()
        pressure = 23830000000.0
        temperature = 2000.0
        a, b = burnman.calculate_nakajima_fp_pv_partition_coefficient(
            pressure, temperature, bulk_composition.molar_composition, 0.5
        )
        self.assertFloatEqual(a, 0.184533288)
        self.assertFloatEqual(b, 0.102937268)
