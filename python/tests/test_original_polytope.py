# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_polytope.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, make_assemblage, native_endmembers
from burnman_cpp.minerals import SLB_2011, SLB_2024
from burnman_cpp.tools.polytope import simplify_composite_with_composition


class polytope(BurnManTest):

    def test_simplify_composite(self):
        gt = SLB_2011.garnet()
        ol = SLB_2011.mg_fe_olivine()
        assemblage = make_assemblage([ol, gt], [0.7, 0.3])
        composition = {"Fe": 3.0, "Mg": 1.0, "Si": 3.9, "O": 11.8}
        new_assemblage = simplify_composite_with_composition(assemblage, composition)
        new_gt = new_assemblage.phases[1]
        self.assertTrue(new_gt.n_endmembers == 2)
        strings = list(sorted([e[1] for e in native_endmembers(new_gt)]))
        self.assertEqual(strings[0], "[Fe]3[Mg][Si]")
        self.assertEqual(strings[1], "[Mg]3[Mg][Si]")

    def test_simplify_composite_and_composition(self):
        gt = SLB_2011.garnet()
        gt.set_composition([-0.1, 0.1, 0.0, 1.0, 0.0])
        ol = SLB_2011.mg_fe_olivine()
        assemblage = make_assemblage([ol, gt], [0.7, 0.3])
        composition = {"Fe": 3.0, "Mg": 1.0, "Si": 3.9, "O": 11.8}
        new_assemblage = simplify_composite_with_composition(assemblage, composition)
        new_gt = new_assemblage.phases[1]
        self.assertTrue(new_gt.n_endmembers == 2)
        strings = list(sorted([e[1] for e in native_endmembers(new_gt)]))
        self.assertEqual(strings[0], "[Fe]3[Mg][Si]")
        self.assertEqual(strings[1], "[Mg]3[Mg][Si]")
        self.assertArraysAlmostEqual([0.1, 0.9], new_gt.molar_fractions)

    def test_simplified_composite_endmember_order(self):
        wu = SLB_2024.ferropericlase()
        wu.set_composition([0.0, 0.97, 0.005, 0.0, 0.025])
        wu = simplify_composite_with_composition(
            make_assemblage([wu]), {"Fe": 1, "O": 1.01}
        ).phases[0]
        self.assertTrue(len(native_endmembers(wu)) == 3)
        self.assertArraysAlmostEqual(wu.molar_fractions, [0.97, 0.005, 0.025])
