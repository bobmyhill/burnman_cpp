# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_equilibration.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, make_assemblage, equilibrate
from burnman_cpp.minerals import HP_2011_ds62, SLB_2011
import numpy as np


def make_ol_wad_assemblage():
    ol = SLB_2011.mg_fe_olivine()
    wad = SLB_2011.mg_fe_wadsleyite()
    assemblage = make_assemblage([ol, wad], [0.7, 0.3])
    ol.set_composition([0.5, 0.5])
    wad.set_composition([0.6, 0.4])
    return assemblage


class equilibration(BurnManTest):

    def test_univariant_line(self):
        andalusite = HP_2011_ds62.andalusite()
        kyanite = HP_2011_ds62.ky()
        composition = kyanite.formula
        pressures = np.linspace(100000.0, 1000000000.0, 11)
        assemblage = make_assemblage([andalusite, kyanite])
        equality_constraints = [
            ("P", pressures),
            ("phase_fraction", (andalusite, np.array([0.0]))),
        ]
        sols, prm = equilibrate(composition, assemblage, equality_constraints)
        Ts = [sol.assemblage.temperature for sol in sols]
        Ts_ref = [
            464.95783870074905,
            544.0828494722284,
            623.6421086627697,
            703.7607424193357,
            784.4975351197959,
            865.8789417005959,
            947.9152419887187,
            1030.608906228574,
            1113.95923718214,
            1197.9651099824446,
            1282.6266945380332,
        ]
        self.assertArraysAlmostEqual(Ts, Ts_ref)

    def test_invariant(self):
        sillimanite = HP_2011_ds62.sill()
        andalusite = HP_2011_ds62.andalusite()
        kyanite = HP_2011_ds62.ky()
        composition = sillimanite.formula
        assemblage = make_assemblage([sillimanite, andalusite, kyanite])
        equality_constraints = [
            ("phase_fraction", (kyanite, np.array([0.0]))),
            ("phase_fraction", (sillimanite, np.array([0.0]))),
        ]
        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        self.assertArraysAlmostEqual(sol.x, [430671426.0, 809.342875, 0.0, 1.0, 0.0])

    def test_univariant_line_fo(self):
        forsterite = SLB_2011.forsterite()
        periclase = SLB_2011.periclase()
        bridgmanite = SLB_2011.mg_bridgmanite()
        composition = forsterite.formula
        temperatures = np.linspace(300.0, 2000.0, 11)
        assemblage = make_assemblage([forsterite, periclase, bridgmanite])
        equality_constraints = [
            ("T", temperatures),
            ("phase_fraction", (forsterite, np.array([0.0]))),
        ]
        sols, prm = equilibrate(composition, assemblage, equality_constraints)
        Ps = np.array([sol.assemblage.pressure for sol in sols])
        Ps_ref = [
            17405868549,
            17657896452,
            17941848050,
            18231743600,
            18517274303,
            18793435710,
            19057314295,
            19306923194,
            19540715743,
            19757356711,
            19955602937,
        ]
        self.assertArraysAlmostEqual(Ps, Ps_ref)

    def test_conservation_of_elements(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage.set_state(10000000000.0, 1200.0)
        equality_constraints = [
            ("P", 10000000000.0),
            (
                "phase_composition",
                (ol, (["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45)),
            ),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        for key in composition:
            self.assertAlmostEqual(assemblage.formula[key], composition[key], places=5)
        old_number_of_moles = assemblage.n_moles
        composition = {key: 2.0 * composition[key] for key in composition}
        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        self.assertAlmostEqual(assemblage.n_moles, 2.0 * old_number_of_moles, places=5)

    def test_ol_wad_eqm(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage.set_state(10000000000.0, 1200.0)
        equality_constraints = [
            ("P", 10000000000.0),
            (
                "phase_composition",
                (ol, (["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45)),
            ),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        sol, prm = equilibrate(composition, assemblage, equality_constraints)
        self.assertArraysAlmostEqual(
            [assemblage.temperature, ol.molar_fractions[1], wad.molar_fractions[1]],
            [1620.532183457096, 0.45, 0.6791743],
        )

    def test_incorrect_tol(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        assemblage.set_state(10000000000.0, 1200.0)
        equality_constraints = [
            ("P", 10000000000.0),
            (
                "phase_composition",
                (ol, (["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], 0.45)),
            ),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        with self.assertRaises((ValueError, RuntimeError)):
            _ = equilibrate(
                composition, assemblage, equality_constraints, tol=[0.001, 0.001]
            )
        with self.assertRaises((ValueError, RuntimeError)):
            _ = equilibrate(
                composition, assemblage, equality_constraints, tol=[[0.001] * 6] * 2
            )
        _ = equilibrate(composition, assemblage, equality_constraints, tol=0.001)
        _ = equilibrate(composition, assemblage, equality_constraints, tol=[0.001] * 6)
        _ = equilibrate(composition, assemblage, equality_constraints)

    def test_ol_wad_eqm_entropy(self):
        assemblage = make_ol_wad_assemblage()
        wad = assemblage.phases[1]
        equality_constraints = [("P", 10000000000.0), ("phase_fraction", (wad, 0.5))]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        P = assemblage.pressure
        T = assemblage.temperature
        S = assemblage.molar_entropy
        assemblage = make_ol_wad_assemblage()
        equality_constraints = [("P", P), ("S", S)]
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(assemblage.temperature, T, places=5)

    def test_ol_wad_eqm_volume(self):
        assemblage = make_ol_wad_assemblage()
        wad = assemblage.phases[1]
        equality_constraints = [("P", 10000000000.0), ("phase_fraction", (wad, 0.5))]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        P = assemblage.pressure
        T = assemblage.temperature
        V = assemblage.molar_volume * assemblage.n_moles
        assemblage = make_ol_wad_assemblage()
        equality_constraints = [("P", P), ("V", V)]
        sol, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(assemblage.temperature, T, places=5)

    def test_ol_wad_eqm_compositional_constraint(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage.set_state(10000000000.0, 1200.0)
        equality_constraints = [
            ("P", 10000000000.0),
            ("X", [[0.0, 0.0, 0.0, 2.0, 0.0, 0.0], 0.9]),
        ]
        composition = {"Mg": 1.0, "Fe": 1.0, "Si": 1.0, "O": 4.0}
        _, _ = equilibrate(composition, assemblage, equality_constraints)
        self.assertArraysAlmostEqual(
            [assemblage.temperature, ol.molar_fractions[1], wad.molar_fractions[1]],
            [1620.532183457096, 0.45, 0.6791743],
        )

    def test_mg_rich_ol_wad_eqm_with_free_compositional_vector(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage = make_assemblage([ol, wad], [1.0, 0.0])
        P = 12000000000.0
        T = 1673.0
        assemblage.set_state(P, T)
        x_fe_ol = 0.0001
        ol.set_composition([1.0 - x_fe_ol, x_fe_ol])
        wad.set_composition([1.0 - x_fe_ol, x_fe_ol])
        composition = {"Fe": 0.2, "Mg": 1.8, "Si": 1.0, "O": 4.0}
        free_compositional_vectors = [{"Mg": 1.0, "Fe": -1.0}]
        equality_constraints = [
            ("T", T),
            (
                "phase_composition",
                (ol, [["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], x_fe_ol]),
            ),
            ("phase_fraction", (wad, 0.0)),
        ]
        sol, _ = equilibrate(
            composition, assemblage, equality_constraints, free_compositional_vectors
        )
        self.assertEqual(sol.code, 0)

    def test_very_mg_rich_ol_wad_eqm_with_free_compositional_vector(self):
        assemblage = make_ol_wad_assemblage()
        ol = assemblage.phases[0]
        wad = assemblage.phases[1]
        assemblage = make_assemblage([ol, wad], [1.0, 0.0])
        P = 12000000000.0
        T = 1673.0
        assemblage.set_state(P, T)
        x_fe_ol = 1e-10
        ol.set_composition([0.8, 0.2])
        wad.set_composition([0.7, 0.3])
        composition = {"Fe": 0.1, "Mg": 1.9, "Si": 1.0, "O": 4.0}
        free_compositional_vectors = [{"Mg": 1.0, "Fe": -1.0}]
        equality_constraints = [
            ("T", T),
            (
                "phase_composition",
                (ol, [["Mg_A", "Fe_A"], [0.0, 1.0], [1.0, 1.0], x_fe_ol]),
            ),
            ("phase_fraction", (wad, 0.5)),
        ]
        sol, _ = equilibrate(
            composition, assemblage, equality_constraints, free_compositional_vectors
        )
        self.assertEqual(sol.code, 0)

    def test_ill_posed_problem(self):
        composition = {"Mg": 1.8, "Fe": 0.2, "Si": 1.0, "O": 4.0}
        a = make_assemblage(
            [SLB_2011.mg_fe_olivine(), SLB_2011.mg_fe_olivine()], [0.5, 0.5]
        )
        a.phases[0].set_composition([0.4, 0.6])
        a.phases[1].set_composition([0.5, 0.5])
        equality_constraints = [("P", 100000.0), ("T", 300.0)]
        sol, _ = equilibrate(composition, a, equality_constraints)
        self.assertEqual(sol.code, 2)
