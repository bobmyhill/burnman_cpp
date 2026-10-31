# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_chemical_potentials.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, make_assemblage, equilibrate
from burnman_cpp import minerals
from burnman_cpp.tools.chemistry import relative_fugacity
from burnman_cpp.tools.chemistry import fugacity


class ChemicalPotentials(BurnManTest):

    def test_chemical_potentials(self):
        bdg = minerals.SLB_2011.mg_fe_perovskite()
        per = minerals.SLB_2011.periclase()
        assemblage = make_assemblage([bdg, per])
        bdg.set_composition([1.0, 0.0, 0.0])
        assemblage.set_state(25000000000.0, 2000.0)
        assemblage.set_components([{"Si": 1.0, "O": 2.0}])
        mu_SiO2 = assemblage.chemical_potential()[0]
        self.assertFloatEqual(bdg.molar_gibbs - per.molar_gibbs, mu_SiO2)

    def test_fugacity(self):
        bdg = minerals.SLB_2011.mg_fe_perovskite()
        stv = minerals.SLB_2011.stishovite()
        assemblage = make_assemblage([bdg, stv])
        bdg.set_composition([1.0, 0.0, 0.0])
        assemblage.set_state(25000000000.0, 2000.0)
        self.assertFloatEqual(1.0, fugacity(stv, assemblage))

    def test_relative_fugacity(self):
        bdg = minerals.SLB_2011.mg_fe_perovskite()
        stv = minerals.SLB_2011.stishovite()
        assemblage = make_assemblage([bdg, stv])
        bdg.set_composition([1.0, 0.0, 0.0])
        assemblage.set_state(25000000000.0, 2000.0)
        self.assertFloatEqual(
            1.0, relative_fugacity({"Si": 1.0, "O": 2.0}, assemblage, assemblage)
        )

    def test_assemblage_single_mineral_mu(self):
        sillimanite = minerals.HP_2011_ds62.sill()
        assemblage = make_assemblage([sillimanite])
        assemblage.set_state(100000.0, 2000.0)
        mus = assemblage.chemical_potential(
            [sillimanite.formula, {"Al": 4.0, "Si": 2.0, "O": 10.0}]
        )
        self.assertFloatEqual(mus[0], mus[1] / 2.0)

    def test_assemblage_duplicate_mineral_mu(self):
        sillimanite = minerals.HP_2011_ds62.sill()
        sillimanite2 = minerals.HP_2011_ds62.sill()
        assemblage = make_assemblage([sillimanite, sillimanite2])
        assemblage.set_state(100000.0, 2000.0)
        mus = assemblage.chemical_potential([sillimanite.formula, sillimanite.formula])
        self.assertFloatEqual(mus[0], mus[1])

    def test_assemblage_two_mineral_no_reaction_mu(self):
        fo = minerals.HP_2011_ds62.fo()
        en = minerals.HP_2011_ds62.en()
        assemblage = make_assemblage([fo, en])
        assemblage.set_state(100000.0, 2000.0)
        mus = assemblage.chemical_potential(
            [fo.formula, en.formula, {"Si": 1.0, "O": 2.0}]
        )
        self.assertFloatEqual(mus[1] - mus[0], mus[2])

    def test_assemblage_three_mineral_reaction_mu(self):
        ol = minerals.SLB_2011.mg_fe_olivine()
        opx = minerals.SLB_2011.orthopyroxene()
        gt = minerals.SLB_2011.garnet()
        pressure = 10000000000.0
        temperature = 1500.0
        composition = {
            "Na": 0.02,
            "Fe": 0.2,
            "Mg": 2.0,
            "Si": 1.9,
            "Ca": 0.2,
            "Al": 0.4,
            "O": 6.81,
        }
        assemblage = make_assemblage([ol, opx, gt], [0.7, 0.1, 0.2])
        ol.set_composition([0.93, 0.07])
        opx.set_composition([0.8, 0.1, 0.05, 0.05])
        gt.set_composition([0.8, 0.1, 0.05, 0.03, 0.02])
        assemblage.set_state(pressure, temperature)
        equality_constraints = [("P", pressure), ("T", temperature)]
        sol, prm = equilibrate(
            composition, assemblage, equality_constraints, max_iterations=20
        )
        mus = assemblage.chemical_potential(
            [
                {"Fe": 2.0, "Al": -2.0, "Si": 1.0, "O": 1.0},
                {"Fe": 2.0, "Al": -2.0, "Si": 2.0, "O": 3.0},
                {"Si": 1.0, "O": 2.0},
            ]
        )
        self.assertFloatEqual(mus[1] - mus[0], mus[2])
