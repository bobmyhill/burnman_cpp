# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_composite.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    parameter_dict,
    make_assemblage,
    native_method,
)
import burnman_cpp as burnman
from burnman_cpp import minerals


class composite(BurnManTest):

    def test_evaluate(self):
        rock = make_assemblage(
            [minerals.SLB_2005.wuestite(), minerals.SLB_2005.mg_perovskite()],
            [0.5, 0.5],
        )
        K, G = rock.evaluate(["K_eff", "G_eff"], 40.0e9, 2000.0)
        rock.set_state(40.0e9, 2000.0)
        # Python's K_eff is the effective seismic modulus, not K_S (Reuss).
        K2 = rock.K_eff
        G2 = rock.G_eff
        self.assertFloatEqual(K, K2)
        self.assertFloatEqual(G, G2)

    def test_density_composite(self):
        pyrolite = make_assemblage(
            [minerals.SLB_2005.mg_perovskite(), minerals.SLB_2005.periclase()],
            [0.95, 0.05],
        )
        pyrolite.set_method(native_method("slb3"))
        pyrolite.set_state(40000000000.0, 2000)
        d1 = pyrolite.phases[0].density
        d2 = pyrolite.phases[1].density
        dmix = pyrolite.density
        self.assertFloatEqual(d1, 4494.483)
        self.assertFloatEqual(d2, 4130.096)
        self.assertFloatEqual(dmix, 4486.263)

    def test_thermodynamic_potentials(self):
        rock1 = make_assemblage(
            [minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],
            [1.0, 0.0],
        )
        rock2 = make_assemblage(
            [minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],
            [0.0, 1.0],
        )
        rock3 = make_assemblage(
            [minerals.SLB_2011.mg_perovskite(), minerals.SLB_2011.periclase()],
            [0.4, 0.6],
        )
        mineral1 = minerals.SLB_2011.mg_perovskite()
        mineral2 = minerals.SLB_2011.periclase()
        rock1.set_state(40000000000.0, 2000.0)
        rock2.set_state(40000000000.0, 2000.0)
        rock3.set_state(40000000000.0, 2000.0)
        mineral1.set_state(40000000000.0, 2000.0)
        mineral2.set_state(40000000000.0, 2000.0)
        gibbs1 = mineral1.molar_gibbs
        gibbs2 = mineral2.molar_gibbs
        G1 = rock1.molar_gibbs
        G2 = rock2.molar_gibbs
        G3 = rock3.molar_gibbs
        self.assertFloatEqual(G1, gibbs1)
        self.assertFloatEqual(G2, gibbs2)
        self.assertFloatEqual(G3, 0.4 * gibbs1 + 0.6 * gibbs2)
        helmholtz1 = mineral1.molar_helmholtz
        helmholtz2 = mineral2.molar_helmholtz
        F1 = rock1.molar_helmholtz
        F2 = rock2.molar_helmholtz
        F3 = rock3.molar_helmholtz
        self.assertFloatEqual(F1, helmholtz1)
        self.assertFloatEqual(F2, helmholtz2)
        self.assertFloatEqual(F3, 0.4 * helmholtz1 + 0.6 * helmholtz2)
        enthalpy1 = mineral1.molar_enthalpy
        enthalpy2 = mineral2.molar_enthalpy
        H1 = rock1.molar_enthalpy
        H2 = rock2.molar_enthalpy
        H3 = rock3.molar_enthalpy
        self.assertFloatEqual(H1, enthalpy1)
        self.assertFloatEqual(H2, enthalpy2)
        self.assertFloatEqual(H3, 0.4 * enthalpy1 + 0.6 * enthalpy2)
        internal_energy1 = mineral1.molar_internal_energy
        internal_energy2 = mineral2.molar_internal_energy
        U1 = rock1.molar_internal_energy
        U2 = rock2.molar_internal_energy
        U3 = rock3.molar_internal_energy
        self.assertFloatEqual(U1, internal_energy1)
        self.assertFloatEqual(U2, internal_energy2)
        self.assertFloatEqual(U3, 0.4 * internal_energy1 + 0.6 * internal_energy2)
        molar_entropy1 = mineral1.molar_entropy
        molar_entropy2 = mineral2.molar_entropy
        S1 = rock1.molar_entropy
        S2 = rock2.molar_entropy
        S3 = rock3.molar_entropy
        self.assertFloatEqual(S1, molar_entropy1)
        self.assertFloatEqual(S2, molar_entropy2)
        self.assertFloatEqual(S3, 0.4 * molar_entropy1 + 0.6 * molar_entropy2)

    def test_swap_averaging(self):
        rock = make_assemblage(
            [minerals.SLB_2005.wuestite(), minerals.SLB_2005.mg_perovskite()],
            [0.5, 0.5],
        )
        rock.set_state(5000000000.0, 1000.0)
        G1 = rock.shear_modulus
        self.assertFloatEqual(108.686, G1 / 1000000000.0)
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanLower)
        rock.set_state(5000000000.0, 1000.0)
        G2 = rock.G_eff
        self.assertFloatEqual(104.451, G2 / 1000000000.0)
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanUpper)
        rock.set_state(5000000000.0, 1000.0)
        G3 = rock.shear_modulus
        self.assertFloatEqual(115.155, G3 / 1000000000.0)

    def test_mass_to_molar_fractions(self):
        pv = burnman.minerals.HP_2011_ds62.mpv()
        en = burnman.minerals.HP_2011_ds62.en()
        c = make_assemblage([pv, en], [0.5, 0.5], "mass")
        self.assertArraysAlmostEqual(c.molar_fractions, [2.0 / 3.0, 1.0 / 3.0])

    def test_mass_to_molar_fractions_2(self):
        min1 = minerals.SLB_2005.periclase()
        mass_fractions = [0.8, 0.2]
        c = make_assemblage([min1, min1], mass_fractions, fraction_type="mass")
        self.assertArraysAlmostEqual(c.molar_fractions, mass_fractions)

    def test_formula(self):
        min1 = minerals.SLB_2005.mg_perovskite()
        min2 = minerals.SLB_2005.periclase()
        molar_fractions = [0.5, 0.5]
        c = make_assemblage([min1, min2], molar_fractions, fraction_type="molar")
        self.assertArraysAlmostEqual(
            [c.formula["Mg"], c.formula["Si"], c.formula["O"]], [1.0, 0.5, 2.0]
        )

    def test_query_properties(self):
        min1 = minerals.SLB_2011.periclase()
        rock1 = make_assemblage([min1, min1], [0.5, 0.5])
        rock1.set_state(
            parameter_dict(min1.params)["P_0"], parameter_dict(min1.params)["T_0"]
        )
        self.assertFloatEqual(rock1.molar_volume, parameter_dict(min1.params)["V_0"])
        self.assertFloatEqual(
            rock1.molar_mass, parameter_dict(min1.params)["molar_mass"]
        )
        self.assertFloatEqual(
            rock1.isothermal_bulk_modulus_reuss, parameter_dict(min1.params)["K_0"]
        )
        self.assertFloatEqual(
            rock1.isothermal_compressibility_reuss,
            1.0 / parameter_dict(min1.params)["K_0"],
        )
        self.assertFloatEqual(
            rock1.isentropic_compressibility_reuss,
            1.0 / rock1.isentropic_bulk_modulus_reuss,
        )
        self.assertFloatEqual(
            rock1.grueneisen_parameter, parameter_dict(min1.params)["grueneisen_0"]
        )
        self.assertFloatEqual(rock1.thermal_expansivity, min1.thermal_expansivity)
        self.assertFloatEqual(rock1.molar_heat_capacity_v, min1.molar_heat_capacity_v)
        self.assertFloatEqual(rock1.molar_heat_capacity_p, min1.molar_heat_capacity_p)

    def test_stoichiometric_matrix_single_mineral(self):
        rock = make_assemblage([minerals.SLB_2011.quartz()])
        self.assertTrue(len(rock.endmember_names) == 1)
        self.assertTrue(len(rock.elements) == 2)
        self.assertTrue(rock.elements[0] == "Si")
        self.assertTrue(rock.elements[1] == "O")
        self.assertTrue(rock.stoichiometric_matrix.shape == (1, 2))
        self.assertTrue(rock.reaction_basis.shape[0] == 0)
        self.assertArraysAlmostEqual(rock.compositional_null_basis[0], [-2, 1])
        self.assertTrue(rock.independent_element_indices[0] == 0)
        self.assertTrue(rock.dependent_element_indices[0] == 1)
        self.assertTrue(rock.n_reactions == 0)

    def test_stoichiometric_matrix_two_minerals(self):
        rock = make_assemblage(
            [minerals.HP_2011_ds62.andr(), minerals.HP_2011_ds62.alm()]
        )
        self.assertTrue(len(rock.endmember_names) == 2)
        self.assertTrue(len(rock.elements) == 5)
        self.assertTrue(rock.stoichiometric_matrix.shape == (2, 5))
        self.assertTrue(rock.reaction_basis.shape[0] == 0)
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[0], [4.0 / 9.0, -2.0 / 3.0, 1, 0, 0]
        )
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[1], [-1.0 / 3.0, -1.0, 0, 1, 0]
        )
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[2], [-4.0 / 3.0, -4.0, 0, 0, 1]
        )
        self.assertArraysAlmostEqual(rock.independent_element_indices, [0, 1])
        self.assertArraysAlmostEqual(rock.dependent_element_indices, [2, 3, 4])
        self.assertTrue(rock.n_reactions == 0)

    def test_stoichiometric_matrix_binary_solution(self):
        rock = make_assemblage([minerals.SLB_2011.mg_fe_olivine()])
        self.assertTrue(len(rock.endmember_names) == 2)
        self.assertTrue(len(rock.elements) == 4)
        self.assertTrue(rock.stoichiometric_matrix.shape == (2, 4))
        self.assertTrue(rock.reaction_basis.shape[0] == 0)
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[0], [-1.0 / 2.0, -1.0 / 2.0, 1, 0]
        )
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[1], [-2.0, -2.0, 0, 1]
        )
        self.assertArraysAlmostEqual(rock.independent_element_indices, [0, 1])
        self.assertArraysAlmostEqual(rock.dependent_element_indices, [2, 3])
        self.assertTrue(rock.n_reactions == 0)

    def test_stoichiometric_matrix_reaction(self):
        rock = make_assemblage(
            [minerals.SLB_2011.mg_fe_olivine(), minerals.SLB_2011.garnet()]
        )
        self.assertTrue(len(rock.endmember_names) == 7)
        self.assertTrue(len(rock.elements) == 7)
        self.assertTrue(rock.stoichiometric_matrix.shape == (7, 7))
        self.assertArraysAlmostEqual(
            rock.reaction_basis[0], [3.0 / 2.0, -3.0 / 2.0, -1.0, 1.0, 0.0, 0.0, 0]
        )
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[0],
            [-1.0 / 2.0, -1.0, -1.0, -1.0, -3.0 / 2.0, -2.0, 1.0],
        )
        self.assertArraysAlmostEqual(
            rock.independent_element_indices, [0, 1, 2, 3, 4, 5]
        )
        self.assertArraysAlmostEqual(rock.dependent_element_indices, [6])
        self.assertTrue(rock.n_reactions == 1)

    def test_stoichiometric_matrix_reactions(self):
        rock = make_assemblage(
            [
                minerals.SLB_2011.mg_fe_olivine(),
                minerals.SLB_2011.mg_fe_olivine(),
                minerals.SLB_2011.mg_fe_olivine(),
            ]
        )
        self.assertTrue(len(rock.endmember_names) == 6)
        self.assertTrue(len(rock.elements) == 4)
        self.assertTrue(rock.stoichiometric_matrix.shape == (6, 4))
        self.assertArraysAlmostEqual(rock.reaction_basis[0], [-1, 0, 1, 0, 0, 0])
        self.assertArraysAlmostEqual(rock.reaction_basis[1], [0, -1, 0, 1, 0, 0])
        self.assertArraysAlmostEqual(rock.reaction_basis[2], [-1, 0, 0, 0, 1, 0])
        self.assertArraysAlmostEqual(rock.reaction_basis[3], [0, -1, 0, 0, 0, 1])
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[0], [-1.0 / 2.0, -1.0 / 2.0, 1, 0]
        )
        self.assertArraysAlmostEqual(
            rock.compositional_null_basis[1], [-2.0, -2.0, 0, 1]
        )
        self.assertArraysAlmostEqual(rock.independent_element_indices, [0, 1])
        self.assertArraysAlmostEqual(rock.dependent_element_indices, [2, 3])
        self.assertTrue(rock.n_reactions == 4)

    def test_single_phase_composite(self):
        min1 = minerals.SLB_2011.mg_fe_olivine()
        min2 = minerals.SLB_2011.mg_fe_olivine()
        min1.set_composition([0.8, 0.2])
        min2.set_composition([0.8, 0.2])
        rock1 = make_assemblage([min1, min2], [0.6, 0.4])
        properties = [
            "gibbs",
            "molar_entropy",
            "molar_volume",
            "molar_heat_capacity_p",
            "molar_heat_capacity_v",
            "thermal_expansivity",
            "density",
            "grueneisen_parameter",
            "isothermal_bulk_modulus_reuss",
            "isentropic_bulk_modulus_reuss",
        ]
        rock_prps = rock1.evaluate(properties, 1000000000.0, 1400.0)
        min_prps = min1.evaluate(properties, 1000000000.0, 1400.0)
        for i in range(len(properties)):
            self.assertFloatEqual(rock_prps[i], min_prps[i])

    def test_composite_state_not_set(self):
        min1 = minerals.SLB_2011.mg_perovskite()
        min2 = minerals.SLB_2011.periclase()
        molar_fractions = [0.5, 0.5]
        c = make_assemblage([min1, min2], molar_fractions, fraction_type="molar")
        with self.assertRaises(RuntimeError):
            _ = c.density
