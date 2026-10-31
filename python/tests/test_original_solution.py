# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_solution.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict, native_method
import numpy as np
import warnings
import burnman_cpp as burnman
from burnman_cpp import Mineral
from burnman_cpp import CombinedMineral
from burnman_cpp import IdealSolution
from burnman_cpp import SymmetricRegularSolution
from burnman_cpp import SubregularSolution
from burnman_cpp import AsymmetricRegularSolution
from burnman_cpp import PolynomialSolution
from burnman_cpp.utils.chemistry import dictionarize_formula, formula_mass
from burnman_cpp.utils.chemistry import formula_to_string, sum_formulae
from burnman_cpp.minerals import HGP_2018_ds633
from burnman_cpp.tools.eos import check_eos_consistency
from burnman_cpp import transform_solution_to_new_basis


class forsterite(Mineral):

    def __init__(self):
        formula = "Mg2.0Si1.0O4.0"
        formula = dictionarize_formula(formula)
        params = {
            "name": "fo",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -2172590.0,
            "S_0": 95.1,
            "V_0": 4.366e-05,
            "Cp": [233.3, 0.001494, -603800.0, -1869.7],
            "a_0": 2.85e-05,
            "K_0": 128500000000.0,
            "Kprime_0": 3.84,
            "Kdprime_0": -3e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        params.setdefault("name", "forsterite")
        burnman.Mineral.__init__(self, params)


class fayalite(Mineral):

    def __init__(self):
        formula = "Fe2.0Si1.0O4.0"
        formula = dictionarize_formula(formula)
        params = {
            "name": "fa",
            "formula": formula,
            "equation_of_state": "hp_tmt",
            "H_0": -1477720.0,
            "S_0": 151.0,
            "V_0": 4.631e-05,
            "Cp": [201.1, 0.01733, -1960600.0, -900.9],
            "a_0": 2.82e-05,
            "K_0": 125600000000.0,
            "Kprime_0": 4.68,
            "Kdprime_0": -3.7e-11,
            "n": sum(formula.values()),
            "molar_mass": formula_mass(formula),
        }
        params.setdefault("name", "fayalite")
        burnman.Mineral.__init__(self, params)


made_forsterite = CombinedMineral([forsterite(), forsterite()], [0.5, 0.5])


class forsterite_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "Dummy solid solution"
        solution_model = SymmetricRegularSolution(
            endmembers=[[forsterite(), "[Mg]2SiO4"]], energy_interaction=[]
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class forsterite_forsterite_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "Fo-Fo solid solution"
        solution_model = SymmetricRegularSolution(
            endmembers=[[forsterite(), "[Mg]2SiO4"], [forsterite(), "[Mg]2SiO4"]],
            energy_interaction=[[0.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class olivine_ideal_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "Fo-Fo solid solution"
        solution_model = IdealSolution(
            endmembers=[[forsterite(), "[Mg]2SiO4"], [fayalite(), "[Fe]2SiO4"]]
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class olivine_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "Olivine"
        solution_model = SymmetricRegularSolution(
            endmembers=[[forsterite(), "[Mg]2SiO4"], [fayalite(), "[Fe]2SiO4"]],
            energy_interaction=[[8400.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class olivine_ss2(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "Olivine"
        solution_model = SymmetricRegularSolution(
            endmembers=[[made_forsterite, "[Mg]2SiO4"], [fayalite(), "[Fe]2SiO4"]],
            energy_interaction=[[8400.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class orthopyroxene(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "orthopyroxene"
        solution_model = SymmetricRegularSolution(
            endmembers=[
                [forsterite(), "[Mg][Mg]Si2O6"],
                [forsterite(), "[Mg1/2Al1/2][Mg1/2Al1/2]AlSiO6"],
            ],
            energy_interaction=[[burnman.constants.gas_constant * 1000.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss"
        solution_model = SymmetricRegularSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            energy_interaction=[[10000.0, 5000.0], [-10000.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_asymmetric(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (asymmetric)"
        solution_model = AsymmetricRegularSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            alphas=[1.0, 2.0, 2.0],
            energy_interaction=[[10000.0, 5000.0], [-10000.0]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_subregular(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular symmetric)"
        solution_model = SubregularSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            energy_interaction=[
                [[10000.0, 10000.0], [5000.0, 5000.0]],
                [[-10000.0, -10000.0]],
            ],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_subregular_asymmetric(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular symmetric)"
        solution_model = SubregularSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            energy_interaction=[
                [[10000.0, -10000.0], [5000.0, 3000.0]],
                [[-10000.0, -10000.0]],
            ],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_subregular_ternary(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular)"
        solution_model = SubregularSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            energy_interaction=[
                [[10000.0, -10000.0], [5000.0, 3000.0]],
                [[-10000.0, -10000.0]],
            ],
            entropy_interaction=[[[1.0, -2.0], [0.0, 1.0]], [[0.0, 0.0]]],
            energy_ternary_terms=[[0, 1, 2, 3000.0]],
            volume_ternary_terms=[[0, 1, 2, 3e-05]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_polynomial_ternary(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular from polynomial)"
        solution_model = PolynomialSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
            ],
            ESV_interactions=[
                [10000.0, 1.0, 0.0, 0, 1, 1, 2],
                [-10000.0, -2.0, 0.0, 0, 2, 1, 1],
                [5000.0, 0.0, 0.0, 0, 1, 2, 2],
                [3000.0, 1.0, 0.0, 0, 2, 2, 1],
                [-10000.0, 0.0, 0.0, 1, 1, 2, 2],
                [-10000.0, 0.0, 0.0, 1, 2, 2, 1],
                [5000.0, 0.5, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, -1.0, 0.0, 0, 1, 1, 1, 2, 1],
                [2500.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [1500.0, 0.5, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [3000.0, 0.0, 3e-05, 0, 1, 1, 1, 2, 1],
            ],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_polynomial_ternary_transformed(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular from polynomial)"
        solution_model = PolynomialSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
            ],
            ESV_interactions=[
                [10000.0, 1.0, 0.0, 0, 1, 1, 2],
                [-10000.0, -2.0, 0.0, 0, 2, 1, 1],
                [5000.0, 0.0, 0.0, 0, 1, 2, 2],
                [3000.0, 1.0, 0.0, 0, 2, 2, 1],
                [-10000.0, 0.0, 0.0, 1, 1, 2, 2],
                [-10000.0, 0.0, 0.0, 1, 2, 2, 1],
                [5000.0, 0.5, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, -1.0, 0.0, 0, 1, 1, 1, 2, 1],
                [2500.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [1500.0, 0.5, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [-5000.0, 0.0, 0.0, 0, 1, 1, 1, 2, 1],
                [3000.0, 0.0, 3e-05, 0, 1, 1, 1, 2, 1],
            ],
            transformation_matrix=np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            ),
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class two_site_ss_polynomial_high_order_transformed(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (high order from polynomial)"
        solution_model = PolynomialSolution(
            endmembers=[
                [forsterite(), "[Mg]3[Mg1/2Si1/2]2Si3O12"],
                [forsterite(), "[Mg]3[Al]2Si3O12"],
                [forsterite(), "[Fe]3[Al]2Si3O12"],
            ],
            ESV_interactions=[
                [10000.0, 1.0, 0.0, 0, 1, 1, 2],
                [-10000.0, -2.0, 0.0, 0, 2, 1, 6],
                [5000.0, 0.5, 0.0, 0, 6, 1, 1, 2, 1],
            ],
            transformation_matrix=np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            ),
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


fo1 = forsterite()
fo2 = forsterite()
fo2.set_property_modifiers(
    [["linear", {"delta_E": 1200.0, "delta_S": 5.0, "delta_V": 1e-06}]]
)


class polynomial_mbr_mixing(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "binary mbr mixing (polynomial)"
        solution_model = PolynomialSolution(
            endmembers=[[forsterite(), "[Mg]2"], [fayalite(), "[Mg]2"]],
            interaction_endmembers=[fo2, fo1],
            endmember_coefficients_and_interactions=[[1.0, -1.0, 0, 1, 1, 1]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class regular_copy_polynomial_mbr_mixing(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "binary mbr mixing (regular)"
        solution_model = SymmetricRegularSolution(
            endmembers=[[forsterite(), "[Mg]2"], [fayalite(), "[Mg]2"]],
            energy_interaction=[[1200.0]],
            entropy_interaction=[[5.0]],
            volume_interaction=[[1e-06]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


fo3 = forsterite()
fo3.set_property_modifiers(
    [["linear", {"delta_E": 600.0, "delta_S": 2.5, "delta_V": 5e-07}]]
)


class polynomial_esv_and_mbr_mixing(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular symmetric)"
        solution_model = PolynomialSolution(
            endmembers=[[forsterite(), "[Mg]2"], [fayalite(), "[Mg]2"]],
            ESV_interactions=[[600.0, 2.5, 5e-07, 0, 1, 1, 1]],
            interaction_endmembers=[fo3, fo1],
            endmember_coefficients_and_interactions=[[1.0, -1.0, 0, 1, 1, 1]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class polynomial_esv_and_mbr_mixing_2(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "two_site_ss (subregular symmetric)"
        solution_model = PolynomialSolution(
            endmembers=[[forsterite(), "[Mg]2"], [fayalite(), "[Mg]2"]],
            ESV_interactions=[[600.0, 2.5, 5e-07, 0, 1, 1, 1]],
            interaction_endmembers=[fayalite(), fo1],
            endmember_coefficients_and_interactions=[[1.0, -1.0, 0, 1, 1, 1]],
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class temkin_ss(burnman.Solution):

    def __init__(self, molar_fractions=None):
        name = "ol-q-water melt (HGP 2018)"
        solution_model = IdealSolution(
            endmembers=[
                [HGP_2018_ds633.foL(), "[Mg]4[Sitet]1[Vac]2"],
                [HGP_2018_ds633.faL(), "[Fe]4[Sitet]1[Vac]2"],
                [HGP_2018_ds633.qL(), "[]0[Sinet]1[Vac]2"],
                [HGP_2018_ds633.h2oL(), "[]0[]0[H]2"],
            ]
        )
        burnman.Solution.__init__(
            self,
            solution_model,
            (
                np.full(solution_model.n_endmembers, 1.0 / solution_model.n_endmembers)
                if molar_fractions is None
                else molar_fractions
            ),
            name=name,
        )


class test_solidsolution(BurnManTest):

    def setup_1min_ss(self):
        P = 100000.0
        T = 1000.0
        fo = forsterite()
        fo.set_state(P, T)
        fo_ss = forsterite_ss()
        fo_ss.set_composition([1.0])
        fo_ss.set_state(P, T)
        return (fo, fo_ss)

    def setup_2min_ss(self):
        P = 100000.0
        T = 1000.0
        fo = forsterite()
        fo.set_state(P, T)
        fo_fo_ss = forsterite_forsterite_ss()
        fo_fo_ss.set_composition([0.3, 0.7])
        fo_fo_ss.set_state(P, T)
        return (fo, fo_fo_ss)

    def setup_ol_ss(self):
        P = 100000.0
        T = 1000.0
        fo = forsterite()
        fo.set_state(P, T)
        ol_ss = olivine_ss()
        ol_ss.set_composition([1.0, 0.0])
        ol_ss.set_state(P, T)
        return (fo, ol_ss)

    def setup_ol_ss2(self):
        P = 100000.0
        T = 1000.0
        fo = forsterite()
        fo.set_state(P, T)
        ol_ss = olivine_ss2()
        ol_ss.set_composition([1.0, 0.0])
        ol_ss.set_state(P, T)
        return (fo, ol_ss)

    def test_ol_ss_formulae(self):
        P = 100000.0
        T = 1000.0
        ol_ss = olivine_ss()
        ol_ss.set_composition([0.4, 0.6])
        ol_ss.set_state(P, T)
        self.assertEqual(ol_ss.site_formula(), "[Mg0.40Fe0.60]2SiO4")

    def test_ol_ss_formulae_high_precision(self):
        P = 100000.0
        T = 1000.0
        ol_ss = olivine_ss()
        f = 0.43673678
        ol_ss.set_composition([f, 1 - f])
        ol_ss.set_state(P, T)
        self.assertEqual(ol_ss.site_formula(precision=4), "[Mg0.4367Fe0.5633]2SiO4")

    def test_ol_ss_formulae_absent_species(self):
        P = 100000.0
        T = 1000.0
        ol_ss = olivine_ss()
        ol_ss.set_composition([0.0, 1.0])
        ol_ss.set_state(P, T)
        self.assertEqual(ol_ss.site_formula(), "[Mg0.00Fe1.00]2SiO4")
        self.assertEqual(
            ol_ss.site_formula(print_absent_species=False), "[Fe1.00]2SiO4"
        )

    def test_ol_ss_formulae_custom_site_occupancies(self):
        P = 100000.0
        T = 1000.0
        ol_ss = olivine_ss()
        ol_ss.set_composition([0.4, 0.6])
        ol_ss.set_state(P, T)
        self.assertEqual(
            ol_ss.site_formula(site_occupancies=[1.0, 2.0]), "[Mg1.00Fe2.00]2SiO4"
        )

    def test_ol_gibbs(self):
        fo, fo_ss = self.setup_ol_ss()
        endmember_properties = [
            fo.molar_gibbs,
            fo.molar_enthalpy,
            fo.molar_entropy,
            fo.molar_volume,
            fo.molar_heat_capacity_p,
            fo.molar_heat_capacity_v,
            fo.thermal_expansivity,
            fo.isothermal_bulk_modulus_reuss,
            fo.isentropic_bulk_modulus_reuss,
            fo.grueneisen_parameter,
        ]
        ss_properties = [
            fo_ss.molar_gibbs,
            fo_ss.molar_enthalpy,
            fo_ss.molar_entropy,
            fo_ss.molar_volume,
            fo_ss.molar_heat_capacity_p,
            fo_ss.molar_heat_capacity_v,
            fo_ss.thermal_expansivity,
            fo_ss.isothermal_bulk_modulus_reuss,
            fo_ss.isentropic_bulk_modulus_reuss,
            fo_ss.grueneisen_parameter,
        ]
        self.assertArraysAlmostEqual(endmember_properties, ss_properties)

    def test_ol_gibbs2(self):
        fo, fo_ss = self.setup_ol_ss2()
        endmember_properties = [
            fo.molar_gibbs,
            fo.molar_enthalpy,
            fo.molar_entropy,
            fo.molar_volume,
            fo.molar_heat_capacity_p,
            fo.molar_heat_capacity_v,
            fo.thermal_expansivity,
            fo.isothermal_bulk_modulus_reuss,
            fo.isentropic_bulk_modulus_reuss,
            fo.grueneisen_parameter,
        ]
        ss_properties = [
            fo_ss.molar_gibbs,
            fo_ss.molar_enthalpy,
            fo_ss.molar_entropy,
            fo_ss.molar_volume,
            fo_ss.molar_heat_capacity_p,
            fo_ss.molar_heat_capacity_v,
            fo_ss.thermal_expansivity,
            fo_ss.isothermal_bulk_modulus_reuss,
            fo_ss.isentropic_bulk_modulus_reuss,
            fo_ss.grueneisen_parameter,
        ]
        self.assertArraysAlmostEqual(endmember_properties, ss_properties)

    def test_set_state_with_volume(self):
        m = olivine_ss()
        m.set_composition([0.5, 0.5])
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

    def test_ol_Wh(self):
        ol_ss = olivine_ss()
        H_excess = ol_ss.solution_model.compute_excess_enthalpy(
            100000.0, 1000.0, [0.5, 0.5]
        )
        We = ol_ss.solution_model.We[0][1]
        self.assertArraysAlmostEqual([We / 4.0], [H_excess])

    def test_order_disorder(self):
        opx = orthopyroxene()
        opx.set_composition(np.array([0.0, 1.0]))
        opx.set_state(100000.0, 300.0)
        self.assertArraysAlmostEqual([opx.excess_gibbs], [0.0])

    def test_site_totals(self):
        ss = two_site_ss()
        ss.set_composition([0.3, 0.3, 0.4])
        ss.set_state(100000.0, 300.0)
        site_fractions = np.dot(
            ss.molar_fractions, ss.solution_model.endmember_occupancies
        )
        i = 0
        site_fill = []
        ones = [1.0] * ss.solution_model.n_sites
        for site in ss.solution_model.sites:
            site_fill.append(sum(site_fractions[i : i + len(site)]))
            i += len(site)
        self.assertArraysAlmostEqual(site_fill, ones)

    def test_set_method(self):
        ss = olivine_ss()
        ss.set_method(native_method("hp_tmt"))

    def test_molar_mass(self):
        ss = olivine_ss()
        ss.set_composition(np.array([0.5, 0.5]))
        self.assertArraysAlmostEqual(
            [ss.molar_mass],
            [
                0.5 * parameter_dict(forsterite().params)["molar_mass"]
                + 0.5 * parameter_dict(fayalite().params)["molar_mass"]
            ],
        )

    def test_asymmetric_model_hessian_one_component_change(self):
        ss = two_site_ss_asymmetric()
        f0 = np.array([0.25, 0.35, 0.4])
        ss.set_state(100000.0, 300.0)
        dp = 1e-06
        H = np.zeros((3, 3))
        for i in range(3):
            f = np.array(f0)
            f[i] -= dp / 2.0
            f /= np.sum(f)
            ss.set_composition(f)
            dGdx1 = ss.partial_gibbs
            f = np.array(f0)
            f[i] += dp / 2.0
            f /= np.sum(f)
            ss.set_composition(f)
            dGdx2 = ss.partial_gibbs
            H[i, :] = (dGdx2 - dGdx1) / dp
        ss.set_composition(f0)
        for i in range(3):
            self.assertArraysAlmostEqual(ss.gibbs_hessian[i], H[i])

    def test_subregular_model_hessian_one_component_change(self):
        ss = two_site_ss_subregular()
        f0 = np.array([0.25, 0.35, 0.4])
        ss.set_state(100000.0, 300.0)
        dp = 1e-06
        H = np.zeros((3, 3))
        for i in range(3):
            f = np.array(f0)
            f[i] -= dp / 2.0
            f /= np.sum(f)
            ss.set_composition(f)
            dGdx1 = ss.partial_gibbs
            f = np.array(f0)
            f[i] += dp / 2.0
            f /= np.sum(f)
            ss.set_composition(f)
            dGdx2 = ss.partial_gibbs
            H[i, :] = (dGdx2 - dGdx1) / dp
        ss.set_composition(f0)
        for i in range(3):
            self.assertArraysAlmostEqual(ss.gibbs_hessian[i], H[i])

    def test_asymmetric_model_hessian_multicomponent_change(self):
        ss = two_site_ss_asymmetric()
        f0 = np.array([0.25, 0.35, 0.4])
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        H0 = ss.gibbs_hessian
        df = np.array([2e-06, -1.5e-06, -5e-07])
        ss.set_composition(f0 - df / 2.0)
        dGdx1 = ss.partial_gibbs
        ss.set_composition(f0 + df / 2.0)
        dGdx2 = ss.partial_gibbs
        self.assertArraysAlmostEqual(H0.dot(df), dGdx2 - dGdx1)

    def test_subregular_model_hessian_multicomponent_change(self):
        ss = two_site_ss_subregular_asymmetric()
        f0 = [0.25, 0.35, 0.4]
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        H0 = ss.gibbs_hessian
        df = np.array([2e-06, -1.5e-06, -5e-07])
        ss.set_composition(f0 - df / 2.0)
        dGdx1 = ss.partial_gibbs
        ss.set_composition(f0 + df / 2.0)
        dGdx2 = ss.partial_gibbs
        self.assertArraysAlmostEqual(H0.dot(df), dGdx2 - dGdx1)

    def test_subregular_model_ternary_partial_entropy_multicomponent_change(self):
        ss = two_site_ss_subregular_ternary()
        f0 = np.array([0.25, 0.35, 0.4])
        ss.set_composition(f0)
        ss.set_state(1000000000.0, 1000.0)
        dSdx = ss.partial_entropies
        df = 0.0001
        dSdx2 = np.empty(3)
        for i, f_mod in enumerate(np.eye(3) * df):
            ss.set_composition((f0 - f_mod / 2) / (1.0 - df / 2.0))
            S0 = ss.molar_entropy * (1.0 - df / 2.0)
            ss.set_composition((f0 + f_mod / 2) / (1.0 + df / 2.0))
            S1 = ss.molar_entropy * (1.0 + df / 2.0)
            dSdx2[i] = (S1 - S0) / df
        self.assertArraysAlmostEqual(dSdx, dSdx2)

    def test_subregular_model_ternary_hessian_multicomponent_change(self):
        ss = two_site_ss_subregular_ternary()
        f0 = [0.25, 0.35, 0.4]
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        H0 = ss.gibbs_hessian
        df = np.array([2e-06, -1.5e-06, -5e-07])
        ss.set_composition(f0 - df / 2.0)
        dGdx1 = ss.partial_gibbs
        ss.set_composition(f0 + df / 2.0)
        dGdx2 = ss.partial_gibbs
        self.assertArraysAlmostEqual(H0.dot(df), dGdx2 - dGdx1)

    def test_slb_ferropericlase_partial_gibbs_multicomponent_change(self):
        ss = burnman.minerals.SLB_2024.ferropericlase()
        f0 = [0.15, 0.25, 0.35, 0.05, 0.2]
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        mu0 = ss.partial_gibbs
        df = np.array([2e-06, -1.5e-06, -5e-07, 2.5e-07, -2.5e-07])
        ss.set_composition(f0 - df / 2.0)
        G1 = ss.molar_gibbs
        ss.set_composition(f0 + df / 2.0)
        G2 = ss.molar_gibbs
        self.assertAlmostEqual(mu0.dot(df), G2 - G1)

    def test_slb_ferropericlase_hessian_multicomponent_change(self):
        ss = burnman.minerals.SLB_2024.ferropericlase()
        f0 = [0.15, 0.25, 0.35, 0.05, 0.2]
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        H0 = ss.gibbs_hessian
        df = np.array([2e-06, -1.5e-06, -5e-07, 2.5e-07, -2.5e-07])
        ss.set_composition(f0 - df / 2.0)
        dGdx1 = ss.partial_gibbs
        ss.set_composition(f0 + df / 2.0)
        dGdx2 = ss.partial_gibbs
        self.assertArraysAlmostEqual(H0.dot(df), dGdx2 - dGdx1)

    def test_polynomial_model_partial_entropy_multicomponent_change(self):
        ss = two_site_ss_polynomial_ternary()
        f0 = np.array([0.25, 0.35, 0.4])
        ss.set_composition(f0)
        ss.set_state(1000000000.0, 1000.0)
        dSdx = ss.partial_entropies
        df = 0.0001
        dSdx2 = np.empty(3)
        for i, f_mod in enumerate(np.eye(3) * df):
            ss.set_composition((f0 - f_mod / 2) / (1.0 - df / 2.0))
            S0 = ss.molar_entropy * (1.0 - df / 2.0)
            ss.set_composition((f0 + f_mod / 2) / (1.0 + df / 2.0))
            S1 = ss.molar_entropy * (1.0 + df / 2.0)
            dSdx2[i] = (S1 - S0) / df
        self.assertArraysAlmostEqual(dSdx, dSdx2)

    def test_polynomial_high_order_transformed(self):
        ss = two_site_ss_polynomial_high_order_transformed()
        f0 = [0.25, 0.35, 0.4]
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        HG0 = ss.gibbs_hessian
        HS0 = ss.entropy_hessian
        HV0 = ss.volume_hessian
        dGdx0 = ss.partial_gibbs
        dSdx0 = ss.partial_entropies
        dVdx0 = ss.partial_volumes
        df = np.array([2e-06, -1.5e-06, -5e-07])
        ss.set_composition(f0 - df / 2.0)
        G1 = ss.molar_gibbs
        S1 = ss.molar_entropy
        V1 = ss.molar_volume
        dGdx1 = ss.partial_gibbs
        dSdx1 = ss.partial_entropies
        dVdx1 = ss.partial_volumes
        ss.set_composition(f0 + df / 2.0)
        G2 = ss.molar_gibbs
        S2 = ss.molar_entropy
        V2 = ss.molar_volume
        dGdx2 = ss.partial_gibbs
        dSdx2 = ss.partial_entropies
        dVdx2 = ss.partial_volumes
        self.assertArraysAlmostEqual(
            [dGdx0.dot(df), dSdx0.dot(df), dVdx0.dot(df)], [G2 - G1, S2 - S1, V2 - V1]
        )
        self.assertArraysAlmostEqual(HG0.dot(df), dGdx2 - dGdx1)
        self.assertArraysAlmostEqual(HS0.dot(df), dSdx2 - dSdx1)
        self.assertArraysAlmostEqual(HV0.dot(df), dVdx2 - dVdx1)

    def test_polynomial_using_subregular_scalars(self):
        ss1 = two_site_ss_subregular_ternary()
        ss2 = two_site_ss_polynomial_ternary()
        f0 = [0.25, 0.35, 0.4]
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
            ss.set_composition(f0)
        ss1_properties = [
            ss1.molar_gibbs,
            ss1.molar_enthalpy,
            ss1.molar_entropy,
            ss1.molar_volume,
            ss1.molar_heat_capacity_p,
            ss1.molar_heat_capacity_v,
            ss1.thermal_expansivity,
            ss1.isothermal_bulk_modulus_reuss,
            ss1.isentropic_bulk_modulus_reuss,
            ss1.grueneisen_parameter,
        ]
        ss2_properties = [
            ss2.molar_gibbs,
            ss2.molar_enthalpy,
            ss2.molar_entropy,
            ss2.molar_volume,
            ss2.molar_heat_capacity_p,
            ss2.molar_heat_capacity_v,
            ss2.thermal_expansivity,
            ss2.isothermal_bulk_modulus_reuss,
            ss2.isentropic_bulk_modulus_reuss,
            ss2.grueneisen_parameter,
        ]
        self.assertArraysAlmostEqual(ss1_properties, ss2_properties)

    def test_polynomial_using_subregular_gibbs(self):
        ss1 = two_site_ss_subregular_ternary()
        ss2 = two_site_ss_polynomial_ternary()
        f0 = [0.25, 0.35, 0.4]
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
            ss.set_composition(f0)
        self.assertArraysAlmostEqual(ss1.excess_partial_gibbs, ss2.excess_partial_gibbs)
        self.assertArraysAlmostEqual(
            ss1.gibbs_hessian.flatten(), ss2.gibbs_hessian.flatten()
        )

    def test_polynomial_using_subregular_entropy(self):
        ss1 = two_site_ss_subregular_ternary()
        ss2 = two_site_ss_polynomial_ternary()
        f0 = [0.25, 0.35, 0.4]
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
            ss.set_composition(f0)
        self.assertArraysAlmostEqual(
            ss1.excess_partial_entropies, ss2.excess_partial_entropies
        )
        self.assertArraysAlmostEqual(
            ss1.entropy_hessian.flatten(), ss2.entropy_hessian.flatten()
        )

    def test_polynomial_using_subregular_volume(self):
        ss1 = two_site_ss_subregular_ternary()
        ss2 = two_site_ss_polynomial_ternary()
        f0 = [0.25, 0.35, 0.4]
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
            ss.set_composition(f0)
        self.assertArraysAlmostEqual(
            ss1.excess_partial_volumes, ss2.excess_partial_volumes
        )
        self.assertArraysAlmostEqual(
            ss1.volume_hessian.flatten(), ss2.volume_hessian.flatten()
        )
        self.assertArraysAlmostEqual(ss1.partial_gibbs, ss2.partial_gibbs)
        self.assertArraysAlmostEqual(ss1.partial_entropies, ss2.partial_entropies)
        self.assertArraysAlmostEqual(ss1.partial_volumes, ss2.partial_volumes)

    def test_polynomial_consistency(self):
        ss = polynomial_esv_and_mbr_mixing_2()
        ss.set_composition([0.4, 0.6])
        self.assertTrue(check_eos_consistency(ss))

    def test_polynomial_mbr_mixing(self):
        ss1 = polynomial_mbr_mixing()
        ss2 = regular_copy_polynomial_mbr_mixing()
        f0 = [0.25, 0.75]
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
            ss.set_composition(f0)
        self.assertAlmostEqual(
            ss1.excess_gibbs, 0.25 * 0.75 * (1200.0 - 300.0 * 5 + 100000.0 * 1e-06)
        )
        self.assertAlmostEqual(ss1.excess_entropy, 0.25 * 0.75 * 5)
        self.assertAlmostEqual(ss1.excess_volume, 0.25 * 0.75 * 1e-06)
        self.assertArraysAlmostEqual(ss1.partial_gibbs, ss2.partial_gibbs)
        self.assertArraysAlmostEqual(ss1.partial_entropies, ss2.partial_entropies)
        self.assertArraysAlmostEqual(ss1.partial_volumes, ss2.partial_volumes)

    def test_polynomial_ESV_and_mbr_mixing(self):
        ss = polynomial_esv_and_mbr_mixing()
        f0 = [0.25, 0.75]
        ss.set_state(100000.0, 300.0)
        ss.set_composition(f0)
        self.assertAlmostEqual(
            ss.excess_gibbs, 0.25 * 0.75 * (1200.0 - 300.0 * 5 + 100000.0 * 1e-06)
        )
        self.assertAlmostEqual(ss.excess_entropy, 0.25 * 0.75 * 5)
        self.assertAlmostEqual(ss.excess_volume, 0.25 * 0.75 * 1e-06)

    def test_polynomial_ternary_transformed(self):
        ss1 = two_site_ss_polynomial_ternary()
        ss2 = two_site_ss_polynomial_ternary_transformed()
        for ss in [ss1, ss2]:
            ss.set_state(100000.0, 300.0)
        f = np.array([0.25, 0.35, 0.4])
        ss1.set_composition(f)
        ss2.set_composition(np.roll(f, 1))
        self.assertAlmostEqual(ss1.excess_gibbs, ss2.excess_gibbs)
        self.assertArraysAlmostEqual(ss1.partial_gibbs, np.roll(ss2.partial_gibbs, -1))
        self.assertArraysAlmostEqual(
            ss1.partial_entropies, np.roll(ss2.partial_entropies, -1)
        )
        self.assertArraysAlmostEqual(
            ss1.partial_volumes, np.roll(ss2.partial_volumes, -1)
        )

    def test_subregular(self):
        ss0 = two_site_ss()
        ss1 = two_site_ss_subregular()
        ss0.set_composition([0.3, 0.3, 0.4])
        ss0.set_state(100000.0, 300.0)
        ss1.set_composition([0.3, 0.3, 0.4])
        ss1.set_state(100000.0, 300.0)
        self.assertArraysAlmostEqual(ss0.excess_partial_gibbs, ss1.excess_partial_gibbs)

    def test_activities_ideal(self):
        ol = olivine_ideal_ss()
        ol.set_composition(np.array([0.5, 0.5]))
        ol.set_state(100000.0, 1000.0)
        self.assertArraysAlmostEqual(ol.activities, [0.25, 0.25])
        self.assertArraysAlmostEqual(ol.activity_coefficients, [1.0, 1.0])

    def test_entropy_ideal(self):
        ol = olivine_ideal_ss()
        ol.set_composition(np.array([0.5, 0.5]))
        ol.set_state(100000.0, 1000.0)
        R = burnman.constants.gas_constant
        # These ordered endmembers have zero configurational entropy, so
        # native excess entropy equals the mixture's configurational entropy.
        self.assertAlmostEqual(
            ol.solution_model.compute_excess_entropy(
                100000.0, 1000.0, ol.molar_fractions
            ),
            2.0 * np.log(2) * R,
        )
        S_hess = 2 * np.array([[-R, R], [R, -R]])
        np.testing.assert_allclose(ol.entropy_hessian, S_hess)
        np.testing.assert_allclose(ol.volume_hessian, np.zeros((2, 2)))

    def test_activity_coefficients_non_ideal(self):
        opx = orthopyroxene()
        opx.set_composition(np.array([0.0, 1.0]))
        opx.set_state(100000.0, 1000.0)
        self.assertArraysAlmostEqual(opx.activity_coefficients, [np.exp(1.0), 1.0])

    def test_formula(self):
        ol = olivine_ideal_ss()
        ol.set_composition([0.5, 0.5])
        self.assertEqual(formula_to_string(ol.formula), "MgFeSiO4")
        self.assertEqual(
            formula_to_string(sum_formulae(ol.endmember_formulae, [0.5, 0.5])),
            "MgFeSiO4",
        )
        s = np.array([[2, 0, 1, 4], [0, 2, 1, 4]])
        self.assertArraysAlmostEqual(ol.stoichiometric_matrix[0], s[0])
        self.assertArraysAlmostEqual(ol.stoichiometric_matrix[1], s[1])

    def test_stoichiometric_matrix_binary_solution(self):
        olivine = burnman.minerals.SLB_2011.mg_fe_olivine()
        self.assertTrue(len(olivine.endmember_names) == 2)
        self.assertTrue(len(olivine.elements) == 4)
        self.assertTrue(olivine.stoichiometric_matrix.shape == (2, 4))
        self.assertTrue(olivine.reaction_basis.shape[0] == 0)
        self.assertArraysAlmostEqual(
            olivine.compositional_null_basis[0], [-1.0 / 2.0, -1.0 / 2.0, 1, 0]
        )
        self.assertArraysAlmostEqual(
            olivine.compositional_null_basis[1], [-2.0, -2.0, 0, 1]
        )
        self.assertArraysAlmostEqual(olivine.independent_element_indices, [0, 1])
        self.assertArraysAlmostEqual(olivine.dependent_element_indices, [2, 3])
        self.assertTrue(olivine.n_reactions == 0)

    def test_stoichiometric_matrix_complex_solution(self):
        opx = burnman.minerals.JH_2015.orthopyroxene()
        self.assertTrue(len(opx.endmember_names) == 7)
        self.assertTrue(len(opx.elements) == 7)
        self.assertTrue(opx.stoichiometric_matrix.shape == (7, 7))
        self.assertTrue(opx.reaction_basis.shape[0] == 1)
        self.assertArraysAlmostEqual(
            opx.compositional_null_basis[0],
            [-3.0 / 2.0, -3.0 / 2.0, -3.0 / 2.0, -3.0 / 2.0, -3.0 / 2.0, -3.0 / 2.0, 1],
        )
        self.assertArraysAlmostEqual(
            opx.independent_element_indices, [0, 1, 2, 3, 4, 5]
        )
        self.assertArraysAlmostEqual(opx.dependent_element_indices, [6])
        self.assertTrue(opx.n_reactions == 1)
        self.assertArraysAlmostEqual(
            opx.reaction_basis[0], [-1.0 / 2.0, -1.0 / 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_temkin_entropies(self):
        ss = temkin_ss()
        R = burnman.constants.gas_constant
        ss.set_state(1000000000.0, 600.0)
        S = np.empty((4, 2))
        ss.set_composition([0.5, 0.5, 0.0, 0.0])
        S[0] = [ss.excess_entropy, -4.0 * R * np.log(0.5)]
        ss.set_composition([0.5, 0.0, 0.5, 0.0])
        S[1] = [ss.excess_entropy, -R * np.log(0.5)]
        ss.set_composition([0.25, 0.25, 0.5, 0.0])
        S[2] = [ss.excess_entropy, -3.0 * R * np.log(0.5)]
        ss.set_composition([0.0, 0.0, 0.5, 0.5])
        S[3] = [ss.excess_entropy, -2.0 * R * np.log(0.5)]
        self.assertArraysAlmostEqual(S[:, 0], S[:, 1])

    def test_temkin_activities(self):
        ss = temkin_ss()
        f0 = np.array([0.25, 0.35, 0.3, 0.1])
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        S_partial = ss.excess_partial_entropies
        activities = ss.activities
        self.assertArraysAlmostEqual(
            np.exp(-S_partial / burnman.constants.gas_constant), activities
        )

    def test_temkin_partial_gibbs(self):
        ss = temkin_ss()
        f0 = np.array([0.25, 0.35, 0.3, 0.1])
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        dGdx = ss.partial_gibbs
        df = 0.0001
        dGdx2 = np.empty(4)
        for i, f_mod in enumerate(np.eye(4) * df):
            ss.set_composition((f0 - f_mod / 2) / (1.0 - df / 2.0))
            G0 = ss.molar_gibbs * (1.0 - df / 2.0)
            ss.set_composition((f0 + f_mod / 2) / (1.0 + df / 2.0))
            G1 = ss.molar_gibbs * (1.0 + df / 2.0)
            dGdx2[i] = (G1 - G0) / df
        self.assertArraysAlmostEqual(dGdx, dGdx2)

    def test_temkin_hessian(self):
        ss = temkin_ss()
        f0 = np.array([0.25, 0.35, 0.3, 0.1])
        ss.set_composition(f0)
        ss.set_state(100000.0, 300.0)
        H0 = ss.gibbs_hessian
        df = np.array([2e-06, -1.5e-06, -6e-07, 1e-07])
        ss.set_composition(f0 - df / 2.0)
        dGdx1 = ss.partial_gibbs
        ss.set_composition(f0 + df / 2.0)
        dGdx2 = ss.partial_gibbs
        self.assertArraysAlmostEqual(H0.dot(df), dGdx2 - dGdx1)

    def test_relaxed_solution(self):
        opx = burnman.minerals.JH_2015.orthopyroxene()
        ropx = burnman.RelaxedSolution(opx, opx.reaction_basis, opx.compositional_basis)
        ropx.set_composition([0.2, 0.2, 0.1, 0.1, 0.2, 0.2])
        for T in [300.0, 1000.0, 2000.0]:
            ropx.set_state(1000000000.0, T)
            mu = np.einsum("ij, j", opx.reaction_basis, ropx.partial_gibbs)[0]
            self.assertTrue(np.abs(mu) < 5.0)
        V = ropx.molar_volume
        ropx.set_state(10000000000.0, T)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ropx.set_state_with_volume(V * 1.000001, T)
            self.assertTrue(ropx.pressure < 1000000000.0)
            ropx.set_state_with_volume(V * 0.999999, T)
            self.assertTrue(ropx.pressure > 1000000000.0)
        self.assertAlmostEqual(
            ropx.isothermal_compressibility_reuss,
            1.0 / ropx.isothermal_bulk_modulus_reuss,
        )
        self.assertAlmostEqual(
            ropx.isentropic_compressibility_reuss,
            1.0 / ropx.isentropic_bulk_modulus_reuss,
        )
        self.assertTrue(
            ropx.isothermal_compressibility_reuss
            > ropx.isentropic_compressibility_reuss
        )
        self.assertTrue(ropx.molar_heat_capacity_p > ropx.molar_heat_capacity_v)
        # Native RelaxedSolution stores its own model rather than an
        # unrelaxed child; compare the same composition and state explicitly.
        opx.set_composition(ropx.molar_fractions)
        opx.set_state(ropx.pressure, ropx.temperature)
        self.assertTrue(ropx.thermal_expansivity > opx.thermal_expansivity)
        self.assertTrue(ropx.molar_heat_capacity_p > opx.molar_heat_capacity_p)
        self.assertTrue(
            ropx.isothermal_compressibility_reuss > opx.isothermal_compressibility_reuss
        )

    def test_transform_ideal(self):
        ol = olivine_ideal_ss()
        new_ol = transform_solution_to_new_basis(ol, [[0.25, 0.75], [0.75, 0.25]])
        P = 1000000000.0
        T = 1000.0
        ol.set_composition([0.5, 0.5])
        new_ol.set_composition([0.5, 0.5])
        ol.set_state(P, T)
        new_ol.set_state(P, T)
        old_formula = burnman.utils.chemistry.formula_to_string(ol.formula)
        new_formula = burnman.utils.chemistry.formula_to_string(new_ol.formula)
        self.assertEqual(old_formula, new_formula)
        self.assertAlmostEqual(ol.molar_gibbs, new_ol.molar_gibbs)

    def test_transform_asymmetric(self):
        cpx = burnman.minerals.JH_2015.clinopyroxene()
        new_cpx = transform_solution_to_new_basis(
            cpx,
            [[1, 1, 0, 0, 0, 0, 0, -1], [0, 0, 1, 0, 0, 0, 0, 0]],
            solution_name="fdi-cats",
        )
        P = 1000000000.0
        T = 1000.0
        cpx.set_composition([0.5, 0.5, 0.5, 0, 0, 0, 0, -0.5])
        new_cpx.set_composition([0.5, 0.5])
        cpx.set_state(P, T)
        new_cpx.set_state(P, T)
        old_formula = burnman.utils.chemistry.formula_to_string(cpx.formula)
        new_formula = burnman.utils.chemistry.formula_to_string(new_cpx.formula)
        self.assertEqual(old_formula, new_formula)
        self.assertAlmostEqual(cpx.molar_gibbs, new_cpx.molar_gibbs)

    def test_transform_asymmetric_new_disordered_endmember(self):
        cpx = burnman.minerals.JH_2015.clinopyroxene()
        new_cpx = transform_solution_to_new_basis(
            cpx,
            [[0.5, 0.5, 0, 0, 0, 0, 0, 0.0], [0, 0, 1, 0, 0, 0, 0, 0]],
            solution_name="fdi-cats",
        )
        P = 1000000000.0
        T = 1000.0
        cpx.set_composition([0.25, 0.25, 0.5, 0, 0, 0, 0, 0])
        new_cpx.set_composition([0.5, 0.5])
        cpx.set_state(P, T)
        new_cpx.set_state(P, T)
        old_formula = burnman.utils.chemistry.formula_to_string(cpx.formula)
        new_formula = burnman.utils.chemistry.formula_to_string(new_cpx.formula)
        self.assertEqual(old_formula, new_formula)
        self.assertAlmostEqual(cpx.molar_gibbs, new_cpx.molar_gibbs)

    def test_mechanical_solution_model_functions(self):
        sm = burnman.MechanicalSolution([[forsterite(), ""], [fayalite(), ""]])
        P = 1000000000.0
        T = 1000.0
        f = np.array([1.0, 0.0])
        pGs = sm.compute_excess_partial_gibbs_free_energies(P, T, f)
        pSs = sm.compute_excess_partial_entropies(P, T, f)
        pVs = sm.compute_excess_partial_volumes(P, T, f)
        H_xs = sm.compute_excess_enthalpy(P, T, f)
        gammas = sm.compute_activity_coefficients(P, T, f)
        alphas = sm.compute_activities(P, T, f)
        zeros = np.zeros(2)
        ones = np.ones(2)
        np.testing.assert_allclose(pGs, zeros)
        np.testing.assert_allclose(pSs, zeros)
        np.testing.assert_allclose(pVs, zeros)
        np.testing.assert_allclose(gammas, ones)
        np.testing.assert_allclose(alphas, ones)
        np.testing.assert_almost_equal(H_xs, 0.0)
