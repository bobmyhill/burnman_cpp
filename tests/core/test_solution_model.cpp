// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

// Numerical snapshots: Python BurnMan 69e700647ae7efeed91dfbce147c036bce619516.
// Reproduce with tools/check_reference_data.py; see docs/reference_data.md.
#include "burnman/core/mineral.hpp"
#include "burnman/core/solution_model.hpp"
#include "tolerances.hpp"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <map>
#include <string>
#include <vector>

using namespace Catch::Matchers;
using namespace burnman;

namespace burnman::solution_models {
struct RegularSolutionTestHelper {
  static const Eigen::ArrayXd &
  get_alphas(const solution_models::AsymmetricRegularSolution &sol) {
    return sol.alphas;
  }

  static const Eigen::MatrixXd &
  get_W_e(const solution_models::AsymmetricRegularSolution &sol) {
    return sol.W_e;
  }

  static const Eigen::MatrixXd &
  get_W_s(const solution_models::AsymmetricRegularSolution &sol) {
    return sol.W_s;
  }

  static const Eigen::MatrixXd &
  get_W_v(const solution_models::AsymmetricRegularSolution &sol) {
    return sol.W_v;
  }
};
} // namespace burnman::solution_models

struct OlivineFixture {
  Mineral forsterite;
  Mineral fayalite;
  types::PairedEndmemberList olivine_endmembers;

  OlivineFixture() {
    // Set fo parameters
    forsterite.params.name = "Forsterite";
    forsterite.params.formula =
        types::FormulaMap{{"Mg", 2.0}, {"Si", 1.0}, {"O", 4.0}};
    forsterite.params.napfu = 7;
    forsterite.params.molar_mass = 0.1406931;
    forsterite.params.V_0 = 4.366e-05;
    forsterite.params.K_0 = 1.285e11;
    forsterite.params.Kprime_0 = 3.84;
    forsterite.params.Kdprime_0 = -3e-11;
    forsterite.params.H_0 = -2172590.0;
    forsterite.params.S_0 = 95.1;
    forsterite.params.Cp = types::CpParams{233.3, 0.001494, -603800.0, -1869.7};
    forsterite.params.a_0 = 2.85e-05;
    forsterite.params.equation_of_state = types::EOSType::SLB3; // TODO! Change!
    // Params not needed here... use for Solution test

    // Fayalite
    fayalite.params.name = "Fayalite";
    fayalite.params.formula =
        types::FormulaMap{{"Fe", 2.0}, {"Si", 1.0}, {"O", 4.0}};
    fayalite.params.napfu = 7;
    fayalite.params.molar_mass = 0.2037731;
    fayalite.params.V_0 = 4.631e-05;
    fayalite.params.K_0 = 1.256e11;
    fayalite.params.Kprime_0 = 4.68;
    fayalite.params.Kdprime_0 = -3.7e-11;
    fayalite.params.H_0 = -1477720.0;
    fayalite.params.S_0 = 151.0;
    fayalite.params.Cp = types::CpParams{201.1, 0.01733, -1960600.0, -900.9};
    fayalite.params.a_0 = 2.82e-05;
    fayalite.params.equation_of_state = types::EOSType::SLB3; // TODO!

    // Paired list for solution model setup
    olivine_endmembers = {{forsterite, "[Mg]2SiO4"}, {fayalite, "[Fe]2SiO4"}};
  }
};

struct MultiSiteFixture {
  Mineral min;
  types::PairedEndmemberList em;
  MultiSiteFixture() {
    em = {{min, "[Mg]3[Al]2Si3O12"},
          {min, "[Fe]3[Al]2Si3O12"},
          {min, "[Mg]3[Mg1/2Si1/2]2Si3O12"}};
  }
};

TEST_CASE_METHOD(OlivineFixture, "IdealSolution", "[core][solution_model]") {
  // Create IdealSolution model
  solution_models::IdealSolution ol_ss(olivine_endmembers);
  SECTION("Derived parameters") {
    // Check endmembers (Mineral objects) can be accessed in order
    REQUIRE(ol_ss.endmembers[0].get_name() == "Forsterite");
    REQUIRE(ol_ss.endmembers[1].get_name() == "Fayalite");
    // ints
    CHECK(ol_ss.get_n_endmembers() == 2);
    CHECK(ol_ss.get_n_occupancies() == 2);
    CHECK(ol_ss.get_n_sites() == 1);
    // strings
    CHECK(ol_ss.get_empty_formula() == "[]2SiO4");
    CHECK(ol_ss.get_general_formula() == "[Mg,Fe]2SiO4");
    CHECK(ol_ss.get_formulas() ==
          std::vector<std::string>{"[Mg]2SiO4", "[Fe]2SiO4"});
    CHECK(ol_ss.get_site_names() == std::vector<std::string>{"Mg_A", "Fe_A"});
    CHECK(ol_ss.get_sites() ==
          std::vector<std::vector<std::string>>{{"Mg", "Fe"}});
    CHECK(ol_ss.get_solution_formulae() ==
          std::vector<std::map<std::string, double>>{{{"Mg", 2.0}},
                                                     {{"Fe", 2.0}}});
    // Eigen - can do CHECK(m1.isApprox(m2, rel_tolerance));
    CHECK(ol_ss.get_site_multiplicities().isApprox(
        (Eigen::ArrayXXd(2, 2) << 2.0, 2.0, 2.0, 2.0).finished(), tol_rel));
    CHECK(ol_ss.get_endmember_occupancies().isApprox(
        (Eigen::ArrayXXd(2, 2) << 1, 0, 0, 1).finished(), tol_rel));
    CHECK(ol_ss.get_endmember_n_occupancies().isApprox(
        (Eigen::ArrayXXd(2, 2) << 2, 0, 0, 2).finished(), tol_rel));
  }
  SECTION("Default returns") {
    CHECK(ol_ss.compute_Cp_excess() == 0);
    CHECK(ol_ss.compute_alphaV_excess() == 0);
    CHECK(ol_ss.compute_VoverKT_excess() == 0);
    // Dummy args
    double a = 0;
    double b = 0;
    Eigen::ArrayXd c = Eigen::ArrayXd::Zero(2);
    CHECK(ol_ss.compute_excess_partial_volumes(a, b, c).isZero());
    CHECK(ol_ss.compute_volume_hessian(a, b, c).isZero());
    CHECK(ol_ss.compute_activity_coefficients(a, b, c).isOnes());
  }
}

TEST_CASE_METHOD(OlivineFixture, "Single ss", "[core][solution_model]") {
  types::PairedEndmemberList fo_repeat = {{forsterite, "[Mg]2SiO4"},
                                          {forsterite, "[Mg]2SiO4"}};
  solution_models::IdealSolution fo_ss(fo_repeat);
  // Ensure different objects
  REQUIRE(&fo_ss.endmembers[0] != &fo_ss.endmembers[1]);
  // ints
  CHECK(fo_ss.get_n_endmembers() == 2);
  CHECK(fo_ss.get_n_occupancies() == 1);
  CHECK(fo_ss.get_n_sites() == 1);
  // strings
  CHECK(fo_ss.get_empty_formula() == "[]2SiO4");
  CHECK(fo_ss.get_general_formula() == "[Mg]2SiO4");
  CHECK(fo_ss.get_formulas() ==
        std::vector<std::string>{"[Mg]2SiO4", "[Mg]2SiO4"});
  CHECK(fo_ss.get_site_names() == std::vector<std::string>{"Mg_A"});
  CHECK(fo_ss.get_sites() == std::vector<std::vector<std::string>>{{"Mg"}});
  CHECK(
      fo_ss.get_solution_formulae() ==
      std::vector<std::map<std::string, double>>{{{"Mg", 2.0}}, {{"Mg", 2.0}}});
  CHECK(fo_ss.get_site_multiplicities().isApprox(
      (Eigen::ArrayXXd(2, 1) << 2.0, 2.0).finished(), tol_rel));
  CHECK(fo_ss.get_endmember_occupancies().isApprox(
      (Eigen::ArrayXXd(2, 1) << 1, 1).finished(), tol_rel));
  CHECK(fo_ss.get_endmember_n_occupancies().isApprox(
      (Eigen::ArrayXXd(2, 1) << 2, 2).finished(), tol_rel));
}

TEST_CASE_METHOD(MultiSiteFixture, "Check multi-site solutions",
                 "[core][solution_model]") {
  solution_models::IdealSolution sol(em);
  // ints
  CHECK(sol.get_n_endmembers() == 3);
  CHECK(sol.get_n_occupancies() == 5);
  CHECK(sol.get_n_sites() == 2);
  // strings
  CHECK(sol.get_empty_formula() == "[]3[]2Si3O12");
  CHECK(sol.get_general_formula() == "[Mg,Fe]3[Al,Mg,Si]2Si3O12");
  CHECK(sol.get_formulas() ==
        std::vector<std::string>{"[Mg]3[Al]2Si3O12", "[Fe]3[Al]2Si3O12",
                                 "[Mg]3[Mg1/2Si1/2]2Si3O12"});
  CHECK(sol.get_site_names() ==
        std::vector<std::string>{"Mg_A", "Fe_A", "Al_B", "Mg_B", "Si_B"});
  CHECK(sol.get_sites() == std::vector<std::vector<std::string>>{
                               {"Mg", "Fe"}, {"Al", "Mg", "Si"}});
  CHECK(sol.get_solution_formulae() ==
        std::vector<std::map<std::string, double>>{{{"Mg", 3.0}, {"Al", 2.0}},
                                                   {{"Fe", 3.0}, {"Al", 2.0}},
                                                   {{"Mg", 4.0}, {"Si", 1.0}}});
  CHECK(sol.get_site_multiplicities().isApprox(
      (Eigen::ArrayXXd(3, 5) << 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2)
          .finished(),
      tol_rel));
  CHECK(sol.get_endmember_occupancies().isApprox(
      (Eigen::ArrayXXd(3, 5) << 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0.5, 0.5)
          .finished(),
      tol_rel));
  CHECK(sol.get_endmember_n_occupancies().isApprox(
      (Eigen::ArrayXXd(3, 5) << 3, 0, 2, 0, 0, 0, 3, 2, 0, 0, 3, 0, 0, 1, 1)
          .finished(),
      tol_rel));
}

TEST_CASE_METHOD(MultiSiteFixture, "Regular Solutions",
                 "[core][solution_model]") {
  std::vector<double> alphas = {1.0, 2.0, 2.0};
  std::vector<double> a_ones = {1.0, 1.0, 1.0};
  std::vector<std::vector<double>> interactions = {{10.0e3, 5.0e3}, {-10.0e3}};
  Eigen::MatrixXd ref_W(3, 3);
  ref_W << 0, 10000, 5000, 0, 0, -10000, 0, 0, 0;
  Eigen::MatrixXd ref_Wa(3, 3);
  ref_Wa << 0, 10000 * (2.0 / 3.0), 5000 * (2.0 / 3.0), 0, 0, -5000, 0, 0, 0;
  SECTION("Symmetric Solution - alphas") {
    solution_models::SymmetricRegularSolution sym_sol(em, interactions);
    solution_models::AsymmetricRegularSolution asym_sol(em, a_ones,
                                                        interactions);
    REQUIRE(solution_models::RegularSolutionTestHelper::get_alphas(sym_sol)
                .size() == 3);
    REQUIRE((solution_models::RegularSolutionTestHelper::get_alphas(sym_sol) ==
             solution_models::RegularSolutionTestHelper::get_alphas(asym_sol))
                .all());
    REQUIRE(solution_models::RegularSolutionTestHelper::get_alphas(sym_sol)
                .isOnes());
  }
  SECTION("Symmetric Solution - interactions") {
    solution_models::SymmetricRegularSolution e_sol(em, interactions);
    solution_models::SymmetricRegularSolution evs_sol(
        em, interactions, interactions, interactions);
    REQUIRE(solution_models::RegularSolutionTestHelper::get_W_e(e_sol).rows() ==
            solution_models::RegularSolutionTestHelper::get_W_e(e_sol).cols());
    REQUIRE(solution_models::RegularSolutionTestHelper::get_W_v(e_sol).rows() ==
            solution_models::RegularSolutionTestHelper::get_W_v(e_sol).cols());
    REQUIRE(solution_models::RegularSolutionTestHelper::get_W_s(e_sol).rows() ==
            solution_models::RegularSolutionTestHelper::get_W_s(e_sol).cols());
    CHECK(solution_models::RegularSolutionTestHelper::get_W_v(e_sol).isZero());
    CHECK(solution_models::RegularSolutionTestHelper::get_W_s(e_sol).isZero());
    CHECK(solution_models::RegularSolutionTestHelper::get_W_e(e_sol).isApprox(
        ref_W, tol_rel));
    CHECK(solution_models::RegularSolutionTestHelper::get_W_e(evs_sol).isApprox(
        ref_W, tol_rel));
    CHECK(solution_models::RegularSolutionTestHelper::get_W_e(evs_sol).isApprox(
        solution_models::RegularSolutionTestHelper::get_W_v(evs_sol), tol_rel));
    CHECK(solution_models::RegularSolutionTestHelper::get_W_e(evs_sol).isApprox(
        solution_models::RegularSolutionTestHelper::get_W_s(evs_sol), tol_rel));
  }
  SECTION("Asymmetric Solution - interactions") {
    solution_models::AsymmetricRegularSolution sol(em, alphas, interactions);
    REQUIRE(
        solution_models::RegularSolutionTestHelper::get_alphas(sol).size() ==
        3);
    REQUIRE((solution_models::RegularSolutionTestHelper::get_alphas(sol) ==
             (Eigen::ArrayXd(3) << 1, 2, 2).finished())
                .all());
    CHECK(solution_models::RegularSolutionTestHelper::get_W_e(sol).isApprox(
        ref_Wa, tol_rel));
  }
}

TEST_CASE("Reference value tests", "[core][solution_model]") {
  Mineral min;
  types::PairedEndmemberList gr;
  gr = {{min, "[Mg]3[Al]2Si3O12"},
        {min, "[Fe]3[Al]2Si3O12"},
        {min, "[Ca]3[Al]2Si3O12"},
        {min, "[Ca]3[Fef]2Si3O12"},
        {min, "[Mg]3[Cr]2Si3O12"}};
  std::vector<double> alphas = {1.0, 1.0, 2.7, 0.4, 1.2};
  std::vector<std::vector<double>> energy_interaction = {
      {4.0e3, 35.0e3, 91.0e3, 2.0e3},
      {4.0e3, 60.0e3, 6.0e3},
      {2.0e3, 47.0e3},
      {101.0e3}};
  std::vector<std::vector<double>> volume_interaction = {
      {0.1e-5, 0.1e-5, 0.032e-5, 0.0},
      {0.1e-5, 0.032e-5, 0.01e-5},
      {0.0, 0.221e-5},
      {0.153e-5}};
  std::vector<std::vector<double>> entropy_interaction = {
      {0.0, 0.0, -1.7, 0.0}, {0.0, -1.7, 0.0}, {0.0, 33.8}, {32.1}};
  double P = 2.e9;
  double T = 800;
  Eigen::ArrayXd x(5);
  x << 0.23, 0.17, 0.08, 0.34, 0.18;

  SECTION("IdealSolution Model") {
    solution_models::IdealSolution ideal_sol(gr);
    double ref_ideal_excess_gibbs_free_energy = -34248.553969072134;
    double ref_ideal_excess_entropy = 42.81069246134017;
    Eigen::ArrayXd ref_ideal_excess_partial_gibbs_free_energies(5);
    ref_ideal_excess_partial_gibbs_free_energies << -27555.676990211556,
        -45122.98024626602, -27074.817329908074, -31662.278661905468,
        -40603.78604454302;
    Eigen::ArrayXd ref_ideal_excess_partial_entropies(5);
    ref_ideal_excess_partial_entropies << 34.44459623776444, 56.40372530783253,
        33.84352166238509, 39.57784832738184, 50.75473255567878;
    Eigen::ArrayXd ref_ideal_activities(5);
    ref_ideal_activities << 0.015879398399999994, 0.0011319551999999994,
        0.017069875199999994, 0.008564572799999996, 0.0022330403999999983;
    Eigen::ArrayXd ref_ideal_activity_coefficients(5);
    ref_ideal_activity_coefficients << 1.0, 1.0, 1.0, 1.0, 1.0;
    Eigen::MatrixXd ref_ideal_gibbs_hessian(5, 5);
    ref_ideal_gibbs_hessian << 43127.050003103795, -5542.9750787688245,
        -5542.9750787688245, -33257.85047261295, 15412.174609259666,
        -5542.9750787688245, 111837.67364810043, -5542.9750787688245,
        -33257.85047261295, -33257.85047261295, -5542.9750787688245,
        -5542.9750787688245, 41968.23988210683, 14253.3644882627,
        -33257.85047261295, -33257.85047261295, -33257.85047261295,
        14253.3644882627, 53380.247397219115, -33257.85047261295,
        15412.174609259666, -33257.85047261295, -33257.85047261295,
        -33257.85047261295, 89318.50899284401;
    Eigen::MatrixXd ref_ideal_entropy_hessian(5, 5);
    ref_ideal_entropy_hessian << -53.908812503879744, 6.92871884846103,
        6.92871884846103, 41.57231309076619, -19.26521826157458,
        6.92871884846103, -139.79709206012555, 6.92871884846103,
        41.57231309076619, 41.57231309076619, 6.92871884846103,
        6.92871884846103, -52.46029985263353, -17.816705610328373,
        41.57231309076619, 41.57231309076619, 41.57231309076619,
        -17.816705610328373, -66.72530924652389, 41.57231309076619,
        -19.26521826157458, 41.57231309076619, 41.57231309076619,
        41.57231309076619, -111.64813624105501;
    // Check sizes first
    REQUIRE(
        ideal_sol.compute_excess_partial_gibbs_free_energies(P, T, x).size() ==
        5);
    REQUIRE(ideal_sol.compute_excess_partial_entropies(P, T, x).size() == 5);
    REQUIRE(ideal_sol.compute_excess_partial_volumes(P, T, x).size() == 5);
    REQUIRE(ideal_sol.compute_activities(P, T, x).size() == 5);
    REQUIRE(ideal_sol.compute_activity_coefficients(P, T, x).size() == 5);
    REQUIRE(ideal_sol.compute_gibbs_hessian(P, T, x).rows() == 5);
    REQUIRE(ideal_sol.compute_entropy_hessian(P, T, x).rows() == 5);
    REQUIRE(ideal_sol.compute_volume_hessian(P, T, x).rows() == 5);
    REQUIRE(ideal_sol.compute_gibbs_hessian(P, T, x).cols() == 5);
    REQUIRE(ideal_sol.compute_entropy_hessian(P, T, x).cols() == 5);
    REQUIRE(ideal_sol.compute_volume_hessian(P, T, x).cols() == 5);
    CHECK(ideal_sol.compute_excess_volume(P, T, x) == 0);
    // H_ex = G_ex + T*S_ex cancels two finite values for an ideal solution.
    CHECK_THAT(
        ideal_sol.compute_excess_enthalpy(P, T, x),
        WithinAbs(0.0, roundoff_tolerance(ref_ideal_excess_gibbs_free_energy)));
    CHECK_THAT(ideal_sol.compute_excess_gibbs_free_energy(P, T, x),
               WithinRel(ref_ideal_excess_gibbs_free_energy, tol_rel) ||
                   WithinAbs(ref_ideal_excess_gibbs_free_energy, tol_abs));
    CHECK_THAT(ideal_sol.compute_excess_entropy(P, T, x),
               WithinRel(ref_ideal_excess_entropy, tol_rel) ||
                   WithinAbs(ref_ideal_excess_entropy, tol_abs));
    CHECK(
        ideal_sol.compute_excess_partial_gibbs_free_energies(P, T, x).isApprox(
            ref_ideal_excess_partial_gibbs_free_energies, tol_rel));
    CHECK(ideal_sol.compute_excess_partial_entropies(P, T, x).isApprox(
        ref_ideal_excess_partial_entropies, tol_rel));
    CHECK(ideal_sol.compute_excess_partial_volumes(P, T, x).isZero());
    CHECK(ideal_sol.compute_activities(P, T, x).isApprox(ref_ideal_activities,
                                                         tol_rel));
    CHECK(ideal_sol.compute_activity_coefficients(P, T, x).isOnes());
    CHECK(ideal_sol.compute_gibbs_hessian(P, T, x).isApprox(
        ref_ideal_gibbs_hessian, tol_rel));
    CHECK(ideal_sol.compute_entropy_hessian(P, T, x).isApprox(
        ref_ideal_entropy_hessian, tol_rel));
    CHECK(ideal_sol.compute_volume_hessian(P, T, x).isZero());
  }
  SECTION("SymmetricRegularSolution Model") {
    solution_models::SymmetricRegularSolution sym_sol(
        gr, energy_interaction, volume_interaction, entropy_interaction);
    double ref_sym_excess_gibbs_free_energy = -16920.505969072132;
    double ref_sym_excess_volume = 2.4314000000000005e-07;
    double ref_sym_excess_entropy = 45.03073246134017;
    double ref_sym_excess_enthalpy = 19104.08000000001;
    Eigen::ArrayXd ref_sym_excess_partial_gibbs_free_energies(5);
    ref_sym_excess_partial_gibbs_free_energies << -8923.724990211555,
        -38395.02824626602, -29804.465329908075, -2791.926661905469,
        -27818.234044543016;
    Eigen::ArrayXd ref_sym_excess_partial_entropies(5);
    ref_sym_excess_partial_entropies << 31.646556237764443, 53.60568530783253,
        37.70748166238509, 42.45580832738184, 62.15269255567878;
    Eigen::ArrayXd ref_sym_excess_partial_volumes(5);
    ref_sym_excess_partial_volumes << 1.1565999999999999e-07, 1.9366e-07,
        5.546600000000001e-07, 1.6025999999999998e-07, 4.7086000000000013e-07;
    Eigen::ArrayXd ref_sym_activities(5);
    ref_sym_activities << 0.2614278934913154, 0.0031125106783936127,
        0.011324162689802091, 0.6572180071334156, 0.01526480156715288;
    Eigen::ArrayXd ref_sym_activity_coefficients(5);
    ref_sym_activity_coefficients << 16.46333739515695, 2.749676558218571,
        0.6634004383231867, 76.73681133674476, 6.835882399240466;
    Eigen::MatrixXd ref_sym_gibbs_hessian(5, 5);
    ref_sym_gibbs_hessian << 5863.1460031038005, -24902.879078768827,
        15554.720921231172, 12239.84552738705, -14005.329390740331,
        -24902.879078768827, 98381.76964810042, -3541.2790787688264,
        -6856.154472612943, -46571.35447261295, 15554.720921231172,
        -3541.2790787688264, 47427.53588210682, -9887.339511737295,
        -18933.754472612945, 12239.84552738705, -6856.154472612943,
        -9887.339511737295, -4360.456602780869, 3466.2455273870582,
        -14005.329390740331, -46571.35447261295, -18933.754472612945,
        3466.2455273870582, 63747.40499284401;
    Eigen::MatrixXd ref_sym_entropy_hessian(5, 5);
    ref_sym_entropy_hessian << -48.312732503879744, 12.524798848461032,
        5.862798848461031, 39.79239309076619, -27.865138261574582,
        12.524798848461032, -134.20101206012555, 5.862798848461031,
        39.79239309076619, 32.97239309076619, 5.862798848461031,
        5.862798848461031, -60.18821985263353, -24.55862561032837,
        60.11039309076619, 39.79239309076619, 39.79239309076619,
        -24.55862561032837, -72.4812292465239, 59.39639309076619,
        -27.865138261574582, 32.97239309076619, 60.11039309076619,
        59.39639309076619, -134.44405624105502;
    Eigen::MatrixXd ref_sym_volume_hessian(5, 5);
    ref_sym_volume_hessian << -2.3131999999999996e-07, 6.906799999999999e-07,
        3.2967999999999987e-07, 4.408000000000009e-08, -5.8652e-07,
        6.906799999999999e-07, -3.873200000000001e-07, 2.5167999999999983e-07,
        -3.3919999999999976e-08, -5.645200000000001e-07, 3.2967999999999987e-07,
        2.5167999999999983e-07, -1.1093200000000002e-06, -7.1492e-07,
        1.1844800000000003e-06, 4.408000000000009e-08, -3.3919999999999976e-08,
        -7.1492e-07, -3.205199999999999e-07, 8.988799999999999e-07, -5.8652e-07,
        -5.645200000000001e-07, 1.1844800000000003e-06, 8.988799999999999e-07,
        -9.417200000000002e-07;
    CHECK_THAT(sym_sol.compute_excess_gibbs_free_energy(P, T, x),
               WithinRel(ref_sym_excess_gibbs_free_energy, tol_rel) ||
                   WithinAbs(ref_sym_excess_gibbs_free_energy, tol_abs));
    CHECK_THAT(sym_sol.compute_excess_volume(P, T, x),
               WithinRel(ref_sym_excess_volume, tol_rel) ||
                   WithinAbs(ref_sym_excess_volume, tol_abs));
    CHECK_THAT(sym_sol.compute_excess_entropy(P, T, x),
               WithinRel(ref_sym_excess_entropy, tol_rel) ||
                   WithinAbs(ref_sym_excess_entropy, tol_abs));
    CHECK_THAT(sym_sol.compute_excess_enthalpy(P, T, x),
               WithinRel(ref_sym_excess_enthalpy, tol_rel) ||
                   WithinAbs(ref_sym_excess_enthalpy, tol_abs));
    CHECK(sym_sol.compute_excess_partial_gibbs_free_energies(P, T, x).isApprox(
        ref_sym_excess_partial_gibbs_free_energies, tol_rel));
    CHECK(sym_sol.compute_excess_partial_entropies(P, T, x).isApprox(
        ref_sym_excess_partial_entropies, tol_rel));
    CHECK(sym_sol.compute_excess_partial_volumes(P, T, x).isApprox(
        ref_sym_excess_partial_volumes, tol_rel));
    CHECK(sym_sol.compute_activities(P, T, x).isApprox(ref_sym_activities,
                                                       tol_rel));
    CHECK(sym_sol.compute_activity_coefficients(P, T, x).isApprox(
        ref_sym_activity_coefficients, tol_rel));
    CHECK(sym_sol.compute_gibbs_hessian(P, T, x).isApprox(ref_sym_gibbs_hessian,
                                                          tol_rel));
    CHECK(sym_sol.compute_entropy_hessian(P, T, x).isApprox(
        ref_sym_entropy_hessian, tol_rel));
    CHECK(sym_sol.compute_volume_hessian(P, T, x).isApprox(
        ref_sym_volume_hessian, tol_rel));
  }
  SECTION("AsymmetricRegularSolution Model") {
    solution_models::AsymmetricRegularSolution asym_sol(
        gr, alphas, energy_interaction, volume_interaction,
        entropy_interaction);
    double ref_asym_excess_gibbs_free_energy = -22525.964050283328;
    double ref_asym_excess_volume = 2.3044199994778507e-07;
    double ref_asym_excess_entropy = 44.72732646370144;
    double ref_asym_excess_enthalpy = 13255.897120677822;
    Eigen::ArrayXd ref_asym_excess_partial_gibbs_free_energies(5);
    ref_asym_excess_partial_gibbs_free_energies << -15077.67204960215,
        -41382.15436514973, -37381.679205789085, -8797.400367311966,
        -33563.79319694547;
    Eigen::ArrayXd ref_asym_excess_partial_entropies(5);
    ref_asym_excess_partial_entropies << 32.123398183377056, 54.08252725344514,
        38.94051359794765, 41.9658389993695, 59.784938337875545;
    Eigen::ArrayXd ref_asym_excess_partial_volumes(5);
    ref_asym_excess_partial_volumes << 1.2240308407066145e-07,
        2.046720547693842e-07, 6.431305619057479e-07, 1.5103931525180075e-07,
        3.5939571757036417e-07;
    Eigen::ArrayXd ref_asym_activities(5);
    ref_asym_activities << 0.10364497647997141, 0.0019864394720921197,
        0.0036247199750732892, 0.26644029998475605, 0.006435050662822336;
    Eigen::ArrayXd ref_asym_activity_coefficients(5);
    ref_asym_activity_coefficients << 6.527009013135626, 1.7548746382296054,
        0.21234601498863276, 31.109584354838592, 2.881743950007506;
    Eigen::MatrixXd ref_asym_gibbs_hessian(5, 5);
    ref_asym_gibbs_hessian << 17346.048059695935, -16099.618489642478,
        26085.380847094428, -7135.147092427123, -5075.202037962081,
        -16099.618489642478, 104108.69455488495, 3716.7240468755845,
        -21824.624458634215, -38180.73016403933, 26085.380847094428,
        3716.7240468755845, 99465.19662772406, -43823.98908400308,
        1730.3326419031218, -7135.147092427123, -21824.624458634215,
        -43823.98908400308, 34483.65376532366, -15929.184245909644,
        -5075.202037962081, -38180.73016403933, 1730.3326419031218,
        -15929.184245909644, 71863.9812715278;
    Eigen::MatrixXd ref_asym_entropy_hessian(5, 5);
    ref_asym_entropy_hessian << -49.112948755145474, 11.724582597195297,
        8.137647372514113, 39.06101141549627, -25.71642499394225,
        11.724582597195297, -135.0012283113913, 8.137647372514113,
        39.06101141549627, 35.12110635839852, 8.137647372514113,
        8.137647372514113, -80.8939325510197, -26.583618408460318,
        68.08269952195798, 39.06101141549627, 39.06101141549627,
        -26.583618408460318, -68.69885525643106, 54.77719829813819,
        -25.71642499394225, 35.12110635839852, 68.08269952195798,
        54.77719829813819, -134.03707619691477;
    Eigen::MatrixXd ref_asym_volume_hessian(5, 5);
    ref_asym_volume_hessian << -2.528989340302922e-07, 6.951703111156555e-07,
        5.01901415870791e-07, -1.7710130188969437e-08, -5.230159281561548e-07,
        6.951703111156555e-07, -4.2287614621773584e-07, 2.724321794177417e-07,
        -5.170557262645827e-08, -5.123050353333843e-07, 5.01901415870791e-07,
        2.724321794177417e-07, -3.5877118122841317e-06, -6.870437767997532e-07,
        1.9936724052408257e-06, -1.7710130188969437e-08, -5.170557262645827e-08,
        -6.870437767997532e-07, -1.2482588037338906e-07, 6.125976597827414e-07,
        -5.230159281561548e-07, -5.123050353333843e-07, 1.9936724052408257e-06,
        6.125976597827414e-07, -8.910637625711509e-07;
    CHECK_THAT(asym_sol.compute_excess_gibbs_free_energy(P, T, x),
               WithinRel(ref_asym_excess_gibbs_free_energy, tol_rel) ||
                   WithinAbs(ref_asym_excess_gibbs_free_energy, tol_abs));
    CHECK_THAT(asym_sol.compute_excess_volume(P, T, x),
               WithinRel(ref_asym_excess_volume, tol_rel) ||
                   WithinAbs(ref_asym_excess_volume, tol_abs));
    CHECK_THAT(asym_sol.compute_excess_entropy(P, T, x),
               WithinRel(ref_asym_excess_entropy, tol_rel) ||
                   WithinAbs(ref_asym_excess_entropy, tol_abs));
    CHECK_THAT(asym_sol.compute_excess_enthalpy(P, T, x),
               WithinRel(ref_asym_excess_enthalpy, tol_rel) ||
                   WithinAbs(ref_asym_excess_enthalpy, tol_abs));
    CHECK(asym_sol.compute_excess_partial_gibbs_free_energies(P, T, x).isApprox(
        ref_asym_excess_partial_gibbs_free_energies, tol_rel));
    CHECK(asym_sol.compute_excess_partial_entropies(P, T, x).isApprox(
        ref_asym_excess_partial_entropies, tol_rel));
    CHECK(asym_sol.compute_excess_partial_volumes(P, T, x).isApprox(
        ref_asym_excess_partial_volumes, tol_rel));
    CHECK(asym_sol.compute_activities(P, T, x).isApprox(ref_asym_activities,
                                                        tol_rel));
    CHECK(asym_sol.compute_activity_coefficients(P, T, x).isApprox(
        ref_asym_activity_coefficients, tol_rel));
    CHECK(asym_sol.compute_gibbs_hessian(P, T, x).isApprox(
        ref_asym_gibbs_hessian, tol_rel));
    CHECK(asym_sol.compute_entropy_hessian(P, T, x).isApprox(
        ref_asym_entropy_hessian, tol_rel));
    CHECK(asym_sol.compute_volume_hessian(P, T, x).isApprox(
        ref_asym_volume_hessian, tol_rel));
  }
}
