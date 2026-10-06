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

// Numerical snapshots: Python BurnMan 39b582cd23954fabd3bdbbee6526a522b1d97bc9.
// Reproduce with tools/check_reference_data.py; see docs/reference_data.md.
#include "burnman/core/solution.hpp"
#include "burnman/core/solution_model.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include "solution_fixtures.hpp"
#include "tolerances.hpp"
#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace Catch::Matchers;
using namespace burnman;

TEST_CASE_METHOD(BridgmaniteFixture, "Test interface", "[core][solution]") {
  // Make solution
  Solution bdg;
  // Test name(s)
  REQUIRE_NOTHROW(bdg.get_name());
  REQUIRE(bdg.get_name() != "Test solution");
  REQUIRE_NOTHROW(bdg.set_name("Test solution"));
  REQUIRE(bdg.get_name() == "Test solution");

  // Check composition can't be set if no solution model
  REQUIRE_THROWS(bdg.set_composition(molar_fractions));
  // Set solution_model
  REQUIRE_NOTHROW(bdg.set_solution_model(bdg_solution_model));

  // Check get_names (and internal set_names from params) is working
  std::vector<std::string> expected_names = {
      "MgSiO3 perovskite", "FeSiO3 perovskite", "AlAlO3 perovskite"};
  REQUIRE(bdg.get_endmember_names() == expected_names);
  // Check endmember_formulae
  std::vector<types::FormulaMap> expected_formulae = {
      {{"Mg", 1.0}, {"Si", 1.0}, {"O", 3.0}},
      {{"Fe", 1.0}, {"Si", 1.0}, {"O", 3.0}},
      {{"Al", 2.0}, {"O", 3.0}}};
  REQUIRE(bdg.get_endmember_formulae() == expected_formulae);
  // Test element list
  std::vector<std::string> expected_elements = {"Mg", "Fe", "Al", "Si", "O"};
  REQUIRE(bdg.get_elements() == expected_elements);

  // Test setting method for all endmembers
  REQUIRE_NOTHROW(bdg.set_method(types::EOSType::Auto));

  // Test composition setting
  // Molar fractions of endmembers (Mg, Fe, Al)
  // Check bad molar fractions array
  REQUIRE_THROWS(bdg.set_composition((Eigen::ArrayXd(2) << 1, 1).finished()));
  REQUIRE_THROWS(
      bdg.set_composition((Eigen::ArrayXd(3) << 0.1, 0.1, 0.1).finished()));
  // Check set molar_fractions
  REQUIRE_NOTHROW(bdg.set_composition(molar_fractions));
  // Test formula
  types::FormulaMap expected_solution_formula = {
      {"Mg", 0.88}, {"Fe", 0.07}, {"Al", 0.1}, {"Si", 0.95}, {"O", 3.0}};
  REQUIRE(bdg.get_formula() == expected_solution_formula);
  // Test state
  REQUIRE_NOTHROW(bdg.set_state(P, T));
  REQUIRE_THAT(bdg.get_pressure(),
               WithinRel(P, tol_rel) || WithinAbs(P, tol_abs));
  REQUIRE_THAT(bdg.get_temperature(),
               WithinRel(T, tol_rel) || WithinAbs(T, tol_abs));
  // Test reset runs
  REQUIRE_NOTHROW(bdg.reset_cache());
}

TEST_CASE_METHOD(BridgmaniteFixture, "Test reference values",
                 "[core][solution]") {
  // Solution functions
  double ref_excess_gibbs = -10757.973872013106;
  double ref_excess_volume = 0.0;
  double ref_excess_entropy = 5.378986936006553;
  double ref_excess_enthalpy = 0.0;
  Eigen::ArrayXd ref_activities(3);
  ref_activities << 0.836, 0.0665, 0.0025000000000000005;
  Eigen::ArrayXd ref_activity_coefficients(3);
  ref_activity_coefficients << 1.0, 1.0, 1.0;
  Eigen::ArrayXd ref_excess_partial_gibbs(3);
  ref_excess_partial_gibbs << -2978.683935037304, -45073.58869554721,
      -99631.61600983949;
  Eigen::ArrayXd ref_excess_partial_volumes(3);
  ref_excess_partial_volumes << 0.0, 0.0, 0.0;
  Eigen::ArrayXd ref_excess_partial_entropies(3);
  ref_excess_partial_entropies << 1.4893419675186519, 22.536794347773604,
      49.815808004919745;
  Eigen::ArrayXd ref_partial_gibbs(3);
  ref_partial_gibbs << -740818.9553017678, -429520.98195517773,
      -988122.8420793302;
  Eigen::ArrayXd ref_partial_entropies(3);
  ref_partial_entropies << 250.5502490938234, 277.19661697059763,
      301.75523194572224;
  Eigen::MatrixXd ref_gibbs_hessian(3, 3);
  ref_gibbs_hessian << 3142.787305426345, -15753.718644921928,
      -33257.85047261296, -15753.718644921928, 221802.3561594563,
      -33257.85047261296, -33257.85047261296, -33257.85047261296,
      631899.1589796463;
  Eigen::MatrixXd ref_entropy_hessian(3, 3);
  ref_entropy_hessian << -1.5713936527131724, 7.876859322460964,
      16.62892523630648, 7.876859322460964, -110.90117807972815,
      16.62892523630648, 16.62892523630648, 16.62892523630648,
      -315.9495794898231;
  Eigen::MatrixXd ref_volume_hessian =
      Eigen::MatrixXd::Zero(3, 3); // DO ISZERO INSTEAD
  // Composite Material functions
  Eigen::Index ref_n_endmembers = 3;
  Eigen::Index ref_n_elements = 5;
  Eigen::Index ref_n_reactions = 0;
  std::vector<Eigen::Index> ref_independent_element_indices = {0, 1, 2};
  std::vector<Eigen::Index> ref_dependent_element_indices = {3, 4};
  Eigen::MatrixXd ref_stoichiometric_matrix(3, 5);
  ref_stoichiometric_matrix << 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 3.0,
      0.0, 0.0, 2.0, 0.0, 3.0;
  Eigen::MatrixXd ref_compositional_basis(3, 3);
  ref_compositional_basis << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::MatrixXd ref_compositional_null_basis(2, 5);
  ref_compositional_null_basis << -1.0, -1.0, 0.0, 1.0, 0.0, -3.0, -3.0, -1.5,
      0.0, 1.0;
  Eigen::MatrixXd ref_reaction_basis(0, 3);
  // ref_reaction_basis << []; // ? Check what to do here....
  //  Material functions
  double ref_molar_internal_energy = -1118021.6963208097;
  double ref_molar_gibbs = -731393.2915063847;
  double ref_molar_helmholtz = -1627973.1842963947;
  double ref_molar_mass = 0.102675125;
  double ref_molar_volume = 2.241449731975025e-05;
  double ref_density = 4580.746270384976;
  double ref_molar_entropy = 254.97574398779255;
  double ref_molar_enthalpy = -221441.8035307995;
  double ref_isothermal_bulk_modulus_reuss = 359295732518.51135;
  double ref_isentropic_bulk_modulus_reuss = 381900522278.24774;
  double ref_isothermal_compressibility_reuss = 2.7832225921260526e-12;
  double ref_isentropic_compressibility_reuss = 2.6184829338133582e-12;
  double ref_shear_modulus = 193760525424.63242;
  double ref_p_wave_velocity = 11822.408444942579;
  double ref_bulk_sound_velocity = 9130.761686548198;
  double ref_shear_wave_velocity = 6503.760400482546;
  double ref_grueneisen_parameter = 1.434732755790773;
  double ref_thermal_expansivity = 2.1925393874095368e-05;
  double ref_molar_heat_capacity_v = 123.07148838087139;
  double ref_molar_heat_capacity_p = 130.81442788301007;
  double ref_isentropic_thermal_gradient = 7.513646471242295e-09;

  // Make solution
  Solution bdg;
  bdg.set_solution_model(bdg_solution_model);
  bdg.set_method(types::EOSType::Auto);
  bdg.set_composition(molar_fractions);
  bdg.set_state(P, T);

  CHECK_THAT(bdg.get_excess_gibbs(), WithinRel(ref_excess_gibbs, tol_rel) ||
                                         WithinAbs(ref_excess_gibbs, tol_abs));
  CHECK_THAT(bdg.get_excess_volume(),
             WithinRel(ref_excess_volume, tol_rel) ||
                 WithinAbs(ref_excess_volume, tol_abs));
  CHECK_THAT(bdg.get_excess_entropy(),
             WithinRel(ref_excess_entropy, tol_rel) ||
                 WithinAbs(ref_excess_entropy, tol_abs));
  CHECK_THAT(
      bdg.get_excess_enthalpy(),
      WithinAbs(ref_excess_enthalpy, roundoff_tolerance(ref_excess_gibbs)));
  CHECK_THAT(bdg.get_molar_internal_energy(),
             WithinRel(ref_molar_internal_energy, tol_rel) ||
                 WithinAbs(ref_molar_internal_energy, tol_abs));
  CHECK_THAT(bdg.get_molar_gibbs(), WithinRel(ref_molar_gibbs, tol_rel) ||
                                        WithinAbs(ref_molar_gibbs, tol_abs));
  CHECK_THAT(bdg.get_molar_helmholtz(),
             WithinRel(ref_molar_helmholtz, tol_rel) ||
                 WithinAbs(ref_molar_helmholtz, tol_abs));
  CHECK_THAT(bdg.get_molar_mass(), WithinRel(ref_molar_mass, tol_rel) ||
                                       WithinAbs(ref_molar_mass, tol_abs));
  CHECK_THAT(bdg.get_molar_volume(), WithinRel(ref_molar_volume, tol_rel) ||
                                         WithinAbs(ref_molar_volume, tol_abs));
  CHECK_THAT(bdg.get_density(), WithinRel(ref_density, tol_rel) ||
                                    WithinAbs(ref_density, tol_abs));
  CHECK_THAT(bdg.get_molar_entropy(),
             WithinRel(ref_molar_entropy, tol_rel) ||
                 WithinAbs(ref_molar_entropy, tol_abs));
  CHECK_THAT(bdg.get_molar_enthalpy(),
             WithinRel(ref_molar_enthalpy, tol_rel) ||
                 WithinAbs(ref_molar_enthalpy, tol_abs));
  CHECK_THAT(bdg.get_isothermal_bulk_modulus_reuss(),
             WithinRel(ref_isothermal_bulk_modulus_reuss, tol_rel) ||
                 WithinAbs(ref_isothermal_bulk_modulus_reuss, tol_abs));
  CHECK_THAT(bdg.get_isentropic_bulk_modulus_reuss(),
             WithinRel(ref_isentropic_bulk_modulus_reuss, tol_rel) ||
                 WithinAbs(ref_isentropic_bulk_modulus_reuss, tol_abs));
  CHECK_THAT(bdg.get_isothermal_compressibility_reuss(),
             WithinRel(ref_isothermal_compressibility_reuss, tol_rel) ||
                 WithinAbs(ref_isothermal_compressibility_reuss, tol_abs));
  CHECK_THAT(bdg.get_isentropic_compressibility_reuss(),
             WithinRel(ref_isentropic_compressibility_reuss, tol_rel) ||
                 WithinAbs(ref_isentropic_compressibility_reuss, tol_abs));
  CHECK_THAT(bdg.get_shear_modulus(),
             WithinRel(ref_shear_modulus, tol_rel) ||
                 WithinAbs(ref_shear_modulus, tol_abs));
  CHECK_THAT(bdg.get_p_wave_velocity(),
             WithinRel(ref_p_wave_velocity, tol_rel) ||
                 WithinAbs(ref_p_wave_velocity, tol_abs));
  CHECK_THAT(bdg.get_bulk_sound_velocity(),
             WithinRel(ref_bulk_sound_velocity, tol_rel) ||
                 WithinAbs(ref_bulk_sound_velocity, tol_abs));
  CHECK_THAT(bdg.get_shear_wave_velocity(),
             WithinRel(ref_shear_wave_velocity, tol_rel) ||
                 WithinAbs(ref_shear_wave_velocity, tol_abs));
  CHECK_THAT(bdg.get_grueneisen_parameter(),
             WithinRel(ref_grueneisen_parameter, tol_rel) ||
                 WithinAbs(ref_grueneisen_parameter, tol_abs));
  CHECK_THAT(bdg.get_thermal_expansivity(),
             WithinRel(ref_thermal_expansivity, tol_rel) ||
                 WithinAbs(ref_thermal_expansivity, tol_abs));
  CHECK_THAT(bdg.get_molar_heat_capacity_v(),
             WithinRel(ref_molar_heat_capacity_v, tol_rel) ||
                 WithinAbs(ref_molar_heat_capacity_v, tol_abs));
  CHECK_THAT(bdg.get_molar_heat_capacity_p(),
             WithinRel(ref_molar_heat_capacity_p, tol_rel) ||
                 WithinAbs(ref_molar_heat_capacity_p, tol_abs));
  CHECK_THAT(bdg.get_isentropic_thermal_gradient(),
             WithinRel(ref_isentropic_thermal_gradient, tol_rel) ||
                 WithinAbs(ref_isentropic_thermal_gradient, tol_abs));
  CHECK(bdg.get_n_endmembers() == ref_n_endmembers);
  CHECK(bdg.get_n_elements() == ref_n_elements);
  CHECK(bdg.get_n_reactions() == ref_n_reactions);
  CHECK(bdg.get_independent_element_indices() ==
        ref_independent_element_indices);
  CHECK(bdg.get_dependent_element_indices() == ref_dependent_element_indices);
  CHECK(bdg.get_excess_partial_volumes().isZero());
  CHECK(bdg.get_volume_hessian().isZero());
  CHECK(bdg.get_activity_coefficients().isOnes());
  CHECK(bdg.get_compositional_basis().isIdentity());
  CHECK(bdg.get_activities().isApprox(ref_activities, tol_rel));
  CHECK(bdg.get_excess_partial_gibbs().isApprox(ref_excess_partial_gibbs,
                                                tol_rel));
  CHECK(bdg.get_excess_partial_entropies().isApprox(
      ref_excess_partial_entropies, tol_rel));
  CHECK(bdg.get_partial_gibbs().isApprox(ref_partial_gibbs, tol_rel));
  CHECK(bdg.get_partial_entropies().isApprox(ref_partial_entropies, tol_rel));
  CHECK(bdg.get_gibbs_hessian().isApprox(ref_gibbs_hessian, tol_rel));
  CHECK(bdg.get_entropy_hessian().isApprox(ref_entropy_hessian, tol_rel));
  CHECK(bdg.get_stoichiometric_matrix().isApprox(ref_stoichiometric_matrix,
                                                 tol_rel));
  CHECK(
      bdg.get_compositional_basis().isApprox(ref_compositional_basis, tol_rel));
  CHECK(bdg.get_compositional_null_basis().isApprox(
      ref_compositional_null_basis, tol_rel));
  Eigen::MatrixXd reaction_basis = bdg.get_reaction_basis();
  CHECK(reaction_basis.size() == 0);
  CHECK(reaction_basis.rows() == ref_reaction_basis.rows());
  CHECK(reaction_basis.cols() == ref_reaction_basis.cols());
}

TEST_CASE_METHOD(BridgmaniteFixture, "Test reactions", "[core][solution]") {
  // Make solution model & solution
  types::PairedEndmemberList em = {{mg_si_perovskite, "[Mg][Si]O3"},
                                   {fe_si_perovskite, "[Fe][Si]O3"},
                                   {mg_si_perovskite, "[Mg][Si]O3"},
                                   {fe_si_perovskite, "[Fe][Si]O3"}};
  std::shared_ptr<solution_models::IdealSolution> sol_model;
  sol_model = std::make_shared<solution_models::IdealSolution>(em);
  Solution sol;
  sol.set_solution_model(sol_model);
  REQUIRE(sol.get_n_endmembers() == 4);
  REQUIRE(sol.get_n_reactions() == 2);

  Eigen::MatrixXd expected_stoich_mat(4, 4);
  expected_stoich_mat << 1, 0, 1, 3, 0, 1, 1, 3, 1, 0, 1, 3, 0, 1, 1, 3;
  Eigen::MatrixXd expected_comp_basis(2, 4);
  expected_comp_basis << 1, 0, 0, 0, 0, 1, 0, 0;
  Eigen::MatrixXd expected_comp_null_basis(2, 4);
  expected_comp_null_basis << -1, -1, 1, 0, -3, -3, 0, 1;
  Eigen::MatrixXd expected_reac_basis(2, 4);
  expected_reac_basis << -1, 0, 1, 0, 0, -1, 0, 1;
  REQUIRE(
      (sol.get_stoichiometric_matrix().array() == expected_stoich_mat.array())
          .all());
  REQUIRE((sol.get_compositional_basis().array() == expected_comp_basis.array())
              .all());
  REQUIRE((sol.get_compositional_null_basis().array() ==
           expected_comp_null_basis.array())
              .all());
  REQUIRE(
      (sol.get_reaction_basis().array() == expected_reac_basis.array()).all());
}
