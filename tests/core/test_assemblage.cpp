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
#include "burnman/core/assemblage.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include "solution_fixtures.hpp"
#include "tolerances.hpp"
#include <Eigen/Dense>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace Catch::Matchers;
using namespace burnman;

struct MultiMineralFixture {
  CaPerovskiteFixture capv_fix;
  StishoviteFixture stish_fix;
};

TEST_CASE_METHOD(MultiMineralFixture, "Test assemblage creation",
                 "[core][assemblage]") {
  // Make assemblage
  Assemblage rock;

  // Test name(s)
  REQUIRE_NOTHROW(rock.get_name());
  REQUIRE(rock.get_name() != "Test assemblage");
  REQUIRE_NOTHROW(rock.set_name("Test assemblage"));
  REQUIRE(rock.get_name() == "Test assemblage");

  // Test set averaging scheme
  REQUIRE_NOTHROW(rock.set_averaging_scheme(types::AveragingType::VRH));

  // Can't set fractions without phases
  REQUIRE_THROWS(
      rock.set_fractions((Eigen::ArrayXd(2) << 0.6, 0.4).finished()));

  // Can add phases
  auto capv_ptr = std::make_shared<Mineral>(capv_fix.ca_perovskite);
  auto stish_ptr = std::make_shared<Mineral>(stish_fix.stishovite);
  REQUIRE_NOTHROW(rock.add_phases({capv_ptr, stish_ptr}));
  // Check get phase
  REQUIRE(rock.get_phase(0) == capv_ptr);
  REQUIRE(rock.get_phase(1) == stish_ptr);
  // And set molar fractions
  REQUIRE_NOTHROW(
      rock.set_fractions((Eigen::ArrayXd(2) << 0.6, 0.4).finished()));
  REQUIRE(rock.get_n_endmembers() == 2);

  // Check computed formula
  types::FormulaMap expected_formula = {{"Si", 1.0}, {"O", 2.6}, {"Ca", 0.6}};
  types::FormulaMap computed_formula = rock.get_formula();
  REQUIRE(computed_formula.size() == computed_formula.size());
  for (const auto &[key, val] : computed_formula) {
    REQUIRE(expected_formula.find(key) != expected_formula.end());
    REQUIRE_THAT(val, WithinAbs(expected_formula.at(key), tol_abs) ||
                          WithinRel(expected_formula.at(key), tol_rel));
  }

  // Check state can be set
  REQUIRE_NOTHROW(rock.set_state(50.e9, 2000.0));
  // Confirm P, T set
  REQUIRE(rock.get_pressure() == 50.e9);
  REQUIRE(rock.get_temperature() == 2000.0);
  REQUIRE(rock.get_phase(0)->get_pressure() == 50.e9);
  REQUIRE(rock.get_phase(0)->get_temperature() == 2000.0);
  REQUIRE(rock.get_phase(1)->get_pressure() == 50.e9);
  REQUIRE(rock.get_phase(1)->get_temperature() == 2000.0);

  // Check get_names (and internal set_names from params) is working
  std::vector<std::string> expected_names = {"Ca-perovskite", "Stishovite"};
  REQUIRE(rock.get_endmember_names() == expected_names);

  // Check endmember_formulae
  std::vector<types::FormulaMap> expected_formulae = {
      {{"Ca", 1.0}, {"Si", 1.0}, {"O", 3.0}}, {{"Si", 1.0}, {"O", 2.0}}};
  REQUIRE(rock.get_endmember_formulae() == expected_formulae);

  // Test element list
  std::vector<std::string> expected_elements = {"Ca", "Si", "O"};
  REQUIRE(rock.get_elements() == expected_elements);

  // Test setting method for all phases
  REQUIRE_NOTHROW(rock.set_method(types::EOSType::Auto));
}

TEST_CASE_METHOD(BdgFperAssemblageFixture, "Test multi-solution assemblage",
                 "[core][assemblage]") {
  REQUIRE(assemblage.get_name() == "Bdg + Fper assemblage");
  REQUIRE(assemblage.get_n_endmembers() == 5);
  REQUIRE(assemblage.get_n_elements() == 5);
  REQUIRE(assemblage.get_endmembers_per_phase() == std::vector<int>{3, 2});
  std::vector<std::string> expected_names = {
      "MgSiO3 perovskite in Bridgmanite", "FeSiO3 perovskite in Bridgmanite",
      "AlAlO3 perovskite in Bridgmanite", "Periclase in Ferro-periclase",
      "Wuestite in Ferro-periclase"};
  REQUIRE(assemblage.get_endmember_names() == expected_names);
  REQUIRE_NOTHROW(assemblage.set_state(50.e9, 2000.0));
  // Confirm P, T set
  REQUIRE(assemblage.get_pressure() == 50.e9);
  REQUIRE(assemblage.get_temperature() == 2000.0);
  REQUIRE(assemblage.get_phase(0)->get_pressure() == 50.e9);
  REQUIRE(assemblage.get_phase(0)->get_temperature() == 2000.0);
  REQUIRE(assemblage.get_phase(1)->get_pressure() == 50.e9);
  REQUIRE(assemblage.get_phase(1)->get_temperature() == 2000.0);
}

TEST_CASE_METHOD(PyroliteAssemblageFixture, "Test mixed phase type assemblage",
                 "[core][assemblage]") {
  REQUIRE(assemblage.get_name() == "Bdg + Fper + CaPv assemblage");
  REQUIRE(assemblage.get_n_endmembers() == 6);
  REQUIRE(assemblage.get_endmembers_per_phase() == std::vector<int>{3, 2, 1});
  REQUIRE(assemblage.get_n_elements() == 6);
  REQUIRE_NOTHROW(assemblage.set_state(50.e9, 2000.0));
  std::vector<types::FormulaMap> expected_formulae = {
      {{"Mg", 1.0}, {"Si", 1.0}, {"O", 3.0}},
      {{"Fe", 1.0}, {"Si", 1.0}, {"O", 3.0}},
      {{"Al", 2.0}, {"O", 3.0}},
      {{"Mg", 1.0}, {"O", 1.0}},
      {{"Fe", 1.0}, {"O", 1.0}},
      {{"Ca", 1.0}, {"Si", 1.0}, {"O", 3.0}}};
  REQUIRE(assemblage.get_endmember_formulae() == expected_formulae);
  types::FormulaMap expected_formula = {{"Mg", 0.796}, {"Fe", 0.069},
                                        {"Ca", 0.1},   {"Al", 0.07},
                                        {"Si", 0.765}, {"O", 2.6}};
  types::FormulaMap computed_formula = assemblage.get_formula();
  REQUIRE(computed_formula.size() == computed_formula.size());
  for (const auto &[key, val] : computed_formula) {
    REQUIRE(expected_formula.find(key) != expected_formula.end());
    REQUIRE_THAT(val, WithinAbs(expected_formula.at(key), tol_abs) ||
                          WithinRel(expected_formula.at(key), tol_rel));
  }
}

TEST_CASE_METHOD(NestedAssemblageFixture, "Test nested assemblage",
                 "[core][assemblage]") {
  REQUIRE(assemblage.get_name() == "Nested assemblage");
  REQUIRE(assemblage.get_n_endmembers() == 7);
  REQUIRE_NOTHROW(assemblage.set_state(50.e9, 2000.0));
  // Check access to nested assemblage phases
  REQUIRE(assemblage.get_pressure() == 50.e9);
  REQUIRE(assemblage.get_temperature() == 2000.0);
  REQUIRE(assemblage.get_phase(0)->get_pressure() == 50.e9);
  REQUIRE(assemblage.get_phase(1)->get_pressure() == 50.e9);
  // Check get_phase works with cast
  REQUIRE(assemblage.get_phase<Assemblage>(0)->get_phase(0)->get_pressure() ==
          50.e9);
}

TEST_CASE_METHOD(PyroliteAssemblageFixture, "Assemblage reference values",
                 "[core][assemblage]") {
  assemblage.set_state(50.e9, 2000.0);
  assemblage.set_method(types::EOSType::Auto);
  // Assemblage functions
  Eigen::Index ref_n_endmembers = 6;
  Eigen::Index ref_n_elements = 6;
  Eigen::Index ref_n_reactions = 1;
  std::vector<std::string> ref_elements = {"Ca", "Mg", "Fe", "Al", "Si", "O"};
  std::vector<Eigen::Index> ref_independent_element_indices = {0, 1, 2, 3, 4};
  std::vector<Eigen::Index> ref_dependent_element_indices = {5};
  Eigen::MatrixXd ref_stoichiometric_matrix(6, 6);
  ref_stoichiometric_matrix << 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 0.0,
      1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0;
  Eigen::MatrixXd ref_compositional_basis(5, 6);
  ref_compositional_basis << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::MatrixXd ref_compositional_null_basis(1, 6);
  ref_compositional_null_basis << -1.0, -1.0, -1.0, -1.5, -2.0, 1.0;
  Eigen::MatrixXd ref_reaction_basis(1, 6);
  ref_reaction_basis << 1.0, -1.0, 0.0, -1.0, 1.0, 0.0;
  // Material functions
  double ref_molar_internal_energy = -974191.8213846639;
  double ref_molar_gibbs = 999561205.8828137;
  double ref_molar_helmholtz = -1419441.7066281182;
  double ref_molar_mass = 0.09218043750000002;
  double ref_molar_volume = 0.020019612951788834;
  double ref_density = 4.604506476822936;
  double ref_molar_entropy = 222.62494262172714;
  double ref_molar_enthalpy = 1000006455.7680572;
  double ref_isothermal_bulk_modulus_reuss = 391190870683433.9;
  double ref_isentropic_bulk_modulus_reuss = 413160929347368.0;
  double ref_isothermal_compressibility_reuss = 2.556296874344077e-15;
  double ref_isentropic_compressibility_reuss = 2.420364388229079e-15;
  double ref_shear_modulus = 173600057792.58862;
  double ref_p_wave_velocity = 20308376.2550252;
  double ref_bulk_sound_velocity = 20307138.56026772;
  double ref_shear_wave_velocity = 194170.56244285888;
  double ref_grueneisen_parameter = 1424.8531625568785;
  double ref_thermal_expansivity = 1.9707992994516333e-08;
  double ref_molar_heat_capacity_v = 108.32200158712241;
  double ref_molar_heat_capacity_p = 114.40558100528757;

  CHECK(assemblage.get_n_endmembers() == ref_n_endmembers);
  CHECK(assemblage.get_n_elements() == ref_n_elements);
  CHECK(assemblage.get_n_reactions() == ref_n_reactions);
  CHECK(assemblage.get_elements() == ref_elements);
  CHECK(assemblage.get_independent_element_indices() ==
        ref_independent_element_indices);
  CHECK(assemblage.get_dependent_element_indices() ==
        ref_dependent_element_indices);
  CHECK(assemblage.get_stoichiometric_matrix().isApprox(
      ref_stoichiometric_matrix, tol_rel));
  CHECK(assemblage.get_compositional_basis().isApprox(ref_compositional_basis,
                                                      tol_rel));
  CHECK(assemblage.get_compositional_null_basis().isApprox(
      ref_compositional_null_basis, tol_rel));
  CHECK(assemblage.get_reaction_basis().isApprox(ref_reaction_basis));
  CHECK_THAT(assemblage.get_molar_internal_energy(),
             WithinRel(ref_molar_internal_energy, tol_rel) ||
                 WithinAbs(ref_molar_internal_energy, tol_abs));
  CHECK_THAT(assemblage.get_molar_gibbs(),
             WithinRel(ref_molar_gibbs, tol_rel) ||
                 WithinAbs(ref_molar_gibbs, tol_abs));
  CHECK_THAT(assemblage.get_molar_helmholtz(),
             WithinRel(ref_molar_helmholtz, tol_rel) ||
                 WithinAbs(ref_molar_helmholtz, tol_abs));
  CHECK_THAT(assemblage.get_molar_mass(),
             WithinRel(ref_molar_mass, tol_rel) ||
                 WithinAbs(ref_molar_mass, tol_abs));
  CHECK_THAT(assemblage.get_molar_volume(),
             WithinRel(ref_molar_volume, tol_rel) ||
                 WithinAbs(ref_molar_volume, tol_abs));
  CHECK_THAT(assemblage.get_density(), WithinRel(ref_density, tol_rel) ||
                                           WithinAbs(ref_density, tol_abs));
  CHECK_THAT(assemblage.get_molar_entropy(),
             WithinRel(ref_molar_entropy, tol_rel) ||
                 WithinAbs(ref_molar_entropy, tol_abs));
  CHECK_THAT(assemblage.get_molar_enthalpy(),
             WithinRel(ref_molar_enthalpy, tol_rel) ||
                 WithinAbs(ref_molar_enthalpy, tol_abs));
  CHECK_THAT(assemblage.get_isothermal_bulk_modulus_reuss(),
             WithinRel(ref_isothermal_bulk_modulus_reuss, tol_rel) ||
                 WithinAbs(ref_isothermal_bulk_modulus_reuss, tol_abs));
  CHECK_THAT(assemblage.get_isentropic_bulk_modulus_reuss(),
             WithinRel(ref_isentropic_bulk_modulus_reuss, tol_rel) ||
                 WithinAbs(ref_isentropic_bulk_modulus_reuss, tol_abs));
  CHECK_THAT(assemblage.get_isothermal_compressibility_reuss(),
             WithinRel(ref_isothermal_compressibility_reuss, tol_rel) ||
                 WithinAbs(ref_isothermal_compressibility_reuss, tol_abs));
  CHECK_THAT(assemblage.get_isentropic_compressibility_reuss(),
             WithinRel(ref_isentropic_compressibility_reuss, tol_rel) ||
                 WithinAbs(ref_isentropic_compressibility_reuss, tol_abs));
  CHECK_THAT(assemblage.get_shear_modulus(),
             WithinRel(ref_shear_modulus, tol_rel) ||
                 WithinAbs(ref_shear_modulus, tol_abs));
  CHECK_THAT(assemblage.get_p_wave_velocity(),
             WithinRel(ref_p_wave_velocity, tol_rel) ||
                 WithinAbs(ref_p_wave_velocity, tol_abs));
  CHECK_THAT(assemblage.get_bulk_sound_velocity(),
             WithinRel(ref_bulk_sound_velocity, tol_rel) ||
                 WithinAbs(ref_bulk_sound_velocity, tol_abs));
  CHECK_THAT(assemblage.get_shear_wave_velocity(),
             WithinRel(ref_shear_wave_velocity, tol_rel) ||
                 WithinAbs(ref_shear_wave_velocity, tol_abs));
  CHECK_THAT(assemblage.get_grueneisen_parameter(),
             WithinRel(ref_grueneisen_parameter, tol_rel) ||
                 WithinAbs(ref_grueneisen_parameter, tol_abs));
  CHECK_THAT(assemblage.get_thermal_expansivity(),
             WithinRel(ref_thermal_expansivity, tol_rel) ||
                 WithinAbs(ref_thermal_expansivity, tol_abs));
  CHECK_THAT(assemblage.get_molar_heat_capacity_v(),
             WithinRel(ref_molar_heat_capacity_v, tol_rel) ||
                 WithinAbs(ref_molar_heat_capacity_v, tol_abs));
  CHECK_THAT(assemblage.get_molar_heat_capacity_p(),
             WithinRel(ref_molar_heat_capacity_p, tol_rel) ||
                 WithinAbs(ref_molar_heat_capacity_p, tol_abs));
}

TEST_CASE_METHOD(CaPvStishAssemblageFixture,
                 "Assemblage thermodynamics follow Gibbs derivatives",
                 "[core][assemblage]") {
  const double pressure = 40.0e9, temperature = 2000.0;
  assemblage.set_method(types::EOSType::Auto);
  assemblage.set_state(pressure, temperature);
  // Thermodynamic derivatives are additive; they need no seismic averaging.
  assemblage.set_averaging_scheme(
      std::shared_ptr<averaging::AveragingScheme>{});
  const double volume = assemblage.get_molar_volume();
  const double kt = assemblage.get_isothermal_bulk_modulus_reuss();
  const double ks = assemblage.get_isentropic_bulk_modulus_reuss();
  const double alpha = assemblage.get_thermal_expansivity();
  const double cp = assemblage.get_molar_heat_capacity_p();
  const double cv = assemblage.get_molar_heat_capacity_v();
  const double dP = 1.0e7, dT = 0.1;
  assemblage.set_state(pressure + dP, temperature);
  const double v_plus_p = assemblage.get_molar_volume();
  assemblage.set_state(pressure - dP, temperature);
  const double v_minus_p = assemblage.get_molar_volume();
  CHECK_THAT(kt,
             WithinRel(-volume * 2.0 * dP / (v_plus_p - v_minus_p), 2.0e-6));
  assemblage.set_state(pressure, temperature + dT);
  const double v_plus_t = assemblage.get_molar_volume();
  const double s_plus_t = assemblage.get_molar_entropy();
  assemblage.set_state(pressure, temperature - dT);
  const double v_minus_t = assemblage.get_molar_volume();
  const double s_minus_t = assemblage.get_molar_entropy();
  CHECK_THAT(alpha,
             WithinRel((v_plus_t - v_minus_t) / (2.0 * dT * volume), 2.0e-6));
  CHECK_THAT(
      cp, WithinRel(temperature * (s_plus_t - s_minus_t) / (2.0 * dT), 2.0e-6));
  // To first order, dP = alpha*K_T*dT keeps total volume fixed.
  assemblage.set_state(pressure + alpha * kt * dT, temperature + dT);
  const double s_plus_v = assemblage.get_molar_entropy();
  assemblage.set_state(pressure - alpha * kt * dT, temperature - dT);
  const double s_minus_v = assemblage.get_molar_entropy();
  CHECK_THAT(
      cv, WithinRel(temperature * (s_plus_v - s_minus_v) / (2.0 * dT), 2.0e-6));
  CHECK_THAT(ks, WithinRel(kt * cp / cv, tol_rel));
  for (auto scheme : {types::AveragingType::Reuss, types::AveragingType::Voigt,
                      types::AveragingType::VRH}) {
    assemblage.set_state(pressure, temperature);
    assemblage.set_averaging_scheme(scheme);
    CHECK_THAT(assemblage.get_isothermal_bulk_modulus_reuss(),
               WithinRel(kt, tol_rel));
    CHECK_THAT(assemblage.get_isentropic_bulk_modulus_reuss(),
               WithinRel(ks, tol_rel));
    CHECK_THAT(assemblage.get_molar_heat_capacity_v(), WithinRel(cv, tol_rel));
  }
}

TEST_CASE_METHOD(CaPvStishAssemblageFixture,
                 "Assemblage bulk modulus has a finite zero temperature limit",
                 "[core][assemblage]") {
  assemblage.set_method(types::EOSType::Auto);
  assemblage.set_state(40.0e9, 0.0);
  CHECK_THAT(
      assemblage.get_isentropic_bulk_modulus_reuss(),
      WithinRel(assemblage.get_isothermal_bulk_modulus_reuss(), tol_rel));
  CHECK(assemblage.get_molar_heat_capacity_v() == 0.0);
}

// TODO:
// reset_cache() (check over)
// add_phase (value)
// add_phases (value - variadic template)
// add_phase (pointer)
// add_phases (vector of ptrs)
// add_phases(init list of ptrs)
// set_fractions using volume fractions
// set_averaging_scheme(custom scheme with unique ptr)
// get_isentropic_thermal_gradient
