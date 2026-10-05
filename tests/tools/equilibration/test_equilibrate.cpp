/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
#include "burnman/tools/equilibration/equality_constraints.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "equilibration_fixtures.hpp"
#include "tolerances.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace burnman;

TEST_CASE_METHOD(PartitioningAssemblageFixture, "equilibrate implementation",
                 "[tools][equilibration]") {
  Eigen::ArrayXd pressures(2), temperatures(2);
  pressures << 1.0e9, 2.0e9;
  temperatures << 1500.0, 2000.0;
  auto constraints = equilibration::make_constraint_list(
      equilibration::make_constraints_from_array<
          equilibration::PressureConstraint>(pressures),
      equilibration::make_constraints_from_array<
          equilibration::TemperatureConstraint>(temperatures));
  auto result = equilibration::equilibrate(bulk, assemblage, constraints, {},
                                           1.0e-8, true, false);
  REQUIRE(result.sol_array.shape() == std::vector<std::size_t>{2, 2});
  REQUIRE(result.prm.n_parameters == 6);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      const auto &solve = result.sol_array({i, j});
      INFO(solve.message);
      REQUIRE(solve.success);
      CHECK(solve.code == 0);
      REQUIRE(solve.iteration_history.has_value());
      CHECK_FALSE(solve.iteration_history->x.empty());
      CHECK(solve.F_norm < 1.0e-7);
      CHECK_THAT(solve.x(0),
                 WithinAbs(pressures(static_cast<Eigen::Index>(i)), 1.0));
      CHECK_THAT(solve.x(1),
                 WithinAbs(temperatures(static_cast<Eigen::Index>(j)), 1.0e-8));
      const double mg =
          equilibrium_mg_fraction(temperatures(static_cast<Eigen::Index>(j)));
      CHECK_THAT(solve.x(2), WithinAbs(0.5, 1.0e-9));
      CHECK_THAT(solve.x(3), WithinAbs(1.0 - mg, 1.0e-9));
      CHECK_THAT(solve.x(4), WithinAbs(0.5, 1.0e-9));
      CHECK_THAT(solve.x(5), WithinAbs(mg, 1.0e-9));
      equilibration::set_composition_and_state_from_parameters(assemblage,
                                                               solve.x);
      const auto amounts = equilibration::get_endmember_amounts(assemblage);
      CHECK((assemblage.get_stoichiometric_matrix().transpose() * amounts -
             result.prm.bulk_composition_vector)
                .norm() < 1.0e-9);
      CHECK(assemblage.get_reaction_affinities().norm() < 1.0e-7);
    }
  }
}

TEST_CASE_METHOD(PartitioningAssemblageFixture,
                 "equilibrate free composition and prescribed phase fraction",
                 "[tools][equilibration]") {
  std::vector<equilibration::FreeVectorMap> free{
      {{"Mg", -1.0}, {"Fe", 1.0}, {"O", 0.0}}};
  auto prm =
      equilibration::get_equilibration_parameters(assemblage, bulk, free);
  auto constraints = equilibration::make_constraint_list(
      equilibration::make_constraint<equilibration::PressureConstraint>(1.0e9),
      equilibration::make_constraint<equilibration::TemperatureConstraint>(
          2000.0),
      equilibration::make_constraint<equilibration::PhaseFractionConstraint>(
          0, 0.4, prm));
  auto result = equilibration::equilibrate(bulk, assemblage, constraints, free,
                                           1.0e-8, false, false);
  const auto &solve = result.sol_array(0);
  INFO(solve.message);
  REQUIRE(solve.success);
  CHECK_THAT(assemblage.get_molar_fractions()(0), WithinAbs(0.4, 1.0e-9));
  const double mg = equilibrium_mg_fraction(2000.0);
  const double bulk_mg = 0.4 * mg + 0.6 * (1.0 - mg);
  CHECK_THAT(solve.x(6), WithinAbs(0.5 - bulk_mg, 1.0e-9));
  CHECK(solve.F_norm < 1.0e-7);
}

TEST_CASE_METHOD(PartitioningAssemblageFixture,
                 "equilibrate rejects incorrect constraint counts",
                 "[tools][equilibration]") {
  auto constraints = equilibration::make_constraint_list(
      equilibration::make_constraint<equilibration::PressureConstraint>(1.0e9));
  CHECK_THROWS(equilibration::equilibrate(bulk, assemblage, constraints));
}
