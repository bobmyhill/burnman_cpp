// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/relaxed_composite.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "equilibration_fixtures.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE_METHOD(
    PartitioningAssemblageFixture,
    "RelaxedComposite equilibrates and differentiates partitioning",
    "[core][relaxed_composite]") {
  auto ordinary = std::make_shared<Assemblage>(assemblage);
  ordinary->set_averaging_scheme(types::AveragingType::VRH);
  RelaxedComposite relaxed(ordinary, ordinary->get_reaction_basis());
  CHECK_FALSE(relaxed.has_state());
  CHECK(relaxed.get_phase(0) == ordinary->get_phase(0));
  const double scale = GENERATE(1., 5.), t = 2000., dt = .05;
  for (auto &entry : bulk)
    entry.second *= scale;
  relaxed.set_state(1.e9, t, bulk);
  CHECK_THAT(relaxed.get_n_moles(), WithinRel(scale, 1.e-12));
  CHECK(ordinary->get_reaction_affinities().norm() < 1.e-6);
  const double mg = equilibrium_mg_fraction(t);
  CHECK_THAT(ordinary->get_phase<Solution>(0)->get_molar_fractions()[0],
             WithinRel(mg, 1.e-10));
  const double extra_cp =
      1.e8 / (constants::physics::gas_constant * t * t) * mg * (1. - mg);
  CHECK_THAT(relaxed.get_molar_heat_capacity_p() -
                 ordinary->get_molar_heat_capacity_p(),
             WithinRel(extra_cp, 1.e-9));
  const Eigen::VectorXd derivative =
      relaxed.get_dndq() * relaxed.get_dqdz_relaxed().col(1);
  relaxed.set_state(1.e9, t + dt, bulk);
  const auto plus = equilibration::get_endmember_amounts(*ordinary);
  const double entropy_plus = relaxed.get_molar_entropy();
  relaxed.set_state(1.e9, t - dt, bulk);
  const auto minus = equilibration::get_endmember_amounts(*ordinary);
  const double entropy_minus = relaxed.get_molar_entropy();
  CHECK(((plus - minus) / (2. * dt) - derivative).norm() < 1.e-10 * scale);
  relaxed.set_state(1.e9, t, bulk);
  CHECK_THAT(relaxed.get_molar_heat_capacity_p(),
             WithinRel(t * (entropy_plus - entropy_minus) / (2. * dt), 2.e-8));
}

TEST_CASE_METHOD(PartitioningAssemblageFixture,
                 "RelaxedComposite copies state, resets derivatives and clones "
                 "independently",
                 "[core][relaxed_composite]") {
  auto ordinary = std::make_shared<Assemblage>(assemblage);
  ordinary->set_averaging_scheme(types::AveragingType::Reuss);
  RelaxedComposite relaxed(ordinary, ordinary->get_reaction_basis());
  relaxed.set_state(1.e9, 2000., bulk);
  const double cp = relaxed.get_molar_heat_capacity_p();
  ordinary->set_state(2.e9, 1800.);
  relaxed.copy_state_from_unrelaxed();
  CHECK(relaxed.get_temperature() == 1800.);
  CHECK(relaxed.get_pressure() == 2.e9);
  CHECK(relaxed.get_molar_heat_capacity_p() != cp);
  auto clone = std::dynamic_pointer_cast<RelaxedComposite>(relaxed.copy());
  REQUIRE(clone);
  CHECK(clone->get_phase(0) != relaxed.get_phase(0));
  CHECK(clone->get_phase(0) == clone->get_unrelaxed()->get_phase(0));
  CHECK_THAT(clone->get_molar_heat_capacity_p(),
             WithinRel(relaxed.get_molar_heat_capacity_p(), 1.e-12));
  clone->set_state(3.e9, 2200., bulk);
  CHECK(relaxed.get_temperature() == 1800.);
  relaxed.clear_state();
  CHECK_FALSE(relaxed.has_state());
}

TEST_CASE_METHOD(
    PartitioningAssemblageFixture,
    "RelaxedComposite validates reaction vectors and zero responses",
    "[core][relaxed_composite]") {
  auto ordinary = std::make_shared<Assemblage>(assemblage);
  ordinary->set_averaging_scheme(types::AveragingType::VRH);
  CHECK_THROWS_AS(RelaxedComposite(nullptr, Eigen::MatrixXd(0, 4)),
                  std::invalid_argument);
  CHECK_THROWS_AS(RelaxedComposite(ordinary, Eigen::MatrixXd::Zero(1, 3)),
                  std::invalid_argument);
  CHECK_THROWS_AS(RelaxedComposite(ordinary, Eigen::MatrixXd::Ones(1, 4)),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      RelaxedComposite(ordinary,
                       Eigen::MatrixXd::Constant(
                           1, 4, std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
  RelaxedComposite validated(ordinary, ordinary->get_reaction_basis());
  auto invalid_bulk = bulk;
  invalid_bulk["O"] += 1.;
  CHECK_THROWS_AS(validated.set_state(1.e9, 2000., invalid_bulk),
                  std::invalid_argument);
  invalid_bulk = bulk;
  invalid_bulk["Ca"] = 1.;
  CHECK_THROWS_AS(validated.set_state(1.e9, 2000., invalid_bulk),
                  std::invalid_argument);
  CHECK_FALSE(validated.has_state());
  const int n_vectors = GENERATE(0, 1);
  RelaxedComposite frozen(ordinary, Eigen::MatrixXd::Zero(n_vectors, 4));
  frozen.set_state(1.e9, 2000., bulk);
  CHECK(frozen.get_dqdz_relaxed().rows() == n_vectors);
  CHECK(frozen.get_dqdz_relaxed().norm() == 0.);
  CHECK_THAT(frozen.get_molar_heat_capacity_p(),
             WithinRel(ordinary->get_molar_heat_capacity_p(), 1.e-12));
  ordinary->set_fractions({1., 0.});
  frozen.copy_state_from_unrelaxed();
  CHECK_THROWS_AS(frozen.get_molar_heat_capacity_p(), std::domain_error);
}
