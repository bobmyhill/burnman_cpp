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

#include "burnman/core/combined_mineral.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/equilibration/equality_constraints.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace burnman;
using namespace burnman::equilibration;
using namespace Catch::Matchers;

TEST_CASE("Native datasets and combined endmembers need no Python",
          "[native_datasets]") {
  auto sill = minerals::HP_2011_ds62::sill();
  auto andal = minerals::HP_2011_ds62::andalusite();
  sill->set_state(3.e9, 1200.);
  andal->set_state(3.e9, 1200.);
  Eigen::ArrayXd amounts(2);
  amounts << .5, .5;
  auto combined = make_combined_mineral({*sill, *andal}, amounts,
                                        Eigen::Vector3d(2500., 2., 1.e-7));
  combined.set_state(3.e9, 1200.);
  double expected = .5 * (sill->get_molar_gibbs() + andal->get_molar_gibbs()) +
                    2500. - 1200. * 2. + 3.e9 * 1.e-7;
  CHECK_THAT(combined.get_molar_gibbs(), WithinRel(expected, 1.e-12));
  CHECK(combined.get_formula() == sill->get_formula());
}

TEST_CASE("Native reduced solution systems", "[native_datasets]") {
  auto opx = minerals::JH_2015::mg_fe_orthopyroxene();
  auto gt = minerals::SLB_2011::pyrope_grossular();
  auto bdg = minerals::SLB_2011::mg_fe_bridgmanite_binary();
  CHECK(opx->get_n_endmembers() == 3);
  CHECK(gt->get_n_endmembers() == 2);
  CHECK(bdg->get_n_endmembers() == 2);
  Eigen::ArrayXd fractions(3);
  fractions << .55, -.1, .55;
  opx->set_composition(fractions);
  CHECK((opx->get_endmember_occupancies().matrix().transpose() *
         fractions.matrix())
            .minCoeff() >= 0.);
  opx->set_state(1.e5, 800.);
  CHECK(std::isfinite(opx->get_molar_gibbs()));
}

TEST_CASE("Aluminosilicate invariant from native factories",
          "[native_datasets]") {
  auto sill = minerals::HP_2011_ds62::sill();
  auto andal = minerals::HP_2011_ds62::andalusite();
  auto ky = minerals::HP_2011_ds62::ky();
  Assemblage a;
  a.add_phases({sill, andal, ky});
  a.set_fractions(Eigen::ArrayXd::Constant(3, 1. / 3.));
  a.set_averaging_scheme(types::AveragingType::VRH);
  const auto composition = sill->get_formula();
  auto prm = get_equilibration_parameters(a, composition, {});
  ConstraintList constraints(2);
  constraints[0].push_back(
      std::make_unique<PhaseFractionConstraint>(2, 0., prm));
  constraints[1].push_back(
      std::make_unique<PhaseFractionConstraint>(0, 0., prm));
  Eigen::VectorXd tolerances = Eigen::VectorXd::Constant(5, 1.e-9);
  tolerances.head(2) << 1., 1.e-6;
  auto result = equilibrate(composition, a, constraints, {}, 1.e-3, false,
                            false, 100, false, tolerances);
  REQUIRE(result.sol_array(0).success);
  CHECK_THAT(result.sol_array(0).x(0) / 1.e9, WithinAbs(.4307, .0001));
  CHECK_THAT(result.sol_array(0).x(1), WithinAbs(809.34, .01));
}

TEST_CASE("Scaled equilibrium keeps physical results and iteration history",
          "[native_datasets][equilibration_scaling]") {
  auto sill = minerals::HP_2011_ds62::sill();
  Assemblage a;
  a.add_phases({sill, minerals::HP_2011_ds62::andalusite(),
                minerals::HP_2011_ds62::ky()});
  a.set_fractions(Eigen::ArrayXd::Constant(3, 1. / 3.));
  a.set_state(8.e8, 900.);
  const auto bulk = sill->get_formula();
  auto prm = get_equilibration_parameters(a, bulk, {});
  ConstraintList constraints(2);
  constraints[0].push_back(
      std::make_unique<PhaseFractionConstraint>(2, 0., prm));
  constraints[1].push_back(
      std::make_unique<PhaseFractionConstraint>(0, 0., prm));
  Eigen::VectorXd tolerances = Eigen::VectorXd::Constant(5, 1.e-10);
  tolerances.head(2) << 1., 1.e-6;
  Eigen::VectorXd scales = Eigen::VectorXd::Ones(5);
  scales.head(2) << 1.e9, 1000.;
  auto result = equilibrate(bulk, a, constraints, {}, 1.e-9, true, false, 100,
                            false, tolerances, scales);
  const auto &s = result.sol_array(0);
  REQUIRE(s.success);
  CHECK_THAT(s.x[0] / 1.e9, WithinAbs(.4307, .0001));
  CHECK_THAT(s.x[1], WithinAbs(809.34, .01));
  CHECK_THAT(s.x[3], WithinAbs(1., 1.e-10));
  ConstraintGroup conditions;
  for (const auto &group : constraints)
    conditions.push_back(group[0]->clone());
  auto residual = F(s.x, a, conditions, prm.reduced_composition_vector,
                    prm.reduced_free_composition_vectors);
  auto jacobian = J(s.x, a, conditions, prm.reduced_free_composition_vectors);
  CHECK((s.F - residual).norm() < 1.e-10);
  CHECK((s.J - jacobian).norm() < 1.e-10);
  REQUIRE(s.iteration_history);
  REQUIRE(s.iteration_history->x.size() == s.iteration_history->F.size());
  for (std::size_t i = 0; i < s.iteration_history->x.size(); ++i) {
    const auto &x = s.iteration_history->x[i];
    CHECK(x[0] > 1.e8);
    auto f = F(x, a, conditions, prm.reduced_composition_vector,
               prm.reduced_free_composition_vectors);
    CHECK((f - s.iteration_history->F[i]).norm() < 1.e-7);
  }
  scales[0] = -1.;
  CHECK_THROWS_AS(equilibrate(bulk, a, constraints, {}, 1.e-9, false, false,
                              100, false, tolerances, scales),
                  std::invalid_argument);
}
