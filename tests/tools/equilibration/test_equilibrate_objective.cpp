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
#include "burnman/tools/equilibration/equilibrate_objective.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "burnman/utils/index_utils.hpp"
#include "equilibration_fixtures.hpp"
#include "tolerances.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace burnman;

TEST_CASE_METHOD(PartitioningAssemblageFixture, "equilibrate_objective",
                 "[tools][equilibration]") {
  std::vector<equilibration::FreeVectorMap> free;
  equilibration::ConstraintGroup constraints;
  constraints.push_back(
      equilibration::make_constraint<equilibration::PressureConstraint>(1.0e9));
  constraints.push_back(
      equilibration::make_constraint<equilibration::TemperatureConstraint>(
          2000.0));
  SECTION("Closed bulk composition") {}
  SECTION("Free bulk composition") {
    free.push_back({{"Mg", -1.0}, {"Fe", 1.0}, {"O", 0.0}});
    Eigen::VectorXd a = Eigen::VectorXd::Zero(7);
    a(6) = 1.0;
    constraints.push_back(
        equilibration::make_constraint<equilibration::LinearXConstraint>(a,
                                                                         0.1));
  }
  auto prm =
      equilibration::get_equilibration_parameters(assemblage, bulk, free);
  auto x = equilibration::get_parameter_vector(
      assemblage, burnman::utils::checked_int(free.size()));
  if (!free.empty())
    x(6) = 0.1;
  auto residual = [&](const Eigen::VectorXd &values) {
    return equilibration::F(values, assemblage, constraints,
                            prm.reduced_composition_vector,
                            prm.reduced_free_composition_vectors);
  };
  const auto f = residual(x);
  CHECK(f.size() == x.size());
  CHECK(f.head(constraints.size()).isZero(tol_abs));
  Eigen::VectorXd expected_bulk(2);
  // 0.4*(0.8 Mg,0.2 Fe) + 0.6*(0.2 Mg,0.8 Fe).
  expected_bulk << -0.06, 0.06;
  if (!free.empty())
    expected_bulk += Eigen::Vector2d(0.1, -0.1);
  CHECK(f.tail(2).isApprox(expected_bulk, tol_rel));
  const auto jacobian = equilibration::J(x, assemblage, constraints,
                                         prm.reduced_free_composition_vectors);
  REQUIRE(jacobian.rows() == x.size());
  REQUIRE(jacobian.cols() == x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const double step = i == 0 ? 1.0e5 : (i == 1 ? 0.01 : 1.0e-6);
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(x.size());
    delta(i) = step;
    const Eigen::VectorXd plus = residual(x + delta);
    const Eigen::VectorXd minus = residual(x - delta);
    const Eigen::VectorXd numerical = (plus - minus) / (2.0 * step);
    CAPTURE(i);
    CHECK((jacobian.col(i) - numerical).norm() <
          2.0e-6 * std::max(1.0, numerical.norm()));
  }
  equilibration::set_composition_and_state_from_parameters(assemblage, x);
}
