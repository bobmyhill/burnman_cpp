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

#include "burnman/optim/roots/damped_newton_solver.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman::optim::roots;
using namespace Catch::Matchers;

TEST_CASE("Newton stagnation is an optional failure", "[optim][newton]") {
  // A finite Newton correction cannot change this large floating-point x.
  const Eigen::VectorXd start = Eigen::VectorXd::Constant(1, 1.e20);
  auto residual = [](const Eigen::VectorXd &x) {
    return Eigen::VectorXd::Constant(1, std::sin(x[0])).eval();
  };
  auto jacobian = [](const Eigen::VectorXd &x) {
    return Eigen::MatrixXd::Constant(1, 1, std::cos(x[0])).eval();
  };
  const LinearConstraints unconstrained{Eigen::MatrixXd(0, 1),
                                        Eigen::VectorXd(0)};
  DampedNewtonSettings settings;
  settings.max_iterations = 40;
  const auto baseline = DampedNewtonSolver(settings).solve(
      start, residual, jacobian, unconstrained);
  REQUIRE_FALSE(baseline.success);
  CHECK(baseline.code == 3);

  settings.stagnation_iterations = 5;
  settings.store_iterates = true;
  const auto stalled = DampedNewtonSolver(settings).solve(
      start, residual, jacobian, unconstrained);
  REQUIRE_FALSE(stalled.success);
  CHECK(stalled.code == 4);
  CHECK(stalled.n_iterations < baseline.n_iterations);
  CHECK(stalled.x[0] == start[0]);
  CHECK(stalled.F_norm > 0.1);
  REQUIRE(stalled.iteration_history);
  CHECK(stalled.iteration_history->x.size() ==
        static_cast<std::size_t>(stalled.n_iterations + 1));
  CHECK((stalled.F - residual(stalled.x)).norm() == 0.);
  CHECK((stalled.J - jacobian(stalled.x)).norm() == 0.);

  settings.stagnation_iterations = -1;
  CHECK_THROWS_AS(DampedNewtonSolver(settings).solve(start, residual, jacobian,
                                                     unconstrained),
                  std::invalid_argument);
}

TEST_CASE("Newton stagnation preserves slowly converging roots",
          "[optim][newton]") {
  DampedNewtonSettings settings;
  settings.stagnation_iterations = 3;
  settings.tol = 1.e-3;
  settings.max_iterations = 100;
  settings.store_iterates = true;
  settings.lambda_bounds_func = [](const auto &, const auto &) {
    return LambdaBounds{0.1, 0.1};
  };
  const auto root = DampedNewtonSolver(settings).solve(
      Eigen::VectorXd::Zero(1),
      [](const Eigen::VectorXd &x) { return (x.array() - 1.).matrix().eval(); },
      [](const auto &) { return Eigen::MatrixXd::Identity(1, 1).eval(); },
      {Eigen::MatrixXd(0, 1), Eigen::VectorXd(0)});
  INFO(root.message);
  REQUIRE(root.success);
  CHECK(root.code == 0);
  CHECK_THAT(root.x[0], WithinAbs(1., 1.e-12));
  CHECK(root.F_norm < 1.e-12);
  // The last several damped movements are small, but still improve the root.
  const auto &history = root.iteration_history->x;
  REQUIRE(history.size() > 4);
  for (std::size_t i = history.size() - 3; i < history.size(); ++i)
    CHECK(std::abs(history[i][0] - history[i - 1][0]) < settings.tol);
}

TEST_CASE("Newton stagnation respects parameter and equation units",
          "[optim][newton]") {
  for (bool rescale : {false, true}) {
    CAPTURE(rescale);
    const Eigen::Vector2d units =
        rescale ? Eigen::Vector2d(1.e9, 1.e-3) : Eigen::Vector2d::Ones();
    const Eigen::Vector2d weights =
        rescale ? Eigen::Vector2d(1.e-12, 1.e12) : Eigen::Vector2d::Ones();
    DampedNewtonSettings settings;
    settings.stagnation_iterations = 5;
    settings.max_iterations = 40;
    settings.parameter_tolerances =
        Eigen::Vector2d(1.e-6, 1.e-9).cwiseQuotient(units);
    const auto stalled = DampedNewtonSolver(settings).solve(
        Eigen::Vector2d(1.e20, 1.).cwiseQuotient(units),
        [&](const Eigen::VectorXd &y) -> Eigen::VectorXd {
          const Eigen::Vector2d x = y.cwiseProduct(units);
          return Eigen::Vector2d(std::sin(x[0]), x[1] - 1.)
              .cwiseProduct(weights);
        },
        [&](const Eigen::VectorXd &y) -> Eigen::MatrixXd {
          const Eigen::Vector2d x = y.cwiseProduct(units);
          return (Eigen::Vector2d(std::cos(x[0]), 1.)
                      .cwiseProduct(units)
                      .cwiseProduct(weights))
              .asDiagonal();
        },
        {Eigen::MatrixXd(0, 2), Eigen::VectorXd(0)});
    REQUIRE_FALSE(stalled.success);
    CHECK(stalled.code == 4);
    CHECK(stalled.n_iterations < settings.max_iterations);
  }
}

TEST_CASE("Newton stagnation preserves nonlinear convergence",
          "[optim][newton]") {
  DampedNewtonSettings settings;
  settings.stagnation_iterations = 5;
  settings.parameter_tolerances = Eigen::Vector2d(1.e-12, 1.e-10);
  const auto root = DampedNewtonSolver(settings).solve(
      Eigen::Vector2d(1.5, 2.),
      [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        return Eigen::Vector2d(x[0] * x[0] - 2., std::log(x[1]) - 1.);
      },
      [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
        return Eigen::Vector2d(2. * x[0], 1. / x[1]).asDiagonal();
      },
      {Eigen::MatrixXd(0, 2), Eigen::VectorXd(0)});
  INFO(root.message);
  REQUIRE(root.success);
  CHECK_THAT(root.x[0], WithinAbs(std::sqrt(2.), 1.e-12));
  CHECK_THAT(root.x[1], WithinAbs(std::exp(1.), 1.e-10));
  CHECK(root.F_norm < 1.e-10);
}
