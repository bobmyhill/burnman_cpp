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

TEST_CASE("Newton solves rank-deficient systems without invalid trials",
          "[optim][newton]") {
  DampedNewtonSettings settings;
  settings.residual_tolerances = Eigen::Vector2d::Constant(1.e-10);
  const auto jacobian = [](const auto &) -> Eigen::MatrixXd {
    return (Eigen::Matrix2d() << 1., 1., 2., 2.).finished();
  };
  auto residual = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    REQUIRE(x.allFinite());
    return Eigen::Vector2d(x.sum() - 2., 2. * x.sum() - 4.);
  };
  const auto root = DampedNewtonSolver(settings).solve(Eigen::Vector2d::Zero(),
                                                       residual, jacobian);
  INFO(root.message);
  REQUIRE(root.success);
  CHECK(root.code == 5);
  CHECK(root.jacobian_rank == 1);
  CHECK(root.x.allFinite());
  CHECK_THAT(root.x.sum(), WithinAbs(2., 1.e-10));

  // The same Jacobian cannot satisfy inconsistent equations. Its null-space
  // residual must not be hidden by a zero minimum-norm Newton correction.
  auto inconsistent = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    REQUIRE(x.allFinite());
    return Eigen::Vector2d(x.sum() - 2., 2. * x.sum() - 5.);
  };
  const auto failed = DampedNewtonSolver(settings).solve(
      Eigen::Vector2d::Zero(), inconsistent, jacobian);
  REQUIRE_FALSE(failed.success);
  CHECK(failed.code == 6);
  CHECK(failed.x.allFinite());
  CHECK(failed.F_norm > 0.1);
}

TEST_CASE("Newton rank detection respects equation and parameter units",
          "[optim][newton]") {
  // Unscaled SVD truncates the pressure column relative to composition.
  DampedNewtonSettings settings;
  settings.parameter_tolerances = Eigen::Vector2d(1., 1.e-9);
  settings.residual_tolerances = Eigen::Vector2d::Constant(1.e-10);
  const auto root = DampedNewtonSolver(settings).solve(
      Eigen::Vector2d::Zero(),
      [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        return Eigen::Vector2d(1.e-12 * x[0] + x[1] - 2., 1.e-12 * x[0] - x[1]);
      },
      [](const auto &) -> Eigen::MatrixXd {
        return (Eigen::Matrix2d() << 1.e-12, 1., 1.e-12, -1.).finished();
      });
  REQUIRE(root.success);
  CHECK(root.code == 0);
  CHECK(root.jacobian_rank == 2);
  CHECK_THAT(root.x[0], WithinAbs(1.e12, 1.));
  CHECK_THAT(root.x[1], WithinAbs(1., 1.e-9));
  CHECK(root.F_norm < 1.e-10);
}

TEST_CASE("Newton validates component residual tolerances", "[optim][newton]") {
  DampedNewtonSettings settings;
  settings.residual_tolerances = Eigen::VectorXd::Constant(1, -1.);
  CHECK_THROWS_AS(
      DampedNewtonSolver(settings).solve(
          Eigen::Vector2d::Zero(), [](const Eigen::VectorXd &x) { return x; },
          [](const auto &) { return Eigen::Matrix2d::Identity().eval(); }),
      std::invalid_argument);
}

TEST_CASE("Newton detects duplicate columns despite a benign LU estimate",
          "[optim][newton]") {
  // A duplicate-phase Jacobian can have an apparently benign LU rcond even
  // though changing the split between the identical phases is unobservable.
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(6, 6);
  matrix(0, 0) = matrix(1, 1) = 1.;
  matrix.row(2) << 0., 0., 0., 3980., 0., -3980.;
  matrix.row(3) << 0., 0., 0., -35820., 0., 35820.;
  matrix.row(4) << 0., 0., 1.8, -1., 1.8, -1.;
  matrix.row(5) << 0., 0., .2, 1., .2, 1.;
  DampedNewtonSettings settings;
  settings.residual_tolerances = Eigen::VectorXd::Constant(6, 1.e-9);
  const auto root = DampedNewtonSolver(settings).solve(
      Eigen::VectorXd::Zero(6),
      [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
        REQUIRE(x.allFinite());
        return matrix * (x.array() - 1.).matrix();
      },
      [&](const auto &) { return matrix; });
  INFO(root.message);
  REQUIRE(root.success);
  CHECK(root.code == 5);
  CHECK(root.jacobian_rank == 5);
  CHECK(root.F_norm < 1.e-9);
}
