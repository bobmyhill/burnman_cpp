// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of tests/test_solvers.py; reference: tests/reference/burnman.json.

#include "burnman/optim/roots/damped_newton_solver.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace burnman::optim::roots;

namespace {
void original_equal(const Eigen::VectorXd &actual,
                    const Eigen::VectorXd &expected) {
  REQUIRE(actual.size() == expected.size());
  for (Eigen::Index i = 0; i < actual.size(); ++i)
    CHECK(
        std::abs(actual[i] - expected[i]) <=
        std::max(1.e-16,
                 std::max(std::abs(actual[i]), std::abs(expected[i])) * 1.e-5));
}
} // namespace

TEST_CASE("Original Rosenbrock root tests", "[original][newton]") {
  auto F = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    return Eigen::Vector2d(10. * (x[1] - x[0] * x[0]), 1. - x[0]);
  };
  auto J = [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
    Eigen::Matrix2d matrix;
    matrix << -20. * x[0], 10., -1., 0.;
    return matrix;
  };
  const Eigen::Vector2d guess(-1.2, 1.);
  SECTION("test_dns_rosenbrock_1") {
    const auto sol = DampedNewtonSolver().solve(guess, F, J);
    REQUIRE(sol.success);
    original_equal(sol.x, Eigen::Vector2d(1., 1.));
  }
  SECTION("test_dns_rosenbrock_1_w_constraint_violation") {
    const auto sol = DampedNewtonSolver().solve(
        guess, F, J, {Eigen::RowVector2d(1., 0.), Eigen::VectorXd::Zero(1)});
    original_equal(sol.x.head(1), Eigen::VectorXd::Zero(1));
    CHECK(sol.code == 2);
  }
  SECTION("test_dns_rosenbrock_1_w_temporary_constraint_violation") {
    const auto sol = DampedNewtonSolver().solve(
        guess, F, J,
        {(Eigen::Matrix2d() << 1., 0., 0., -1.).finished(),
         Eigen::Vector2d(-1., -3.)});
    original_equal(sol.x, Eigen::Vector2d(1., 1.));
  }
}

TEST_CASE("Original Freudenstein Roth root test", "[original][newton]") {
  // test_solvers.test_dns_freudenstein_roth_2
  auto F = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    return Eigen::Vector2d(-13. + x[0] + ((5. - x[1]) * x[1] - 2.) * x[1],
                           -29. + x[0] + ((x[1] + 1.) * x[1] - 14.) * x[1]);
  };
  auto J = [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
    return (Eigen::Matrix2d() << 1., (10. - 3. * x[1]) * x[1] - 2., 1.,
            (3. * x[1] + 2.) * x[1] - 14.)
        .finished();
  };
  const auto sol = DampedNewtonSolver().solve(Eigen::Vector2d(10., 10.), F, J);
  REQUIRE(sol.success);
  original_equal(sol.x, Eigen::Vector2d(5., 4.));
}

TEST_CASE("Original Powell root tests", "[original][newton]") {
  SECTION("test_dns_powell_badly_scaled_3") {
    auto F = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      return Eigen::Vector2d(1.e4 * x[0] * x[1] - 1.,
                             std::exp(-x[0]) + std::exp(-x[1]) - 1.0001);
    };
    auto J = [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
      return (Eigen::Matrix2d() << 1.e4 * x[1], 1.e4 * x[0], -std::exp(-x[0]),
              -std::exp(-x[1]))
          .finished();
    };
    const auto sol = DampedNewtonSolver().solve(Eigen::Vector2d(0., 1.), F, J);
    REQUIRE(sol.success);
    original_equal(sol.x, Eigen::Vector2d(1.09815933e-5, 9.10614674));
  }
  SECTION("test_dns_powell_singular_13") {
    auto F = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      return Eigen::Vector4d(x[0] + 10. * x[1], std::sqrt(5.) * (x[2] - x[3]),
                             std::pow(x[1] - 2. * x[2], 2),
                             std::sqrt(10.) * std::pow(x[0] - x[3], 2));
    };
    auto J = [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
      return (Eigen::Matrix4d() << 1., 10., 0., 0., 0., 0., std::sqrt(5.),
              -std::sqrt(5.), 0., 2. * x[1] - 4. * x[2], 8. * x[2] - 4. * x[1],
              0., std::sqrt(10.) * (2. * x[0] - 2. * x[3]), 0., 0.,
              std::sqrt(10.) * (2. * x[3] - 2. * x[0]))
          .finished();
    };
    const auto sol =
        DampedNewtonSolver().solve(Eigen::Vector4d(3., -1., 0., 1.), F, J);
    REQUIRE(sol.success);
    original_equal(sol.x.array() + 1., Eigen::Vector4d::Ones());
  }
}

TEST_CASE("Original Broyden tridiagonal root test", "[original][newton]") {
  // test_solvers.test_dns_broyden_tridiagonal_30
  const std::vector<std::vector<double>> expected{
      {-.28077641},
      {-.45328926, -.38540505},
      {-.52677285, -.56764891, -.41031222},
      {-.55457673, -.63942044, -.59070079, -.41526838},
      {-.5648284, -.66627372, -.66091704, -.59505005, -.41620111}};
  for (Eigen::Index n = 1; n <= 5; ++n) {
    CAPTURE(n);
    auto F = [n](const Eigen::VectorXd &x) -> Eigen::VectorXd {
      Eigen::VectorXd f(n);
      for (Eigen::Index i = 0; i < n; ++i)
        f[i] = (3. - 2. * x[i]) * x[i] + 1. - (i ? x[i - 1] : 0.) -
               (i + 1 < n ? 2. * x[i + 1] : 0.);
      return f;
    };
    auto J = [n](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
      Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n, n);
      matrix.diagonal() = (3. - 4. * x.array()).matrix();
      for (Eigen::Index i = 1; i < n; ++i) {
        matrix(i, i - 1) = -1.;
        matrix(i - 1, i) = -2.;
      }
      return matrix;
    };
    const auto sol =
        DampedNewtonSolver().solve(Eigen::VectorXd::Constant(n, -1.), F, J);
    REQUIRE(sol.success);
    original_equal(sol.x,
                   Eigen::Map<const Eigen::VectorXd>(
                       expected[static_cast<std::size_t>(n - 1)].data(), n));
  }
}

TEST_CASE("Original generalised Rosenbrock root tests", "[original][newton]") {
  auto F = [](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    return Eigen::Vector2d(2. * (x[0] - 1.) + 60. * x[0] * (x[0] * x[0] - x[1]),
                           30. * (x[1] - x[0] * x[0]));
  };
  auto J = [](const Eigen::VectorXd &x) -> Eigen::MatrixXd {
    return (Eigen::Matrix2d() << 180. * x[0] * x[0] - 60. * x[1] + 2.,
            -60. * x[0], -60. * x[0], 30.)
        .finished();
  };
  DampedNewtonSettings settings;
  settings.tol = 1.e-15;
  Eigen::Vector2d guess(-4., -3.);
  LinearConstraints constraints{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0)};
  bool violation = false;
  SECTION("test_dns_rosenbrock_generalised") {
    guess = Eigen::Vector2d(-1., 1.);
    settings.max_iterations = 600;
    settings.store_iterates = true;
  }
  SECTION("test_dns_rosenbrock_generalised_w_descent_path_constraint") {
    settings.max_iterations = 100;
    constraints = {Eigen::RowVector2d(0., 1.),
                   Eigen::VectorXd::Constant(1, -1.1)};
  }
  SECTION("test_dns_rosenbrock_generalised_w_two_descent_path_constraints") {
    settings.max_iterations = 200;
    constraints = {(Eigen::Matrix2d() << 0., 1., -1., 1.).finished(),
                   Eigen::Vector2d(-1.1, -3.5)};
  }
  SECTION("test_dns_rosenbrock_generalised_w_constraint_violation") {
    settings.max_iterations = 200;
    settings.store_iterates = true;
    constraints = {Eigen::RowVector2d(0., 1.),
                   Eigen::VectorXd::Constant(1, -.6)};
    violation = true;
  }
  const auto sol = DampedNewtonSolver(settings).solve(guess, F, J, constraints);
  INFO(sol.message);
  if (violation) {
    original_equal(sol.x.tail(1), Eigen::VectorXd::Constant(1, .6));
    CHECK(sol.code == 2);
  } else {
    REQUIRE(sol.success);
    original_equal(sol.x, Eigen::Vector2d(1., 1.));
  }
}
