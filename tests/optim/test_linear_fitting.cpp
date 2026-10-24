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

#include "burnman/optim/linear_fitting.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

using namespace burnman::optim;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Catch::Approx;

TEST_CASE(
    "Weighted linear fits retain the parameter covariance and squared residual",
    "[linear_fitting]") {
  Matrix A(3, 2);
  A << 1., 1., 1., 2., 1., 3.;
  Vector b(3);
  b << 1., 2., 2.;
  const auto [parameters, covariance, residual] =
      weighted_constrained_least_squares(A, b);
  CHECK(parameters[0] == Approx(2. / 3.));
  CHECK(parameters[1] == Approx(0.5));
  CHECK(covariance(0, 0) == Approx(7. / 3.));
  CHECK(covariance(0, 1) == Approx(-1.));
  CHECK(covariance(1, 1) == Approx(0.5));
  CHECK(residual == Approx(1. / 6.));
  Matrix data_covariance = Matrix::Identity(3, 3);
  data_covariance(2, 2) = 2.;
  const auto [weighted, weighted_covariance, weighted_residual] =
      weighted_constrained_least_squares(A, b, data_covariance);
  CHECK(weighted[0] == Approx(4. / 7.));
  CHECK(weighted[1] == Approx(4. / 7.));
  CHECK(weighted_covariance(0, 0) == Approx(19. / 7.));
  CHECK(weighted_covariance(0, 1) == Approx(-9. / 7.));
  CHECK(weighted_covariance(1, 1) == Approx(5. / 7.));
  CHECK(weighted_residual == Approx(1. / 7.));
}

TEST_CASE("Redundant equalities specify the same fitted subspace",
          "[linear_fitting]") {
  Matrix A(3, 2), C(3, 2);
  A << 1., 2., 3., 4., 5., 6.;
  C << 1., 1., 2., 2., 0., 0.;
  Vector b(3), c(3);
  b << 7., 8., 9.;
  c << 0.5, 1., 0.;
  const auto [parameters, covariance, residual] =
      weighted_constrained_least_squares(A, b, std::nullopt,
                                         LinearConstraints{C, c});
  CHECK(parameters[0] == Approx(-6.));
  CHECK(parameters[1] == Approx(6.5));
  CHECK(residual == Approx(0.).margin(1.e-24));
  CHECK(covariance.rows() == 2);
  c[1] = 2.;
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, b, std::nullopt,
                                                     LinearConstraints{C, c}),
                  std::invalid_argument);
}

TEST_CASE(
    "Inequality constraints permit boundary fits without changing covariance",
    "[linear_fitting]") {
  Matrix A(3, 2);
  A << 1., 1., 1., 2., 1., 3.;
  Vector b(3), d(2);
  b << 1., 2., 2.;
  d << -2., 0.;
  const auto [parameters, covariance, residual] =
      weighted_constrained_least_squares(
          A, b, std::nullopt, std::nullopt,
          LinearConstraints{-Matrix::Identity(2, 2), d});
  CHECK(parameters[0] == Approx(2.));
  CHECK(parameters[1] == Approx(0.).margin(1.e-10));
  CHECK(residual == Approx(1.));
  CHECK(covariance(0, 0) == Approx(7. / 3.));
}

TEST_CASE("Mixed constraints and dependent active bounds retain their optimum",
          "[linear_fitting]") {
  Matrix D(3, 2), C(1, 2);
  D << -1., 0., 0., -1., -1., -1.;
  C << 1., 1.;
  Vector b(2);
  b << -1., 2.;
  const auto [parameters, covariance, residual] =
      weighted_constrained_least_squares(Matrix::Identity(2, 2), b,
                                         std::nullopt,
                                         LinearConstraints{C, Vector::Ones(1)},
                                         LinearConstraints{D, Vector::Zero(3)});
  CHECK(parameters[0] == Approx(0.).margin(1.e-10));
  CHECK(parameters[1] == Approx(1.));
  CHECK(residual == Approx(2.));
  CHECK(covariance.isApprox(Matrix::Identity(2, 2)));
  b << -1., -1.;
  const auto dependent = weighted_constrained_least_squares(
      Matrix::Identity(2, 2), b, std::nullopt, std::nullopt,
      LinearConstraints{D, Vector::Zero(3)});
  CHECK(std::get<0>(dependent).norm() < 1.e-10);
}

TEST_CASE("Rank deficiency and zero covariance weights have explicit meanings",
          "[linear_fitting]") {
  const Matrix A = Matrix::Identity(2, 2);
  Matrix covariance = Matrix::Zero(2, 2);
  covariance(0, 0) = 1.;
  const Vector b = Vector::Constant(2, 3.);
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, b, covariance),
                  std::invalid_argument);
  const auto [parameters, fitted_covariance, residual] =
      weighted_constrained_least_squares(A, b, covariance, std::nullopt,
                                         std::nullopt, true);
  CHECK(parameters[0] == Approx(3.));
  CHECK(parameters[1] == Approx(0.).margin(1.e-12));
  CHECK(fitted_covariance.isApprox(covariance));
  CHECK(residual == Approx(0.).margin(1.e-24));
}

TEST_CASE("Fully specified parameters still obey inequality constraints",
          "[linear_fitting]") {
  const Matrix A = Matrix::Identity(2, 2);
  const Vector b = Vector::Ones(2);
  const LinearConstraints equality{A, b};
  CHECK_NOTHROW(weighted_constrained_least_squares(
      A, b, std::nullopt, equality, LinearConstraints{-A, Vector::Zero(2)}));
  CHECK_THROWS_AS(
      weighted_constrained_least_squares(A, b, std::nullopt, equality,
                                         LinearConstraints{A, Vector::Zero(2)}),
      std::invalid_argument);
}

TEST_CASE("Linear fits reject invalid observations and indefinite covariance",
          "[linear_fitting]") {
  const Matrix A = Matrix::Identity(2, 2);
  const Vector b = Vector::Ones(2);
  CHECK_THROWS_AS(weighted_constrained_least_squares(Matrix(0, 2), Vector(0)),
                  std::invalid_argument);
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, Vector::Ones(1)),
                  std::invalid_argument);
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, b, -A),
                  std::invalid_argument);
  Matrix covariance = A;
  covariance(0, 1) = 0.5;
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, b, covariance),
                  std::invalid_argument);
  covariance(0, 1) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(weighted_constrained_least_squares(A, b, covariance),
                  std::invalid_argument);
}
