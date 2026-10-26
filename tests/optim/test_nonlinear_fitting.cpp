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

#include "burnman/optim/nonlinear_fitting.hpp"
#include <Eigen/LU>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman::optim;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Catch::Approx;

namespace {
class LineModel : public NonLinearModel {
public:
  LineModel() {
    data.resize(6, 2);
    data << 0., 1.1, 1., 2.9, 2., 5.2, 3., 6.8, 4., 9.1, 5., 10.9;
    Matrix covariance = Matrix::Zero(2, 2);
    covariance(1, 1) = 0.04;
    data_covariances.assign(6, covariance);
    mle_tolerances = Vector::Constant(6, 1.e-10);
    delta_params = Vector::Constant(2, 1.e-6);
    params.resize(2);
    params << 0.5, 0.5;
  }
  Vector get_params() const override { return params; }
  void set_params(const Vector &values) override { params = values; }
  Vector function(const Vector &x, const std::any &) override {
    Vector point(2);
    point << x[0], params[0] * x[0] + params[1];
    return point;
  }
  Vector normal(const Vector &, const std::any &) override {
    Vector n(2);
    n << params[0], -1.;
    return n.normalized();
  }
  Vector params;
};
} // namespace

TEST_CASE("Nonlinear fits match closed-form weighted least squares",
          "[nonlinear_fitting]") {
  LineModel model;
  Matrix X(model.data.rows(), 2);
  X
  .col(0) = model.data.col(0);
  X
  .col(1).setOnes();
  NonlinearFitOptions options;
  Matrix JTJ = X
  .transpose() * X
  / 0.04;
  Vector JTy = X
  .transpose() * model.data.col(1) / 0.04;
  SECTION("With a correlated Gaussian prior") {
    Vector mean(2);
    mean << 1.5, 1.5;
    Matrix precision(2, 2);
    precision << 10., 1., 1., 2.;
    options.param_priors = mean;
    options.param_prior_inv_cov_matrix = precision;
    JTJ += precision;
    JTy += precision * mean;
  }
  SECTION("Without priors") {}
  const Vector expected = JTJ.partialPivLu().solve(JTy);
  nonlinear_least_squares_fit(model, options);
  CHECK(model.popt.isApprox(expected, 1.e-7));
  double wss = (X * expected - model.data.col(1)).squaredNorm() / 0.04;
  if (options.param_priors) {
    const Vector residual = expected - *options.param_priors;
    wss += residual.dot(*options.param_prior_inv_cov_matrix * residual);
  }
  CHECK(model.WSS == Approx(wss).epsilon(1.e-8));
  CHECK(model.dof == 4);
  CHECK(model.goodness_of_fit == Approx(wss / 4.));
  CHECK(model.pcov.isApprox(JTJ.inverse() * wss / 4., 1.e-6));
  CHECK(model.noise_variance ==
        Approx((X * expected - model.data.col(1)).squaredNorm() /
               (4. * (1. + expected[0] * expected[0]))));
  const Matrix jacobian = model.jacobian;
  calculate_jacobian(model);
  const auto [positions, residuals, weights] = find_mle(model);
  CHECK(model.params == model.popt);
  CHECK(model.jacobian == jacobian);
  CHECK(model.data_mle == positions);
  CHECK(model.weighted_residuals == residuals);
  CHECK(model.weights == weights);
}

TEST_CASE("Nonlinear projections use full observation covariances",
          "[nonlinear_fitting]") {
  LineModel model;
  Vector x(2);
  x << 2., 4.;
  const Vector point = model.function(x, {});
  const Vector n = model.normal(point, {});
  Matrix cov(2, 2);
  cov << 0.4, 0.1, 0.1, 0.7;
  const auto [estimate, residual, variance] =
      mle_estimate(model, x, point, cov);
  CHECK(variance == Approx(n.dot(cov * n)));
  CHECK(abs_line_project(cov, 3. * n) == Approx(variance));
  CHECK(residual == Approx((point - x).dot(n)));
  CHECK(estimate.isApprox(x + residual / variance * cov * n));
  CHECK(model.function(estimate, {}).isApprox(estimate));
}

TEST_CASE("Nonlinear fits reject singular problems and invalid dimensions",
          "[nonlinear_fitting]") {
  LineModel model;
  SECTION("Zero variance normal to the curve") {
    model.data_covariances.assign(6, Matrix::Zero(2, 2));
    CHECK_THROWS_AS(find_mle(model), std::invalid_argument);
  }
  SECTION("No residual degrees of freedom") {
    model.data.conservativeResize(2, 2);
    model.data_covariances.resize(2);
    model.mle_tolerances.conservativeResize(2);
    CHECK_THROWS_AS(nonlinear_least_squares_fit(model), std::invalid_argument);
  }
  SECTION("Indistinguishable line parameters") {
    model.data.col(0).setZero();
    CHECK_THROWS_AS(nonlinear_least_squares_fit(model), std::runtime_error);
  }
}
