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

TEST_CASE("Confidence and prediction limits match closed-form line regression",
          "[nonlinear_fitting][bands]") {
  LineModel model;
  nonlinear_least_squares_fit(model);
  Matrix query(2, 2);
  query << 1., 999., 3., -999.;
  const Vector before = model.get_params();
  const Matrix actual = confidence_prediction_bands(
      model, query, 0.95, [](const Vector &point) { return point[1]; });
  const double critical = 2.7764451051977987; // t(4), 97.5th percentile.
  const double noise = model.WSS * 0.04 / 4.;
  for (Eigen::Index i = 0; i < query.rows(); ++i) {
    const Eigen::Vector2d design(query(i, 0), 1.);
    const double mean = design.dot(before);
    const double variance = design.dot(model.pcov * design);
    CHECK(actual(0, i) == Approx(mean - critical * std::sqrt(variance)));
    CHECK(actual(1, i) == Approx(mean + critical * std::sqrt(variance)));
    CHECK(actual(2, i) ==
          Approx(mean - critical * std::sqrt(variance + noise)));
    CHECK(actual(3, i) ==
          Approx(mean + critical * std::sqrt(variance + noise)));
  }
  CHECK(model.get_params() == before);
  CHECK(confidence_prediction_bands(model, Matrix(0, 2), 0.95,
                                    [](const Vector &x) { return x[1]; })
            .cols() == 0);
}

TEST_CASE("Extreme-value diagnostics reproduce Python's approximation",
          "[nonlinear_fitting][diagnostics]") {
  Vector residuals = Vector::Zero(100);
  residuals[7] = 5.;
  residuals[42] = -5.;
  const auto [bound, indices, probabilities] = extreme_values(residuals, 0.95);
  CHECK(bound == Approx(3.4600855156444577).epsilon(1.e-13));
  CHECK(indices == (std::vector<Eigen::Index>{7, 42}));
  REQUIRE(probabilities.size() == 2);
  CHECK(probabilities[0] == Approx(3.29595918e-6).epsilon(1.e-8));
  CHECK(probabilities[1] == probabilities[0]);
  const auto [empty_bound, empty_indices, empty_probabilities] =
      extreme_values(Vector::Zero(100), 0.95);
  CHECK(empty_bound == bound);
  CHECK(empty_indices.empty());
  CHECK(empty_probabilities.size() == 0);
}
