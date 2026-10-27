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
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <gsl/gsl_cdf.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace burnman::optim {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
constexpr int max_mle_iterations = 10000;

void check_vector(const Vector &value, Eigen::Index size, const char *name) {
  if (value.size() != size || !value.allFinite())
    throw std::invalid_argument(std::string(name) +
                                " has the wrong size or non-finite values.");
}

void check_covariance(const Matrix &matrix, Eigen::Index size,
                      const char *name) {
  if (matrix.rows() != size || matrix.cols() != size || !matrix.allFinite())
    throw std::invalid_argument(std::string(name) +
                                " has the wrong shape or non-finite values.");
  const double scale = matrix.cwiseAbs().maxCoeff();
  if (scale == 0.)
    return;
  const Matrix scaled = matrix / scale;
  if ((scaled - scaled.transpose()).cwiseAbs().maxCoeff() > 1.e-12)
    throw std::invalid_argument(std::string(name) + " must be symmetric.");
  Eigen::SelfAdjointEigenSolver<Matrix> eigen(scaled, Eigen::EigenvaluesOnly);
  if (eigen.info() != Eigen::Success ||
      eigen.eigenvalues().minCoeff() < -1.e-12)
    throw std::invalid_argument(std::string(name) +
                                " must be positive semidefinite.");
}

void validate_data(const NonLinearModel &model) {
  const auto n = model.data.rows();
  const auto d = model.data.cols();
  if (n == 0 || d == 0 || !model.data.allFinite() ||
      static_cast<Eigen::Index>(model.data_covariances.size()) != n)
    throw std::invalid_argument(
        "Require finite, nonempty data and one covariance per observation.");
  check_vector(model.mle_tolerances, n, "mle_tolerances");
  if ((model.mle_tolerances.array() < 0.).any())
    throw std::invalid_argument("mle_tolerances must be nonnegative.");
  if (!model.flags.empty() &&
      static_cast<Eigen::Index>(model.flags.size()) != n)
    throw std::invalid_argument("Require one flag per observation.");
  for (const auto &cov : model.data_covariances)
    check_covariance(cov, d, "Data covariance");
}

void validate_differences(const NonLinearModel &model) {
  const Vector params = model.get_params();
  if (params.size() == 0 || !params.allFinite())
    throw std::invalid_argument("Require finite, nonempty model parameters.");
  check_vector(model.delta_params, params.size(), "delta_params");
  if ((model.delta_params.array() == 0.).any())
    throw std::invalid_argument("delta_params must be nonzero.");
}

MleEstimate mle_estimate_impl(NonLinearModel &model, const Vector &x,
                              const Vector &x_m, const Matrix &cov,
                              const std::any &flag) {
  const Vector n = model.normal(x_m, flag);
  check_vector(n, x.size(), "Model normal");
  const double variance = abs_line_project(cov, n);
  if (!std::isfinite(variance) || variance <= 0.)
    throw std::invalid_argument(
        "Data covariance must have positive variance along the model normal.");
  const double residual = (x_m - x).dot(n);
  Vector estimate = x + (residual / variance) * (cov * n);
  if (!estimate.allFinite() || !std::isfinite(residual))
    throw std::runtime_error("Non-finite MLE projection.");
  return {std::move(estimate), residual, variance};
}

MleResult find_mle_impl(NonLinearModel &model) {
  Matrix positions(model.data.rows(), model.data.cols());
  Vector residuals(model.data.rows()), weights(model.data.rows());
  const std::any no_flag;
  for (Eigen::Index i = 0; i < model.data.rows(); ++i) {
    const Vector x = model.data.row(i).transpose();
    const auto &flag = model.flags.empty()
                           ? no_flag
                           : model.flags[static_cast<std::size_t>(i)];
    Vector point = model.function(x, flag);
    check_vector(point, x.size(), "Model function");
    auto [estimate, residual, variance] = mle_estimate_impl(
        model, x, point, model.data_covariances[static_cast<std::size_t>(i)],
        flag);
    double distance = (point - x).norm();
    int iteration = 0;
    while (distance > model.mle_tolerances[i]) {
      if (iteration++ == max_mle_iterations)
        throw std::runtime_error("MLE projection did not converge for datum " +
                                 std::to_string(i) + ".");
      // The first tangent-plane estimate is already available. Subsequent
      // estimates use the newly evaluated point on the model surface.
      if (iteration > 1)
        std::tie(estimate, residual, variance) = mle_estimate_impl(
            model, x, point,
            model.data_covariances[static_cast<std::size_t>(i)], flag);
      point = model.function(estimate, flag);
      check_vector(point, x.size(), "Model function");
      distance = (point - estimate).norm();
    }
    positions.row(i) = point.transpose();
    residuals[i] = residual / std::sqrt(variance);
    weights[i] = 1. / variance;
  }
  if (!residuals.allFinite() || !weights.allFinite())
    throw std::runtime_error("Non-finite MLE residuals or weights.");
  return {std::move(positions), std::move(residuals), std::move(weights)};
}

void calculate_jacobian_impl(NonLinearModel &model) {
  const Vector params = model.get_params();
  Matrix jacobian(model.data.rows(), params.size());
  try {
    for (Eigen::Index j = 0; j < params.size(); ++j) {
      Vector perturbed = params;
      perturbed[j] -= model.delta_params[j];
      model.set_params(perturbed);
      const double p0 = model.get_params()[j];
      const Vector res_0 = std::get<1>(find_mle_impl(model));
      perturbed[j] = params[j] + model.delta_params[j];
      model.set_params(perturbed);
      const double p1 = model.get_params()[j];
      if (p1 == p0)
        throw std::invalid_argument("Cannot perturb parameter " +
                                    std::to_string(j) +
                                    ": both trial values are identical.");
      const Vector res_1 = std::get<1>(find_mle_impl(model));
      jacobian.col(j) = (res_1 - res_0) / (p1 - p0);
    }
  } catch (...) {
    model.set_params(params);
    throw;
  }
  model.set_params(params);
  model.jacobian = std::move(jacobian);
}

void refresh_residuals(NonLinearModel &model) {
  std::tie(model.data_mle, model.weighted_residuals, model.weights) =
      find_mle_impl(model);
}

// Match numpy.linalg.solve/inv: pivoted LU, without a relative rank cutoff
// that could reject valid EOS fits whose parameters have very different units.
Eigen::PartialPivLU<Matrix> factorise(const Matrix &matrix) {
  if (!matrix.allFinite())
    throw std::runtime_error("Non-finite fitting normal matrix.");
  Eigen::PartialPivLU<Matrix> lu(matrix);
  if ((lu.matrixLU().diagonal().array() == 0.).any())
    throw std::runtime_error("Singular or non-finite fitting normal matrix.");
  return lu;
}
} // namespace

void NonLinearModel::validate() const {
  validate_data(*this);
  validate_differences(*this);
}

double abs_line_project(const Matrix &M, const Vector &n) {
  if (n.size() == 0 || !n.allFinite() || M.rows() != n.size() ||
      M.cols() != n.size() || !M.allFinite())
    throw std::invalid_argument("Require a finite covariance and direction.");
  const double length = n.stableNorm();
  if (!std::isfinite(length) || length == 0.)
    throw std::invalid_argument("Model normal must be nonzero and finite.");
  const Vector unit = n / length;
  return unit.dot(M * unit);
}

MleEstimate mle_estimate(NonLinearModel &model, const Vector &x,
                         const Vector &x_m, const Matrix &cov,
                         const std::any &flag) {
  if (x.size() == 0 || !x.allFinite())
    throw std::invalid_argument("Require a finite, nonempty observation.");
  check_vector(x_m, x.size(), "Model point");
  check_covariance(cov, x.size(), "Data covariance");
  return mle_estimate_impl(model, x, x_m, cov, flag);
}

MleResult find_mle(NonLinearModel &model) {
  validate_data(model);
  return find_mle_impl(model);
}

void calculate_jacobian(NonLinearModel &model) {
  model.validate();
  calculate_jacobian_impl(model);
}

void nonlinear_least_squares_fit(NonLinearModel &model,
                                 const NonlinearFitOptions &options) {
  model.validate();
  const auto n_params = model.get_params().size();
  if (model.data.rows() <= n_params)
    throw std::invalid_argument("Require more observations than parameters.");
  if (!std::isfinite(options.lm_damping) || options.lm_damping < 0. ||
      !std::isfinite(options.param_tolerance) ||
      options.param_tolerance <= 0. || options.max_lm_iterations <= 0)
    throw std::invalid_argument(
        "Require nonnegative damping, positive parameter tolerance and "
        "a positive iteration limit.");
  if (options.param_priors.has_value() !=
      options.param_prior_inv_cov_matrix.has_value())
    throw std::invalid_argument("Supply both prior means and prior precision.");
  const bool with_priors = options.param_priors.has_value();
  if (with_priors) {
    check_vector(*options.param_priors, n_params, "Prior means");
    check_covariance(*options.param_prior_inv_cov_matrix, n_params,
                     "Prior precision");
  }
  const auto objective = [&](const Vector &residuals, const Vector &params) {
    double value = residuals.squaredNorm();
    if (with_priors) {
      const Vector difference = params - *options.param_priors;
      value += difference.dot(*options.param_prior_inv_cov_matrix * difference);
    }
    if (!std::isfinite(value))
      throw std::runtime_error("Non-finite fitting objective.");
    return value;
  };

  model.dof = model.data.rows() - n_params;
  model.n_iterations = 0;
  double damping = options.lm_damping;
  bool converged = false;
  for (int iteration = 0; iteration < options.max_lm_iterations; ++iteration) {
    const Vector current = model.get_params();
    try {
      calculate_jacobian_impl(model);
      refresh_residuals(model);
      const double current_wss = objective(model.weighted_residuals, current);
      Matrix JTJ = model.jacobian.transpose() * model.jacobian;
      Vector JTr = model.jacobian.transpose() * model.weighted_residuals;
      if (with_priors) {
        JTJ += *options.param_prior_inv_cov_matrix;
        JTr += *options.param_prior_inv_cov_matrix *
               (current - *options.param_priors);
      }
      bool accepted = false;
      double max_fraction = 0.;
      for (int attempt = 0; attempt < 30; ++attempt) {
        Matrix A = JTJ;
        A.diagonal() += damping * JTJ.diagonal();
        const Vector step = factorise(A).solve(JTr);
        if (!step.allFinite())
          throw std::runtime_error("Non-finite LM parameter update.");
        model.set_params(current - step);
        const Vector updated = model.get_params();
        check_vector(updated, n_params, "Updated parameters");
        max_fraction = 0.;
        for (Eigen::Index j = 0; j < n_params; ++j) {
          const double denominator =
              std::abs(updated[j]) < options.param_tolerance
                  ? options.param_tolerance
                  : updated[j];
          max_fraction = std::max(
              max_fraction, std::abs((current[j] - updated[j]) / denominator));
        }
        const double updated_wss =
            objective(std::get<1>(find_mle_impl(model)), updated);
        if (updated_wss < current_wss ||
            max_fraction < options.param_tolerance) {
          damping /= 10.;
          accepted = true;
          break;
        }
        model.set_params(current);
        damping = damping > 0. ? damping * 10. : 1.e-3;
        if (options.verbose)
          std::cout << "  step rejected (WSS " << current_wss << " -> "
                    << updated_wss << "), increasing damping to " << damping
                    << '\n';
      }
      if (!accepted)
        throw std::runtime_error(
            "Could not reduce the objective after 30 LM damping attempts.");
      model.n_iterations = iteration + 1;
      if (options.verbose)
        std::cout << "Iteration " << iteration
                  << ": max param change = " << max_fraction << '\n';
      if (max_fraction < options.param_tolerance) {
        converged = true;
        break;
      }
    } catch (const std::exception &error) {
      model.set_params(current);
      throw std::runtime_error("During non-linear fitting, iteration " +
                               std::to_string(iteration) + ": " + error.what());
    }
  }
  if (!converged)
    throw std::runtime_error("Maximum number of LM iterations reached before "
                             "convergence.");

  calculate_jacobian_impl(model);
  refresh_residuals(model);
  model.popt = model.get_params();
  model.WSS = objective(model.weighted_residuals, model.popt);
  Matrix JTJ = model.jacobian.transpose() * model.jacobian;
  if (with_priors)
    JTJ += *options.param_prior_inv_cov_matrix;
  model.goodness_of_fit = model.WSS / static_cast<double>(model.dof);
  model.pcov = factorise(JTJ).solve(Matrix::Identity(n_params, n_params)) *
               model.goodness_of_fit;
  model.noise_variance =
      (model.weighted_residuals.array().square() / model.weights.array())
          .sum() /
      static_cast<double>(model.dof);
  if (!model.pcov.allFinite() || !std::isfinite(model.noise_variance))
    throw std::runtime_error("Non-finite fitted covariance or noise variance.");
  if (options.verbose)
    std::cout << std::flush;
}

bool NonLinearModel::flags_equal(const std::any &a, const std::any &b) const {
  if (!a.has_value() || !b.has_value())
    return !a.has_value() && !b.has_value();
  if (a.type() == typeid(std::string) && b.type() == typeid(std::string))
    return std::any_cast<const std::string &>(a) ==
           std::any_cast<const std::string &>(b);
  throw std::invalid_argument("Native model flags must be strings or empty.");
}

Matrix confidence_prediction_bands(NonLinearModel &model, const Matrix &x_array,
                                   double confidence_interval,
                                   const ScalarModelFunction &f,
                                   const std::any &flag) {
  validate_differences(model);
  const Vector params = model.get_params();
  const auto n = x_array.rows(), p = params.size(), d = model.data.cols();
  if (d == 0 || x_array.cols() != d || !x_array.allFinite())
    throw std::invalid_argument(
        "Queries must be finite and match the model dimension.");
  if (!f || !std::isfinite(confidence_interval) || confidence_interval <= 0. ||
      confidence_interval >= 1. || model.dof <= 0)
    throw std::invalid_argument("Require a scalar function, 0 < "
                                "confidence_interval < 1 and positive dof.");
  check_covariance(model.pcov, p, "pcov");
  if (n == 0)
    return Matrix(4, 0);

  Vector mean(n), fitted_quantity(n), parameter_steps = Vector::Zero(p);
  Matrix gradient(n, p), fitted_changes = Matrix::Zero(n, p);
  double noise_variance = std::numeric_limits<double>::quiet_NaN();
  try {
    double squared_residuals = 0.;
    Eigen::Index matching = 0;
    for (Eigen::Index i = 0; i < model.data.rows(); ++i) {
      const auto &observed_flag =
          model.flags.empty() ? flag : model.flags[static_cast<std::size_t>(i)];
      if (model.flags_equal(observed_flag, flag)) {
        const Vector x = model.data.row(i).transpose();
        const double residual =
            model.function(x, observed_flag)[d - 1] - x[d - 1];
        squared_residuals += residual * residual;
        ++matching;
      }
    }
    const auto noise_dof = model.dof - (model.data.rows() - matching);
    if (noise_dof > 0)
      noise_variance = squared_residuals / static_cast<double>(noise_dof);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Vector point = model.function(x_array.row(i).transpose(), flag);
      check_vector(point, d, "Model function");
      fitted_quantity[i] = point[d - 1];
      mean[i] = f(point);
    }
    for (Eigen::Index j = 0; j < p; ++j) {
      Vector shifted = params;
      shifted[j] += model.delta_params[j];
      model.set_params(shifted);
      double step = model.get_params()[j] - params[j];
      if (step == 0.) { // Try the other side at a parameter bound.
        shifted[j] = params[j] - model.delta_params[j];
        model.set_params(shifted);
        step = model.get_params()[j] - params[j];
      }
      if (step == 0.) {
        if (model.pcov(j, j) != 0.)
          throw std::invalid_argument(
              "Cannot perturb a parameter with nonzero covariance.");
        gradient.col(j).setZero();
        continue;
      }
      parameter_steps[j] = step;
      for (Eigen::Index i = 0; i < n; ++i) {
        const Vector point = model.function(x_array.row(i).transpose(), flag);
        check_vector(point, d, "Model function");
        fitted_changes(i, j) = point[d - 1] - fitted_quantity[i];
        gradient(i, j) = (f(point) - mean[i]) / step;
      }
    }
    model.set_params(params);
    model.function(x_array.row(n - 1).transpose(), flag);
  } catch (...) {
    model.set_params(params);
    throw;
  }

  const double critical = gsl_cdf_tdist_Qinv(0.5 * (1. - confidence_interval),
                                             static_cast<double>(model.dof));
  Matrix bands(4, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Vector derivative = gradient.row(i).transpose();
    const double variance = derivative.dot(model.pcov * derivative);
    const double confidence_width = critical * std::sqrt(variance);
    double prediction_noise = std::numeric_limits<double>::quiet_NaN();
    // Match Python: identify local response conversion from actual trial
    // changes.
    const Vector dy = fitted_changes.row(i).transpose();
    const Vector df = derivative.cwiseProduct(parameter_steps);
    const double dy_squared = dy.squaredNorm();
    if (dy_squared > 0.) {
      const double conversion = dy.dot(df) / dy_squared;
      if ((df - conversion * dy).norm() <= 1.e-5 * df.norm())
        prediction_noise = conversion * conversion * noise_variance;
    }
    const double prediction_width =
        critical * std::sqrt(variance + prediction_noise);
    bands(0, i) = mean[i] - confidence_width;
    bands(1, i) = mean[i] + confidence_width;
    bands(2, i) = mean[i] - prediction_width;
    bands(3, i) = mean[i] + prediction_width;
  }
  if (!bands.topRows(2).allFinite())
    throw std::runtime_error("Non-finite confidence bands.");
  return bands;
}

ExtremeValues extreme_values(const Vector &weighted_residuals,
                             double confidence_interval) {
  const auto n = weighted_residuals.size();
  if (n == 0)
    throw std::invalid_argument("Require at least one residual.");
  if (n == 1 || !std::isfinite(confidence_interval) ||
      confidence_interval < 0. || confidence_interval > 1.)
    return {std::numeric_limits<double>::quiet_NaN(), {}, Vector(0)};

  const double count = static_cast<double>(n);
  const double mean = gsl_cdf_ugaussian_Qinv(1. / count);
  const double logarithm = std::log(count);
  const double scale = 0.8 / std::sqrt(logarithm);
  const double c = 0.33 / std::pow(logarithm, 0.75);
  const double p = 1. - std::sqrt(confidence_interval);
  // scipy.stats.genextreme(c) is a reversed, shifted Weibull distribution.
  const double confidence_bound =
      mean + scale * (1. - gsl_cdf_weibull_Pinv(p, 1., 1. / c)) / c;

  std::vector<Eigen::Index> indices;
  for (Eigen::Index i = 0; i < n; ++i)
    if (std::abs(weighted_residuals[i]) > confidence_bound)
      indices.push_back(i);
  Vector probabilities(static_cast<Eigen::Index>(indices.size()));
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const double x = (std::abs(weighted_residuals[indices[i]]) - mean) / scale;
    const double t = 1. - c * x;
    const double survival = t > 0. ? gsl_cdf_weibull_P(t, 1., 1. / c) : 0.;
    probabilities[static_cast<Eigen::Index>(i)] =
        1. - std::pow(survival - 1., 2.);
  }
  return {confidence_bound, std::move(indices), std::move(probabilities)};
}
} // namespace burnman::optim
