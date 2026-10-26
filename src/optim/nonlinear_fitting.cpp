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
      const Vector res_0 = std::get<1>(find_mle_impl(model));
      perturbed[j] = params[j] + model.delta_params[j];
      model.set_params(perturbed);
      const Vector res_1 = std::get<1>(find_mle_impl(model));
      jacobian.col(j) = (res_1 - res_0) / (2. * model.delta_params[j]);
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
} // namespace burnman::optim
