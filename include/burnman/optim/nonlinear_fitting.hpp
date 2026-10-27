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

#pragma once

#include <Eigen/Core>
#include <any>
#include <functional>
#include <optional>
#include <tuple>
#include <vector>

namespace burnman::optim {
/** A hypersurface model for errors-in-variables nonlinear fitting.
 *
 * function() maps a point to the surface; normal() returns a unit normal at a
 * surface point. set_params() may enforce parameter bounds. Empty flags mean
 * no flags; otherwise each observation has one opaque, model-defined flag.
 * Covariances may be singular (exact coordinates), but must have positive
 * variance in the surface-normal direction.
 */
class NonLinearModel {
public:
  virtual ~NonLinearModel() = default;
  virtual Eigen::VectorXd get_params() const = 0;
  virtual void set_params(const Eigen::VectorXd &params) = 0;
  virtual Eigen::VectorXd function(const Eigen::VectorXd &x,
                                   const std::any &flag = {}) = 0;
  virtual Eigen::VectorXd normal(const Eigen::VectorXd &x,
                                 const std::any &flag = {}) = 0;
  virtual void validate() const;
  /// Compare model-defined flags; Python adapters use Python equality.
  virtual bool flags_equal(const std::any &a, const std::any &b) const;

  Eigen::MatrixXd data;
  std::vector<Eigen::MatrixXd> data_covariances;
  Eigen::VectorXd mle_tolerances;
  Eigen::VectorXd delta_params;
  std::vector<std::any> flags;

  // Populated by nonlinear_least_squares_fit(). dof excludes prior terms.
  Eigen::Index dof = 0;
  int n_iterations = 0;
  Eigen::MatrixXd data_mle;
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd weighted_residuals;
  Eigen::VectorXd weights;
  double WSS = 0.;
  Eigen::VectorXd popt;
  Eigen::MatrixXd pcov;
  double goodness_of_fit = 0.;
  double noise_variance = 0.;
};

struct NonlinearFitOptions {
  double lm_damping = 0.;
  double param_tolerance = 1.e-7;
  int max_lm_iterations = 100;
  std::optional<Eigen::VectorXd> param_priors;
  std::optional<Eigen::MatrixXd> param_prior_inv_cov_matrix;
  bool verbose = false;
};

using MleEstimate = std::tuple<Eigen::VectorXd, double, double>;
using MleResult = std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>;

/// Covariance projected onto the unit direction of n.
double abs_line_project(const Eigen::MatrixXd &M, const Eigen::VectorXd &n);

/// Tangent-plane MLE position, signed normal residual and normal variance.
MleEstimate mle_estimate(NonLinearModel &model, const Eigen::VectorXd &x,
                         const Eigen::VectorXd &x_m, const Eigen::MatrixXd &cov,
                         const std::any &flag = {});

/** Iterative MLE positions, standardised residuals and inverse variances.
 * Throws if a projection does not converge within 10000 iterations.
 */
MleResult find_mle(NonLinearModel &model);

/// Central differences of standardised residuals; restores model parameters.
void calculate_jacobian(NonLinearModel &model);

/** Fit using BurnMan's adaptive Levenberg-Marquardt iteration.
 *
 * Mutates model parameters and fills the fit statistics above. Priors must
 * supply both a mean and a symmetric positive-semidefinite precision matrix.
 * pcov = inverse(J'J + prior_precision) * WSS / (n_data - n_params), where
 * WSS includes the prior penalty. noise_variance uses only data residuals.
 * Requires more observations than parameters and a nonsingular final normal
 * matrix. n_iterations counts accepted LM updates.
 */
void nonlinear_least_squares_fit(NonLinearModel &model,
                                 const NonlinearFitOptions &options = {});

using ScalarModelFunction = std::function<double(const Eigen::VectorXd &)>;

/** Pointwise confidence/prediction intervals for f(function(x)).
 * Rows are confidence lower/upper, then prediction lower/upper. Propagates pcov
 * using forward differences at the original queries and a positive Student-t
 * quantile. Parameters are restored on success or error; differences use actual
 * bounded displacements. Prediction noise uses matching response residuals and
 * local response-to-output conversion from parameter trials, following Python.
 * Prediction rows are NaN when conversion or response noise is unidentified.
 */
Eigen::MatrixXd confidence_prediction_bands(NonLinearModel &model,
                                            const Eigen::MatrixXd &x_array,
                                            double confidence_interval,
                                            const ScalarModelFunction &f,
                                            const std::any &flag = {});
} // namespace burnman::optim
