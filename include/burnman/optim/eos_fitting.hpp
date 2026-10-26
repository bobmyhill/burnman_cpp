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

#include "burnman/core/mineral.hpp"
#include "burnman/optim/nonlinear_fitting.hpp"
#include <Eigen/Core>
#include <string>
#include <vector>

namespace burnman {
class Material;

namespace optim {
/** Default MLE tolerances, following Python BurnMan's property flags.
 *
 * Sets and leaves material at 1e5 Pa and 300 K. Energy flags use 1 J/mol,
 * pressure flags use 100 Pa, and other flags use 1e-5 times the property
 * value at that state. Throws if any resulting tolerance is not positive.
 */
Eigen::VectorXd default_mle_tolerances(Material &material,
                                       const std::vector<std::string> &flags);

/** Fit mineral parameters to P-T-property observations, as in Python BurnMan.
 * Numeric scalar parameters and coefficient arrays are flattened in fit_params
 * order. set_params() clips to optional bounds and resets the mineral cache.
 * The supplied mineral is shared and updated in place.
 */
class MineralFit : public NonLinearModel {
public:
  MineralFit(std::shared_ptr<Mineral> mineral, const Eigen::MatrixXd &data,
             const std::vector<Eigen::MatrixXd> &data_covariances,
             const std::vector<std::string> &flags,
             const std::vector<std::string> &fit_params,
             const Eigen::VectorXd &mle_tolerances,
             const std::optional<Eigen::VectorXd> &delta_params = std::nullopt,
             const std::optional<Eigen::MatrixXd> &bounds = std::nullopt);
  Eigen::VectorXd get_params() const override;
  void set_params(const Eigen::VectorXd &params) override;
  Eigen::VectorXd function(const Eigen::VectorXd &x,
                           const std::any &flag = {}) override;
  Eigen::VectorXd normal(const Eigen::VectorXd &x,
                         const std::any &flag = {}) override;
  void validate() const override;

  std::shared_ptr<Mineral> m;
  std::vector<std::string> fit_params;
  std::optional<Eigen::MatrixXd> bounds;
};

/** Fit P-T-V data; omitted covariances give unit variance in pressure only.
 * The returned model and the supplied mineral contain the fitted parameters.
 * Verbose output reports LM iterations; outlier diagnostics are not included.
 */
MineralFit
fit_PTV_data(std::shared_ptr<Mineral> mineral,
             const std::vector<std::string> &fit_params,
             const Eigen::MatrixXd &data,
             const std::vector<Eigen::MatrixXd> &data_covariances = {},
             const std::optional<Eigen::VectorXd> &delta_params = std::nullopt,
             const std::optional<Eigen::MatrixXd> &bounds = std::nullopt,
             const NonlinearFitOptions &options = {0., 1.e-5, 100, std::nullopt,
                                                   std::nullopt, true});
} // namespace optim
} // namespace burnman
