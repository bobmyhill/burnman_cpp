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
#include <variant>
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

/// Volume-coordinate counterpart of MineralFit; shares parameter handling.
class MineralFitV : public MineralFit {
public:
  using MineralFit::MineralFit;
  Eigen::VectorXd function(const Eigen::VectorXd &x,
                           const std::any &flag = {}) override;
  Eigen::VectorXd normal(const Eigen::VectorXd &x,
                         const std::any &flag = {}) override;
  void validate() const override;
};

using PropertyFlags = std::variant<std::string, std::vector<std::string>>;

/** Fit P-T-property data with a shared flag or one flag per observation.
 * Empty mle_tolerances select the property defaults. Omitted covariances give
 * unit pressure variance with exact temperature and property values.
 * Updates the supplied mineral and returns the fitted model. Verbose output
 * reports LM iterations; outlier diagnostics are not included.
 */
MineralFit
fit_PTp_data(std::shared_ptr<Mineral> mineral,
             const std::vector<std::string> &fit_params,
             const PropertyFlags &flags, const Eigen::MatrixXd &data,
             const std::vector<Eigen::MatrixXd> &data_covariances = {},
             const Eigen::VectorXd &mle_tolerances = {},
             const std::optional<Eigen::VectorXd> &delta_params = std::nullopt,
             const std::optional<Eigen::MatrixXd> &bounds = std::nullopt,
             const NonlinearFitOptions &options = {0., 1.e-5, 100, std::nullopt,
                                                   std::nullopt, true});

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

/** Fit V-T-property observations, using unit property variance by default.
 * Flags, tolerances, priors and bounds follow fit_PTp_data().
 */
MineralFitV
fit_VTp_data(std::shared_ptr<Mineral> mineral,
             const std::vector<std::string> &fit_params,
             const PropertyFlags &flags, const Eigen::MatrixXd &data,
             const std::vector<Eigen::MatrixXd> &data_covariances = {},
             const Eigen::VectorXd &mle_tolerances = {},
             const std::optional<Eigen::VectorXd> &delta_params = std::nullopt,
             const std::optional<Eigen::MatrixXd> &bounds = std::nullopt,
             const NonlinearFitOptions &options = {0., 1.e-5, 100, std::nullopt,
                                                   std::nullopt, true});

/// fit_VTp_data() with pressure flags; data rows are V (m3/mol), T (K), P (Pa).
MineralFitV
fit_VTP_data(std::shared_ptr<Mineral> mineral,
             const std::vector<std::string> &fit_params,
             const Eigen::MatrixXd &data,
             const std::vector<Eigen::MatrixXd> &data_covariances = {},
             const std::optional<Eigen::VectorXd> &delta_params = std::nullopt,
             const std::optional<Eigen::MatrixXd> &bounds = std::nullopt,
             const NonlinearFitOptions &options = {0., 1.e-5, 100, std::nullopt,
                                                   std::nullopt, true});
} // namespace optim
} // namespace burnman
