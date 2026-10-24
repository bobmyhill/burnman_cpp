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

#include "burnman/core/assemblage.hpp"
#include "burnman/minerals/covariance.hpp"
#include <limits>
#include <optional>
#include <string>

namespace burnman::thermobarometry {

/// Independent reactions in the original assemblage endmember basis.
/// A positive small_fraction_tol selects physical endmembers greedily from
/// the observed site occupancies, omitting contributions below the tolerance.
Eigen::MatrixXd get_reaction_matrix(const Assemblage &assemblage,
                                    double small_fraction_tol = 0.);

/// Set [pressure (Pa), temperature (K), free compositional coordinates...].
/// Solutions with free_compositional_vectors require baseline_composition.
void assemblage_set_state_from_params(Assemblage &assemblage,
                                      const Eigen::VectorXd &params);

/// Covariance of reaction affinities, in (J/mol reaction)^2. Includes
/// compositional_covariances on solutions and optional dataset covariances.
/// State uncertainties additionally require assemblage.state_covariances.
Eigen::MatrixXd assemblage_affinity_covariance_matrix(
    const Assemblage &assemblage, const Eigen::MatrixXd &reaction_matrix,
    const std::optional<minerals::CovarianceData> &dataset_covariances =
        std::nullopt,
    bool include_state_uncertainties = false);

/// Reaction affinities weighted by the pseudoinverse of their covariance.
double assemblage_affinity_misfit(const Assemblage &assemblage,
                                  const Eigen::MatrixXd &reaction_matrix,
                                  const std::optional<minerals::CovarianceData>
                                      &dataset_covariances = std::nullopt,
                                  bool include_state_uncertainties = false);

/// Squared P-T deviation weighted by state_inverse_covariances.
/// Requires state_priors and state_inverse_covariances on the assemblage.
double assemblage_state_misfit(const Assemblage &assemblage);

/** Estimated pressure, temperature and free compositional coordinates.
 * x is in [Pa, K, molar-fraction coordinates...]. xcov, var and xcorr describe
 * P and T only; dadx contains the reaction derivatives with respect to P and T
 * at the optimized compositions. Covariances of the free compositional
 * coordinates are not estimated. reduced_chisqr and fit are NaN when the
 * degrees of freedom are nonpositive.
 */
struct EstimateConditionsResult {
  Eigen::VectorXd x, jac, affinities, weighted_affinities;
  Eigen::MatrixXd xcov, xcorr, acov, dadx;
  Eigen::VectorXd var;
  double fun = 0.;
  double reduced_chisqr = std::numeric_limits<double>::quiet_NaN();
  double fit = std::numeric_limits<double>::quiet_NaN();
  int n_reactions = 0, n_constraints = 0, n_params = 0, degrees_of_freedom = 0;
  int status = 0, nfev = 0;
  bool success = false;
  std::string message;
};

/** Estimate equilibration conditions by minimizing weighted reaction
 * affinities. Each solution requires compositional_covariances. Optional rows
 * of free_compositional_vectors vary its composition relative to the
 * composition at the start of the fit, stored in baseline_composition. The
 * assemblage is left at the returned conditions. Prior misfit requires
 * state_priors and state_inverse_covariances. A missing guess uses the
 * assemblage's state. Equal bounds fix the corresponding condition. P_scaling
 * is in Pa.
 */
EstimateConditionsResult estimate_conditions(
    Assemblage &assemblage,
    const std::optional<minerals::CovarianceData> &dataset_covariances =
        std::nullopt,
    bool include_state_misfit = false,
    const std::optional<Eigen::Vector2d> &guessed_conditions =
        Eigen::Vector2d(1.e9, 1000.),
    const Eigen::Vector2d &pressure_bounds = Eigen::Vector2d(1.e5, 400.e9),
    const Eigen::Vector2d &temperature_bounds = Eigen::Vector2d(300., 4000.),
    double P_scaling = 1.e6, double small_fraction_tol = 0., int max_it = 100);
} // namespace burnman::thermobarometry
