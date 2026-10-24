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
#include <optional>
#include <tuple>
#include <utility>

namespace burnman::optim {
using LinearConstraints = std::pair<Eigen::MatrixXd, Eigen::VectorXd>;
using LeastSquaresResult = std::tuple<Eigen::VectorXd, Eigen::MatrixXd, double>;

/** Minimise (A*x-b)' * pinv(Cov_b) * (A*x-b), subject to C*x=c and D*x<=d.
 *
 * Cov_b defaults to the identity. It must be symmetric and positive
 * semidefinite. Eigenvalues at or below 1e-15 times the largest eigenvalue
 * have zero weight. Constraints are supplied
 * as (C,c) and (D,d). The weighted design must have full column rank unless
 * allow_rank_deficient is true, even when constraints determine a unique fit.
 *
 * Returns the fitted parameters, pinv(A' * pinv(Cov_b) * A), and the weighted
 * sum of squared residuals. The covariance describes the unconstrained
 * weighted design and is not conditioned on the supplied constraints.
 */
LeastSquaresResult weighted_constrained_least_squares(
    const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
    const std::optional<Eigen::MatrixXd> &Cov_b = std::nullopt,
    const std::optional<LinearConstraints> &equality_constraints = std::nullopt,
    const std::optional<LinearConstraints> &inequality_constraints =
        std::nullopt,
    bool allow_rank_deficient = false);
} // namespace burnman::optim
