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

#include "burnman/optim/linear_fitting.hpp"

namespace burnman::optim {
/** Fit nonnegative phase amounts to a bulk composition by least squares.
 *
 * phase_compositions has one row per chemical component and one column per
 * phase. bulk_composition uses the same component ordering, units and
 * compositional basis. Mass or molar compositions may be used.
 *
 * Returns the fitted phase amounts, pinv(A' * A), and the sum of squared
 * compositional residuals, where A is phase_compositions. Observations have
 * unit covariance. The amounts are not constrained to sum to one, and the
 * covariance is not conditioned on the nonnegativity constraints. The phase
 * composition matrix must have full column rank.
 */
LeastSquaresResult fit_phase_proportions_to_bulk_composition(
    const Eigen::MatrixXd &phase_compositions,
    const Eigen::VectorXd &bulk_composition);
} // namespace burnman::optim
