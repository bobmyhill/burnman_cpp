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

#ifndef BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_LAMBDA_BOUNDS_HPP_INCLUDED
#define BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_LAMBDA_BOUNDS_HPP_INCLUDED

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace burnman {
namespace equilibration {
/**
 * @brief Returns lambda bounds for damped Newton solver.
 *
 * Computes bounds on lambda for the damped affine invariant modification to
 * Newton's method for nonlinear problems (Deuflhard, 1974;1975;2004).
 *
 * @param dx The proposed Newton step.
 * @param x The parameter vector.
 * @returns (min, max)
 */
std::pair<double, double>
lambda_bounds_func(const Eigen::VectorXd &dx, const Eigen::VectorXd &x,
                   const std::vector<int> &endmembers_per_phase);
} // namespace equilibration
} // namespace burnman

#endif // BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_LAMBDA_BOUNDS_HPP_INCLUDED
