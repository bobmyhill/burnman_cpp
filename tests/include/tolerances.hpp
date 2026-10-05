/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
#ifndef TESTS_TOLERANCES_HPP
#define TESTS_TOLERANCES_HPP

#include <cmath>
#include <limits>

constexpr double tol_abs = 1e-16;
constexpr double tol_rel = 1e-12;

// Cancellation leaves an absolute error set by the operands, not the result.
inline double roundoff_tolerance(double operand_scale) {
  return 8.0 * std::numeric_limits<double>::epsilon() * std::abs(operand_scale);
}

#endif // TESTS_TOLERANCES_HPP
