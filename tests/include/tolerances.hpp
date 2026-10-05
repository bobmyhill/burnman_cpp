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
