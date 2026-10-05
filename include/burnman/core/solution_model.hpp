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

#ifndef BURNMAN_CORE_SOLUTION_MODEL_HPP_INCLUDED
#define BURNMAN_CORE_SOLUTION_MODEL_HPP_INCLUDED

// Utility header to include all solution models
#include "burnman/core/solution_models/asymmetric_regular_solution.hpp"
#include "burnman/core/solution_models/ideal_solution.hpp"
#include "burnman/core/solution_models/solution_model_base.hpp"
#include "burnman/core/solution_models/symmetric_regular_solution.hpp"

namespace burnman {
/**
 * @namespace burnman::solution_models
 * @brief Contains classes and functions for solid solutions models.
 *
 * This namespace includes base and derived solution model classes.
 * All `burnman::Solution' mineral objects use a solution model to
 * define how the endmembers interact.
 * These classes handle chemical formulae, sites and site occupancies,
 * as well as excess thermodynamic properties.
 *
 * Currently implemented solution models are:
 * - IdealSolution
 * - AsymmetricRegularSolution
 * - SymmetricRegularSolution
 */
namespace solution_models {
// blank for documentation only
} // namespace solution_models
} // namespace burnman

#endif // BURNMAN_CORE_SOLUTION_MODEL_HPP_INCLUDED
