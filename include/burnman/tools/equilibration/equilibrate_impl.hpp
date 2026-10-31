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

#ifndef BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_IMPL_HPP_INCLUDED
#define BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_IMPL_HPP_INCLUDED

#include "burnman/core/assemblage.hpp"
#include "burnman/tools/equilibration/equality_constraint_base.hpp"
#include "burnman/tools/equilibration/equilibrate_types.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include <Eigen/Dense>
#include <limits>
#include <optional>
#include <vector>

namespace burnman {
namespace equilibration {

/**
 * @brief Finds equilibrium state of assemblage subject to constraints.
 *
 * This finds the thermodynamic equilibrium state of an assemblage subject to
 * given equality constraints by solving a set of nonlinear equations
 * related to the chemical potentials and other state variables of the system.
 *
 * Usage requires an assemblage and 2 + n_c equality constraints, where n_c is
 * the number of bulk compositional degrees of freedom. The equilibrate
 * function attempts to find the remaining unknowns that satisfy those
 * constraints.
 *
 * See `EqualityConstraint' for information on constraints TODO: note namespace
 * here Possible constraints are: PressureConstraint TemperatureConstraint
 *   EntropyConstraint
 *   VolumeConstraint
 *   PTEllipseConstraint
 *   PhaseFractionConstraint
 *   PhaseCompositionConstraint
 *   LinearXConstraint
 *
 * @param composition Bulk composition.
 * @param assemblage Target assemblage to equilibrate.
 * @param equality_constraints Vector of top level ConstraintGroups.
 * @param free_compositional_vectors Optional element maps for bulk
 * compositional degrees of freedom.
 * @param tol Absolute Newton-step tolerance. NaN selects physical defaults:
 * 1 Pa, 1e-6 K and 1e-9 for amounts, compositions and free coordinates.
 * @param store_iterates Store parameter vector and function value convergence
 * history.
 * @param store_assemblage Store assemblage object copies in result.
 * @param max_iterations Maximum allowed iterations.
 * @param verbose Print extra iteration info.
 * @param parameter_tolerances Optional per-parameter step tolerances,
 * overriding tol.
 * @param parameter_scales Optional positive physical units per solver
 * parameter (defaults: 1e9 Pa, 1e3 K, otherwise 1). Internally equilibrates
 * Jacobian rows; results remain in physical units. An explicitly empty vector
 * retains unscaled continuation for existing callers.
 * @param stagnation_iterations Consecutive iterations without meaningful
 * progress before returning failure. Zero disables this optional check.
 * @return EquilibrateResult with solution and parameters.
 */
EquilibrateResult equilibrate(
    const types::FormulaMap &composition, Assemblage &assemblage,
    const ConstraintList &equality_constraints,
    const std::vector<FreeVectorMap> &free_compositional_vectors = {},
    double tol = std::numeric_limits<double>::quiet_NaN(),
    bool store_iterates = false, bool store_assemblage = true,
    int max_iterations = 100, bool verbose = false,
    const Eigen::VectorXd &parameter_tolerances = Eigen::VectorXd(),
    const std::optional<Eigen::VectorXd> &parameter_scales = std::nullopt,
    int stagnation_iterations = 0);

} // namespace equilibration
} // namespace burnman

#endif // BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_IMPL_HPP_INCLUDED
