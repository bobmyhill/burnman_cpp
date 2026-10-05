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

#ifndef BURNMAN_TOOLS_EQUILIBRATION_EQUALITY_CONSTRAINT_BASE_HPP_INCLUDED
#define BURNMAN_TOOLS_EQUILIBRATION_EQUALITY_CONSTRAINT_BASE_HPP_INCLUDED

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace burnman {

// Forward declaration
class Assemblage;

namespace equilibration {

/**
 * @brief Base class for linear equality constraints.
 *
 * Use `make_constraint<ConstraintType>()' to construct a constraint.
 * Currently implemented constraints are:
 *   PressureConstraint
 *   TemperatureConstraint
 *   EntropyConstraint
 *   VolumeConstraint
 *   PTEllipseConstraint
 *   LinearXConstraint
 *   PhaseFractionConstraint
 *   PhaseCompositionConstraint
 *
 * Use `EqualityConstraint::evaluate()' to compute F.
 * Use `EqualityConstraint::derivative()' to compute J.
 *
 * @note Equality constraints should implement evaluate() and
 * derivative() functions, as well as a clone() method.
 */
class EqualityConstraint {
public:
  virtual ~EqualityConstraint() = default;
  virtual std::unique_ptr<EqualityConstraint> clone() const = 0;
  virtual double evaluate(const Eigen::VectorXd &x,
                          const Assemblage &assemblage) const = 0;
  virtual Eigen::VectorXd derivative(const Eigen::VectorXd &x,
                                     const Assemblage &assemblage,
                                     Eigen::Index J_size) const = 0;
};

/**
 * @brief Grouped top-level constraints.
 *
 * Use ConstraintGroup to keep expanded constraint vectors together.
 * Single constraints should also be put in a ConstraintGroup for
 * normalisation.
 *
 * @see `make_constraints_from_array' and `wrap_constraint'.
 */
using ConstraintGroup = std::vector<std::unique_ptr<EqualityConstraint>>;

/**
 * @brief Nested list of ConstraintGroups for equilibrate function.
 *
 * A ConstraintList should be constructed to pass to the equilibrate
 * function. Each top level ConstraintGroup can contain 1 or many
 * constraints. The equilibration routine will loop over all possible
 * lists of top level constraints constructed from the sub-constraints
 * in each group.
 *
 * @see `make_constraint_list'.
 */
using ConstraintList = std::vector<ConstraintGroup>;

} // namespace equilibration
} // namespace burnman

#endif // BURNMAN_TOOLS_EQUILIBRATION_EQUALITY_CONSTRAINT_BASE_HPP_INCLUDED
