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
#include "burnman/tools/pseudosection/common.hpp"
#include <utility>

namespace burnman::pseudosections::detail {
struct Engine;
// Weighted sum of two equality constraints. Keep weights constant while solving
// so the combined value and its derivatives remain consistent.
class SectionConstraint : public EqualityConstraint {
  std::unique_ptr<EqualityConstraint> first_, second_;
  Eigen::Vector2d weights_;

public:
  SectionConstraint(std::unique_ptr<EqualityConstraint> first,
                    std::unique_ptr<EqualityConstraint> second,
                    const Eigen::Vector2d &weights)
      : first_(std::move(first)), second_(std::move(second)),
        weights_(weights) {}
  std::unique_ptr<EqualityConstraint> clone() const override {
    return std::make_unique<SectionConstraint>(first_->clone(),
                                               second_->clone(), weights_);
  }
  double evaluate(const Eigen::VectorXd &x,
                  const Assemblage &a) const override {
    return weights_[0] * first_->evaluate(x, a) +
           weights_[1] * second_->evaluate(x, a);
  }
  Eigen::VectorXd derivative(const Eigen::VectorXd &x, const Assemblage &a,
                             Eigen::Index n) const override {
    return weights_[0] * first_->derivative(x, a, n) +
           weights_[1] * second_->derivative(x, a, n);
  }
};

/// Match phase instances by mineral model/composition, then copy the reference
/// compositions into a trial state. Reject incompatible present-phase pairs
/// beyond tolerance; amounts and local instance IDs are retained.
bool seed_phase_compositions(
    PhaseStates &, const PhaseStates &,
    double tolerance = std::numeric_limits<double>::infinity(),
    double amount_tolerance = 0.);

/// Remove a verified overshoot beyond a junction. Re-solve on the earlier
/// reaction branch before trimming; curved lines need not be monotonic in P/T.
void trim_boundary_endpoints(Boundary &, Engine &,
                             const Eigen::Vector2d &range);

/// Compare compositions while allowing solution-copy renumbering and extra
/// zero-amount phases. A present unmatched phase makes the states different.
bool same_phase_compositions(const PhaseStates &, const PhaseStates &,
                             double tolerance, double amount_tolerance);

/// Compare reaction states, also requiring the zero phases to match in model
/// and composition. Equal present assemblages alone cannot identify a reaction.
bool same_boundary_compositions(const PhaseStates &, int, const PhaseStates &,
                                int, double, double);

/// Map a junction's zero-phase choices onto an incident equilibrium's local
/// copy IDs, selecting the smallest-amount unused copy of each candidate.
std::vector<int> zero_phases_at(const Node &, const PhaseStates &, int copies);

/// Re-equilibrate a trial assemblage under supplied constraints and require
/// stability, mass balance, distinct solution copies and proximity to target.
/// target/allowance use native diagram units. Returns no point on failure;
/// the trial assemblage can still have changed during the attempted solve.
std::optional<BoundaryPoint> corrected_point(Engine &, Assemblage &,
                                             const std::vector<int> &,
                                             ConstraintList &,
                                             const Eigen::Vector2d &target,
                                             const Eigen::Vector2d &allowance);

/// Solve the same reaction near target, using warm phase amounts/compositions.
/// Impose zero amount of line.zero_phase and a weighted plane through target.
/// This follows an existing branch; it does not discover a new phase
/// assemblage.
std::optional<BoundaryPoint>
boundary_point_at(Engine &, const Boundary &, const PhaseStates &warm,
                  const Eigen::Vector2d &target, const Eigen::Vector2d &weights,
                  const Eigen::Vector2d &allowance);

/// Verify a saved double-zero junction using compositions from a reference
/// equilibrium. Preserve the junction's own phase IDs and zero constraints.
std::optional<BoundaryPoint>
junction_point_at(Engine &, const Node &, PhaseStates warm,
                  const PhaseStates &seed, const Eigen::Vector2d &allowance);

/// Test whether two records follow the same equilibrium reaction curve.
/// partial asks whether the first is covered by the second. Near geometric
/// agreement can suffice for ordinary complete arcs; require_equilibrium forces
/// corrector checks, and univariant reactions always need thermodynamic checks.
/// The caller separately establishes compatible reaction IDs/endpoints.
bool same_boundary_curve(Engine &, const Result &, const Boundary &,
                         const Boundary &, bool partial = false,
                         bool require_equilibrium = false);

/// Tangent to an equilibrium reaction curve, from the solve Jacobian's
/// nullspace. Release the moving-plane constraint while retaining reaction and
/// bulk balance. The returned parameter change moves one unit in the rescaled
/// diagram. coordinate_rank counts independent projected directions: zero
/// describes an event with no curve direction; more than one does not define a
/// unique line. Return an empty vector unless exactly one diagram direction is
/// available.
Eigen::VectorXd continuation_tangent(const Engine &, const EquilibriumSolve &,
                                     const Assemblage &,
                                     const Eigen::Vector2d &,
                                     int *coordinate_rank = nullptr);

/// Identify the right/left assemblages of a reaction in calculation-axis order.
/// For a regular phase-out boundary, release the zero-amount condition and
/// calculate which side gives a positive phase amount. For a compound P-T
/// reaction, the sign of dG = dV*dP - dS*dT orders its limiting assemblages.
/// An X section can return just the full-assemblage side when the other needs
/// a separate probe. Optional limiting_states provide contour starting states.
/// Uses derivatives at the boundary, without nonlinear equilibrium solves.
std::array<std::vector<int>, 2>
boundary_sides(Engine &, Assemblage &, const std::vector<int> &, int,
               const EquilibriumSolve &, const Eigen::Vector2d &range,
               const Eigen::Vector2d &normal,
               std::array<State, 2> *limiting_states = nullptr);

/// Moving correction plane through target in the selected diagram coordinates.
/// weights fix its normal, giving one location condition while permitting
/// motion along the reaction curve during the equilibrium solve.
std::unique_ptr<EqualityConstraint>
continuation_plane(const Engine &, const Assemblage &, Eigen::Index,
                   const Eigen::Vector2d &target,
                   const Eigen::Vector2d &weights);
} // namespace burnman::pseudosections::detail
