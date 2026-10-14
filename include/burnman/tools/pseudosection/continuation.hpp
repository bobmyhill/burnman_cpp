// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

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

bool seed_phase_compositions(
    PhaseStates &, const PhaseStates &,
    double tolerance = std::numeric_limits<double>::infinity(),
    double amount_tolerance = 0.);

void trim_boundary_endpoints(Boundary &, Engine &,
                             const Eigen::Vector2d &range);

bool same_phase_compositions(const PhaseStates &, const PhaseStates &,
                             double tolerance, double amount_tolerance);

// Compare intensive branches while retaining the identity of the zero phase.
bool same_boundary_compositions(const PhaseStates &, int, const PhaseStates &,
                                int, double, double);

// Node instance numbers may differ from those in an incident equilibrium.
std::vector<int> zero_phases_at(const Node &, const PhaseStates &, int copies);

std::optional<BoundaryPoint> corrected_point(Engine &, Assemblage &,
                                             const std::vector<int> &,
                                             ConstraintList &,
                                             const Eigen::Vector2d &target,
                                             const Eigen::Vector2d &allowance);

std::optional<BoundaryPoint>
boundary_point_at(Engine &, const Boundary &, const PhaseStates &warm,
                  const Eigen::Vector2d &target, const Eigen::Vector2d &weights,
                  const Eigen::Vector2d &allowance);

std::optional<BoundaryPoint>
junction_point_at(Engine &, const Node &, PhaseStates warm,
                  const PhaseStates &seed, const Eigen::Vector2d &allowance);

bool same_boundary_curve(Engine &, const Result &, const Boundary &,
                         const Boundary &, bool partial = false,
                         bool require_equilibrium = false);

// Find changes in the equilibrium variables that move along a line in the
// diagram. The returned direction moves one unit in the rescaled diagram.
// coordinate_rank counts independent ways to move in the diagram; return an
// empty vector unless there is exactly one, after releasing constraint row 1.
Eigen::VectorXd continuation_tangent(const Engine &, const EquilibriumSolve &,
                                     const Assemblage &,
                                     const Eigen::Vector2d &,
                                     int *coordinate_rank = nullptr);

// Right/left neighbouring assemblages from a regular field response or a
// single P-T reaction's entropy/volume slope. A composition section can return
// only its full branch; the opposite side then needs a local probe. Releases
// the phase-amount constraint and performs no nonlinear equilibrium solves.
std::array<std::vector<int>, 2>
boundary_sides(Engine &, Assemblage &, const std::vector<int> &, int,
               const EquilibriumSolve &, const Eigen::Vector2d &range,
               const Eigen::Vector2d &normal,
               std::array<State, 2> *limiting_states = nullptr);

std::unique_ptr<EqualityConstraint>
continuation_plane(const Engine &, const Assemblage &, Eigen::Index,
                   const Eigen::Vector2d &target,
                   const Eigen::Vector2d &weights);
} // namespace burnman::pseudosections::detail
