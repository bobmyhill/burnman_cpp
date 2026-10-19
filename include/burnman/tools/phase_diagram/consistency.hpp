// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#pragma once
#include "burnman/tools/phase_diagram/stability.hpp"
#include <functional>

namespace burnman::phase_diagrams::detail {
/// Compact thermodynamic evidence for auditing without retaining phase models.
struct ThermodynamicSample {
  Eigen::Vector2d position = Eigen::Vector2d::Zero(),
                  gradient = Eigen::Vector2d::Zero();
  double energy = 0., uncertainty = 0., bulk_coordinate = 0.;
  double amount = 0., minimum_affinity = 0.;
  bool available = false;
  std::array<bool, 2> applicable{false, false}, convex{false, false};
};
ThermodynamicSample thermodynamic_sample(const State &,
                                         const CompositionSection &);
/// Check one final pair and accumulate explicit coverage; performs no solves.
double audit_thermodynamic_pair(const ThermodynamicSample &,
                                const ThermodynamicSample &, std::size_t axis,
                                ThermodynamicValidation &);
/// Audit all aligned saved samples, including later additions and resumed data.
/// Preserve non-thermodynamic diagnostics; never retrace or solve new states.
void audit_thermodynamic_samples(Result &);

/// Total G(P,T), H(P,S), F(T,V) or U(S,V), on the supplied bulk scale.
/// Return NaN for non-natural P-V/T-S sections. X is held fixed when comparing
/// states along the other axis; compositions with different X are not compared.
double thermodynamic_potential(const State &, const CompositionSection &);

/// Energy uncertainty from roundoff, reaction/mass residuals and candidate
/// affinities, accounting for different mole amounts across a reaction.
double thermodynamic_energy_tolerance(const State &, const State &);

/// Excess beyond the tangent/secant bounds along one natural coordinate.
/// Zero means consistent within numerical uncertainty, or inapplicable.
/// The other coordinate and elemental bulk must be fixed. Convex S/V and
/// concave P/T bounds permit phase transitions, latent mixtures and flat modes.
double thermodynamic_violation(const State &, const State &,
                               const CompositionSection &, std::size_t axis);

/// A portable equilibrium snapshot, including extensive energy for comparisons.
State equilibrium_state(Engine &, const Assemblage &, const std::vector<int> &);
State equilibrium_state(const BoundaryPoint &, double gibbs);

/// Revisit a flagged point using independent discovery and neighbouring phase
/// compositions. Compare validated roots at identical coordinates, including
/// different composition basins of the same named assemblage. No tracing here.
WorkState recover_thermodynamic_state(Engine &, WorkState,
                                      const std::vector<State> &neighbours);

/// Check a rectangular seed grid before reaction bracketing. The inner index
/// varies the second diagram coordinate. Update only flagged states, notifying
/// the caller so its saved samples and assemblage memberships stay current.
/// Return outstanding violations and log the added solves/time when verbose.
std::vector<std::string>
recover_thermodynamic_grid(Engine &, std::vector<WorkState> &, int columns,
                           const std::function<void(std::size_t)> &updated);
} // namespace burnman::phase_diagrams::detail
