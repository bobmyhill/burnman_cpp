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
/// Total G(P,T), H(P,S), F(T,V) or U(S,V), on the supplied bulk scale.
/// Return NaN for non-natural P-V/T-S sections. X is held fixed when comparing
/// states along the other axis; compositions with different X are not compared.
double thermodynamic_potential(const State &, const CompositionSection &);

/// Energy uncertainty from roundoff, measured reaction and mass residuals.
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
