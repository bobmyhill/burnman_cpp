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
#include "burnman/tools/pseudosection/refinement.hpp"

namespace burnman::pseudosections {
/// Calculate phase fields for a closed elemental bulk. Find stable starting
/// assemblages by minimising Gibbs energy, then trace boundaries where a phase
/// amount reaches zero. At junctions, try branches with one phase added,
/// removed or exchanged. Accept branches only where equilibrium defines a line
/// in the selected diagram and no lower-Gibbs-energy alternative is found.
/// Finite seed sampling does not guarantee discovery of every disconnected
/// field. Inspect resolved and diagnostics; increase seed density to assess it.
/// section selects any pair of P,T,S,V,X and supplies S/V/X ranges. X diagrams
/// fix one other thermodynamic coordinate. S and V are extensive on the bulk
/// amount scale. Missing P/T are solved using the supplied P/T ranges as search
/// bounds. PX/TX can specify fixed T/P by giving that range equal endpoints.
Result pseudosection(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const std::array<double, 2> &pressure_range,
                     const std::array<double, 2> &temperature_range,
                     const Settings &settings = Settings{},
                     const CompositionSection &section = CompositionSection{});

/// Resume unfinished boundaries from their last verified compositions and
/// revisit junction branches. Supply the same bulk, candidates and phase-ID
/// settings as the original calculation, including candidate order. Recovery
/// may refine straight segments, trim tails or merge duplicate lines.
/// resolution also subdivides every boundary using verified equilibrium solves,
/// limiting point spacing to the axis span divided by (axis_points - 1).
/// With resolution.region, use the window span and refine only segments that
/// intersect it. Retrace unstable arcs and arcs around provisional loops once,
/// using smaller steps within the window when max_recovery_passes is nonzero.
/// Spacing that cannot be reached is reported in the returned diagnostics.
/// Added points per line are limited by
/// max_trace_steps*max_refinement_iterations.
Result refine_pseudosection(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous, const Settings &settings,
    const std::optional<LineResolution> &resolution = std::nullopt);

/// Resume with the settings retained in the previous result.
Result refine_pseudosection(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous,
    const std::optional<LineResolution> &resolution = std::nullopt);

} // namespace burnman::pseudosections
