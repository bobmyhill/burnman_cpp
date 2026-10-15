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
#include "burnman/tools/phase_diagram/refinement.hpp"

namespace burnman::phase_diagrams {
/// Calculate an assemblage diagram for a closed bulk and supplied phase models.
/// composition gives elemental mole amounts; candidates are the allowed mineral
/// and solution models. Initial equilibrium calculations locate changes in
/// stable assemblage. Predictor-corrector continuation then follows the
/// corresponding phase boundaries, usually by imposing zero amount of an
/// appearing phase. Junction calculations test further branches, including
/// solvus reactions. Accepted states satisfy mass balance, reaction equilibrium
/// and candidate affinity checks within settings tolerances. A converged Newton
/// root alone does not establish stability against the other candidate phases.
///
/// section selects P,T,S,V or X axes. Native units are Pa, K, total J/K and
/// total m^3; S and V scale with the supplied bulk amount. Along an X axis,
/// bulk(X) = (1-X)*composition + X*section.composition_end, without automatic
/// normalisation, and one other thermodynamic coordinate is fixed. Unplotted
/// P/T ranges guide inverse seed searches. Legacy PX/TX sections can specify
/// fixed T/P by giving the corresponding range equal endpoints.
///
/// The Result contains equilibrium samples, traced boundaries and junctions.
/// field_polygons() reconstructs the closed fields for plotting. resolved means
/// that the implemented checks have no outstanding diagnostics; finite sampling
/// cannot prove that every disconnected stability field has been discovered.
Result phase_diagram(const types::FormulaMap &composition,
                     const std::vector<std::shared_ptr<Material>> &candidates,
                     const std::array<double, 2> &pressure_range,
                     const std::array<double, 2> &temperature_range,
                     const Settings &settings = Settings{},
                     const CompositionSection &section = CompositionSection{});

/// Continue and refine a saved calculation, returning a new Result.
/// Reuse the original bulk, candidate order and max_phase_instances so saved
/// phase IDs still identify the same minerals and coexisting solution copies.
/// A resolved result retains all connections, labels, nodes and original
/// points; only validated boundary points are added. An unresolved result also
/// runs recovery from accepted states, revisiting junction branches where
/// needed, trimming verified overshoots and removing duplicate traces.
///
/// Optional resolution controls the spacing of solved boundary points, rather
/// than the density of initial assemblage samples: 101 axis points means 100
/// divisions of that axis. With resolution.region, subdivide the window span
/// and refine segments intersecting it, retaining the whole diagram. Recovery
/// can retrace suspect arcs; when resolution is supplied, additional
/// equilibrium samples test closed fields that still lack a trusted label.
///
/// Every inserted point is an equilibrium calculation, not a linear estimate
/// of mineral compositions. Unmet spacing is reported in spacing_diagnostics,
/// separately from the structural diagnostics that determine resolved.
/// Added points per line are limited by
/// max_trace_steps*max_refinement_iterations.
Result refine_phase_diagram(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous, const Settings &settings,
    const std::optional<LineResolution> &resolution = std::nullopt);

/// Resume with the settings retained in the previous result.
Result refine_phase_diagram(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Result &previous,
    const std::optional<LineResolution> &resolution = std::nullopt);

} // namespace burnman::phase_diagrams
