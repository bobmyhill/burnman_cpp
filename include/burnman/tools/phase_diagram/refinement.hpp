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
#include "burnman/tools/phase_diagram/common.hpp"
#include <functional>

namespace burnman::phase_diagrams {
/// Maximum spacing between solved points along existing reaction boundaries.
/// These counts do not introduce a new grid of bulk equilibrium calculations.
struct LineResolution {
  // Axis point counts in calculation order: 101 means 100 equal divisions.
  std::array<int, 2> axis_points;
  // Divide the density axis uniformly instead of the volume axis.
  bool reciprocal_volume = false;
  // Optional window in native diagram-axis units. Refine intersecting segments
  // to subdivisions of this window, retaining the complete diagram graph.
  std::optional<std::array<std::array<double, 2>, 2>> region = std::nullopt;
};

} // namespace burnman::phase_diagrams

namespace burnman::phase_diagrams::detail {
struct Engine;
/// Subdivide boundary segments with equilibrium corrector solves until their
/// coordinate spacing meets resolution. Return messages for unmet spacing;
/// branch discovery/recovery remains the tracing calculation's responsibility.
std::vector<std::string> refine_line_spacing(
    Result &, Engine &, const LineResolution &,
    const std::function<bool(Boundary &, std::size_t)> &refine_segment);

/// Reject invalid point counts, incompatible density axes or out-of-range
/// windows.
void validate_line_resolution(const Result &, const LineResolution &);

/// Local continuation tolerances/steps scaled to a requested refinement window.
Settings refinement_settings(const Result &, const LineResolution &,
                             const Settings &);

/// Choose the ordinary or locally reduced trace step for a proposed segment.
double refinement_step(const Result &, const LineResolution &,
                       const Eigen::Vector2d &, const Eigen::Vector2d &,
                       const Settings &);

/// Reaction IDs near provisional fields that intersect the refinement window.
/// Include incident branches because the missing connection can start at a
/// node.
std::vector<int> regional_retrace_lines(const Result &, const LineResolution &);

/// Find drawn reaction segments crossing a polygon's interior. Such crossings
/// nominate lines for thermodynamic checks; they do not create junctions.
std::vector<int> interior_boundaries(const Result &, const FieldPolygon &);
} // namespace burnman::phase_diagrams::detail
