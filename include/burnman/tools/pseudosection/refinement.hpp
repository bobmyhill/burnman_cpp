// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#pragma once
#include "burnman/tools/pseudosection/common.hpp"
#include <functional>

namespace burnman::pseudosections {
struct LineResolution {
  // Axis point counts in selected diagram-axis order: 101 means 100 divisions.
  std::array<int, 2> axis_points;
  // Divide the density axis uniformly instead of the volume axis.
  bool reciprocal_volume = false;
  // Optional window in native diagram-axis units. Refine intersecting segments
  // to subdivisions of this window, retaining the complete diagram graph.
  std::optional<std::array<std::array<double, 2>, 2>> region = std::nullopt;
};

} // namespace burnman::pseudosections

namespace burnman::pseudosections::detail {
struct Engine;
// Insert verified corrector points until each selected axis meets its spacing.
// Failed segments remain explicit diagnostics; the caller owns branch recovery.
std::vector<std::string> refine_line_spacing(
    Result &, Engine &, const LineResolution &,
    const std::function<bool(Boundary &, std::size_t)> &refine_segment);

void validate_line_resolution(const Result &, const LineResolution &);

Settings refinement_settings(const Result &, const LineResolution &,
                             const Settings &);

double refinement_step(const Result &, const LineResolution &,
                       const Eigen::Vector2d &, const Eigen::Vector2d &,
                       const Settings &);

std::vector<int> regional_retrace_lines(const Result &, const LineResolution &);

// Chord intersections are search candidates, never new thermodynamic nodes.
std::vector<int> interior_boundaries(const Result &, const FieldPolygon &);
} // namespace burnman::pseudosections::detail
