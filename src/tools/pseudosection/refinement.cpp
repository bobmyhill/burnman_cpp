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

#include "internal.hpp"
#include <chrono>
#include <iostream>

namespace burnman::pseudosections::detail {
std::vector<std::string> refine_line_spacing(
    Result &result, Engine &engine, const LineResolution &resolution,
    const std::function<bool(Boundary &, std::size_t)> &refine_segment) {
  std::vector<std::string> issues;
  const auto axes = diagram_axes(result.section.type);
  auto measured = [&](Eigen::Vector2d q) {
    if (resolution.reciprocal_volume)
      for (int axis = 0; axis < 2; ++axis)
        if (axes[static_cast<std::size_t>(axis)] == Coordinate::V)
          q[axis] = 1. / q[axis];
    return q;
  };
  const auto ranges = result.coordinate_ranges();
  Eigen::Vector2d spacing = (measured({ranges[0][1], ranges[1][1]}) -
                             measured({ranges[0][0], ranges[1][0]}))
                                .cwiseAbs();
  for (int axis = 0; axis < 2; ++axis)
    spacing[axis] /= resolution.axis_points[static_cast<std::size_t>(axis)] - 1;
  const double limit = double(engine.settings.max_trace_steps) *
                       engine.settings.max_refinement_iterations;
  if (engine.settings.verbose)
    std::cerr << "Refining " << result.boundaries.size()
              << " phase lines to the requested resolution.\n";
  std::size_t completed = 0;
  for (auto &line : result.boundaries) {
    std::size_t added = 0, failed = 0;
    auto last_progress = std::chrono::steady_clock::now();
    auto progress = [&]() {
      std::cerr << "Line " << completed + 1 << '/' << result.boundaries.size()
                << " (boundary " << line.id << "): " << added
                << " points added, " << failed << " unresolved segments; "
                << engine.equilibrium_solves
                << " equilibrium solves in this refinement.\n";
      last_progress = std::chrono::steady_clock::now();
    };
    for (std::size_t k = 1; k < line.points.size();) {
      const auto delta =
          (measured(diagram_coordinates(line.points[k], result.section.type)) -
           measured(
               diagram_coordinates(line.points[k - 1], result.section.type)))
              .cwiseAbs()
              .eval();
      if ((delta.array() <= spacing.array() * (1. + 1.e-12)).all()) {
        ++k;
      } else if (double(added) < limit && refine_segment(line, k)) {
        ++added;
        // Check both new segments; a curved midpoint need not halve each
        // axis.
      } else {
        ++failed;
        ++k;
      }
      if (engine.settings.verbose &&
          std::chrono::steady_clock::now() - last_progress >=
              std::chrono::seconds(5))
        progress();
    }
    if (engine.settings.verbose)
      progress();
    ++completed;
    if (failed)
      issues.push_back("Boundary " + std::to_string(line.id) +
                       ": requested line resolution was not reached on " +
                       std::to_string(failed) + " segments" +
                       (double(added) >= limit
                            ? " (refinement limit reached)."
                            : " (equilibrium refinement failed)."));
  }
  return issues;
}
} // namespace burnman::pseudosections::detail
