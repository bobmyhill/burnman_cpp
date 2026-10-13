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
namespace {
bool intersects_region(const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                       const std::array<std::array<double, 2>, 2> &ranges) {
  double first = 0., last = 1.;
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    const auto &bounds = ranges[static_cast<std::size_t>(axis)];
    const double delta = b[axis] - a[axis];
    if (delta == 0.) {
      if (a[axis] < bounds[0] || a[axis] > bounds[1])
        return false;
    } else {
      double lo = (bounds[0] - a[axis]) / delta,
             hi = (bounds[1] - a[axis]) / delta;
      first = std::max(first, std::min(lo, hi));
      last = std::min(last, std::max(lo, hi));
    }
  }
  return first <= last;
}
} // namespace

void validate_line_resolution(const Result &result,
                              const LineResolution &resolution) {
  for (int count : resolution.axis_points)
    if (count < 2)
      throw std::invalid_argument(
          "Line resolution must contain axis point counts of at least 2.");
  const auto axes = diagram_axes(result.section.type);
  if (resolution.reciprocal_volume && axes[0] != Coordinate::V &&
      axes[1] != Coordinate::V)
    throw std::invalid_argument(
        "Reciprocal volume resolution requires a V axis.");
  if (resolution.region) {
    const auto ranges = result.coordinate_ranges();
    for (std::size_t axis = 0; axis < 2; ++axis) {
      const auto &bounds = (*resolution.region)[axis];
      if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) ||
          bounds[0] >= bounds[1] || bounds[0] < ranges[axis][0] ||
          bounds[1] > ranges[axis][1])
        throw std::invalid_argument("Refinement region must have finite "
                                    "increasing bounds within the diagram.");
    }
  }
}

Settings refinement_settings(const Result &result,
                             const LineResolution &resolution,
                             const Settings &settings) {
  auto local = settings;
  if (resolution.region) {
    const auto ranges = result.coordinate_ranges();
    for (std::size_t axis = 0; axis < 2; ++axis) {
      const auto &bounds = (*resolution.region)[axis];
      const double spacing = (bounds[1] - bounds[0]) /
                             (ranges[axis][1] - ranges[axis][0]) /
                             (resolution.axis_points[axis] - 1);
      local.step = std::min(local.step, spacing * .25);
    }
    local.node_tolerance = std::min(local.node_tolerance, local.step * .02);
    local.min_step = std::min(local.min_step, local.step * 1.e-4);
  }
  return local;
}

double refinement_step(const Result &result, const LineResolution &resolution,
                       const Eigen::Vector2d &position,
                       const Eigen::Vector2d &direction,
                       const Settings &settings) {
  if (!resolution.region ||
      !intersects_region(position, position + settings.step * direction,
                         *resolution.region))
    return settings.step;
  return refinement_settings(result, resolution, settings).step;
}

std::vector<int> regional_retrace_lines(const Result &result,
                                        const LineResolution &resolution) {
  std::vector<int> selected;
  if (!resolution.region)
    return selected;
  for (const auto &polygon : field_polygons(result).polygons)
    if (polygon.is_provisional && !polygon.outside_model_domain) {
      selected.insert(selected.end(), polygon.boundary_ids.begin(),
                      polygon.boundary_ids.end());
      if (result.section.type == DiagramType::PT) {
        const auto inside = interior_boundaries(result, polygon);
        selected.insert(selected.end(), inside.begin(), inside.end());
      }
    }
  // A wrong arc can bound a plausible adjacent loop while hiding the branch
  // needed by the conflicting loop. Revisit its incident arcs at local nodes.
  std::vector<int> junctions;
  for (const auto &line : result.boundaries)
    if (std::find(selected.begin(), selected.end(), line.id) != selected.end())
      for (int id : {line.start_node, line.end_node})
        if (id >= 0) {
          const auto q =
              diagram_coordinates(result.nodes.at(static_cast<std::size_t>(id)),
                                  result.section.type);
          if (intersects_region(q, q, *resolution.region))
            junctions.push_back(id);
        }
  for (const auto &line : result.boundaries)
    if (std::find(junctions.begin(), junctions.end(), line.start_node) !=
            junctions.end() ||
        std::find(junctions.begin(), junctions.end(), line.end_node) !=
            junctions.end())
      selected.push_back(line.id);
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  selected.erase(
      std::remove_if(
          selected.begin(), selected.end(),
          [&](int id) {
            const auto line =
                std::find_if(result.boundaries.begin(), result.boundaries.end(),
                             [&](const Boundary &b) { return b.id == id; });
            if (line != result.boundaries.end())
              for (std::size_t k = 1; k < line->points.size(); ++k)
                if (intersects_region(diagram_coordinates(line->points[k - 1],
                                                          result.section.type),
                                      diagram_coordinates(line->points[k],
                                                          result.section.type),
                                      *resolution.region))
                  return false;
            return true;
          }),
      selected.end());
  return selected;
}

std::vector<int> interior_boundaries(const Result &result,
                                     const FieldPolygon &polygon) {
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto q = [&](const Eigen::Vector2d &p) -> Eigen::Vector2d {
    return (p - origin).cwiseQuotient(range);
  };
  std::vector<int> crossings;
  for (const auto &line : result.boundaries) {
    if (line.zero_phase < 0 ||
        std::find(polygon.boundary_ids.begin(), polygon.boundary_ids.end(),
                  line.id) != polygon.boundary_ids.end())
      continue;
    bool found = false;
    for (std::size_t k = 1; k < line.points.size() && !found; ++k) {
      const auto a = q(diagram_coordinates(line.points[k - 1],
                                           result.section.type)),
                 b = q(
                     diagram_coordinates(line.points[k], result.section.type));
      const Eigen::Vector2d d = b - a;
      if (d.isZero(0.))
        continue;
      std::vector<double> cuts{0., 1.};
      auto intersect = [&](const Eigen::MatrixXd &ring) {
        for (Eigen::Index j = 1; j < ring.rows(); ++j) {
          const auto c = q(ring.row(j - 1)), z = q(ring.row(j));
          const Eigen::Vector2d e = z - c;
          const double denominator = cross(d, e);
          if (std::abs(denominator) < 1.e-20)
            continue;
          const double t = cross(c - a, e) / denominator,
                       u = cross(c - a, d) / denominator;
          if (t > 0. && t < 1. && u >= 0. && u <= 1.)
            cuts.push_back(t);
        }
      };
      intersect(polygon.vertices);
      for (const auto &hole : polygon.holes)
        intersect(hole);
      std::sort(cuts.begin(), cuts.end());
      for (std::size_t j = 1; j < cuts.size(); ++j) {
        const Eigen::Vector2d p =
            origin + (a + .5 * (cuts[j - 1] + cuts[j]) * d).cwiseProduct(range);
        if (polygon.contains_rectangle(p, range * 1.e-12)) {
          crossings.push_back(line.id);
          found = true;
          break;
        }
      }
    }
  }
  return crossings;
}

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
  const auto ranges = resolution.region.value_or(result.coordinate_ranges());
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
      const auto a = diagram_coordinates(line.points[k - 1],
                                         result.section.type),
                 b = diagram_coordinates(line.points[k], result.section.type);
      if (resolution.region && !intersects_region(a, b, ranges)) {
        ++k;
        continue;
      }
      const auto delta = (measured(b) - measured(a)).cwiseAbs().eval();
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
