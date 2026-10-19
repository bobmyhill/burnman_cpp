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
#include "burnman/tools/phase_diagram/types.hpp"
#include <algorithm>
#include <limits>

namespace burnman::phase_diagrams::detail {
using namespace equilibration;
using PhaseStates = std::vector<PhaseState>;
using EquilibriumSolve = optim::roots::DampedNewtonResult;
// C++17 equivalent of std::erase_if, shared by graph and phase-state updates.
template <typename Container, typename Predicate>
void erase_if(Container &values, Predicate predicate) {
  values.erase(std::remove_if(values.begin(), values.end(), predicate),
               values.end());
}
/// Find one saved phase instance by its local ID; return states.end() if
/// absent.
template <typename States> auto find_phase(States &states, int id) {
  return std::find_if(states.begin(), states.end(),
                      [id](const auto &phase) { return phase.id == id; });
}
/// Total moles of the supplied phases, on the bulk composition's amount scale.
inline double total_amount(const PhaseStates &phases) {
  double total = 0.;
  for (const auto &phase : phases)
    total += phase.amount;
  return total;
}
/// Instance IDs in saved order, including phases with zero amount.
inline std::vector<int> phase_ids(const PhaseStates &phases) {
  std::vector<int> ids;
  for (const auto &phase : phases)
    ids.push_back(phase.id);
  return ids;
}
/// IDs of phases whose mole fractions exceed amount_tolerance. This separates
/// a boundary's zero-amount phase from the phases actually present there.
inline std::vector<int> present_phase_ids(const PhaseStates &phases,
                                          double amount_tolerance) {
  const double cutoff = amount_tolerance * total_amount(phases);
  std::vector<int> ids;
  for (const auto &phase : phases)
    if (phase.amount > cutoff)
      ids.push_back(phase.id);
  return ids;
}
/// Midpoint warm start for matching phase instances: average amounts and use
/// amount-weighted compositions, or an unweighted mean when both amounts
/// vanish. This is a trial state, not a new equilibrium; the caller must
/// re-equilibrate it.
PhaseStates average_phase_states(PhaseStates, const PhaseStates &);
/// Largest absolute reaction affinity (J/mol), or zero if there are no
/// reactions.
inline double reaction_residual(const Assemblage &a) {
  return a.get_n_reactions() ? a.get_reaction_affinities().cwiseAbs().maxCoeff()
                             : 0.;
}
/// Whether adjacent field labels are missing or indistinguishable. A verified
/// switch between solution compositions can retain the same phase names.
inline bool ambiguous_boundary(const Boundary &line) {
  return line.side_a.empty() || line.side_b.empty() ||
         (line.side_a == line.side_b && !line.is_solution_replacement);
}
/// Same two neighbouring assemblages, allowing the trace direction to reverse.
inline bool same_boundary_sides(const Boundary &a, const Boundary &b) {
  return (a.side_a == b.side_a && a.side_b == b.side_b) ||
         (a.side_a == b.side_b && a.side_b == b.side_a);
}
/// Same saved endpoints in either order; this alone does not prove one
/// reaction.
inline bool same_boundary_nodes(const Boundary &a, const Boundary &b) {
  return (a.start_node == b.start_node && a.end_node == b.end_node) ||
         (a.start_node == b.end_node && a.end_node == b.start_node);
}
/// Register a boundary once at each endpoint, for junction branch accounting.
inline void record_incident_line(Result &result, const Boundary &line) {
  for (int id : {line.start_node, line.end_node})
    if (id >= 0) {
      auto &incident =
          result.nodes.at(static_cast<std::size_t>(id)).incident_lines;
      if (std::find(incident.begin(), incident.end(), line.id) ==
          incident.end())
        incident.push_back(line.id);
    }
}
/// Rebuild node connections after arcs have been split, removed or renumbered.
inline void rebuild_boundary_incidence(Result &result) {
  for (auto &node : result.nodes)
    node.incident_lines.clear();
  for (const auto &line : result.boundaries)
    record_incident_line(result, line);
}
/// Signed 2-D cross product; its sign gives the sense of a turn between
/// vectors.
inline double cross(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
  return a.x() * b.y() - a.y() * b.x();
}
/// Test whether a straight segment meets a rectangular refinement/label window.
/// Contact counts as intersection. parallel_tolerance treats a small coordinate
/// change as zero; it uses coordinate units and does not expand the rectangle.
bool intersects_region(const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                       const std::array<std::array<double, 2>, 2> &ranges,
                       double parallel_tolerance = 0.);
/// Shortest distance to a closed straight segment, including its endpoints.
inline double segment_distance(const Eigen::Vector2d &p,
                               const Eigen::Vector2d &a,
                               const Eigen::Vector2d &b) {
  Eigen::Vector2d d = b - a;
  double t = d.squaredNorm() > 0.
                 ? std::clamp((p - a).dot(d) / d.squaredNorm(), 0., 1.)
                 : 0.;
  return (p - a - t * d).norm();
}
/// Distance to a polyline (an ordered chain of straight segments). vertex(i)
/// supplies coordinates in the same units as p; fewer than two vertices gives
/// infinity because no segment is defined.
template <typename Vertex>
double polyline_distance(std::size_t n, Vertex vertex,
                         const Eigen::Vector2d &p) {
  double distance = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < n; ++i)
    distance =
        std::min(distance, segment_distance(p, vertex(i - 1), vertex(i)));
  return distance;
}
/// Whether every point of first lies within tolerance of the second polyline.
/// The test is directional; reversing the arguments tests the other coverage.
template <typename Points, typename Coordinate>
bool follows_polyline(const Points &first, const Points &second,
                      Coordinate coordinate, double tolerance) {
  for (const auto &point : first)
    if (polyline_distance(
            second.size(), [&](std::size_t i) { return coordinate(second[i]); },
            coordinate(point)) > tolerance)
      return false;
  return true;
}
/// Classify a point relative to a closed coordinate ring. vertex(n-1) must
/// repeat vertex(0); p and tolerance use the ring's units. Return -1 outside,
/// 0 within tolerance of an edge, or +1 inside, using the even-odd crossing
/// rule.
template <typename Vertex>
int ring_location(std::size_t n, Vertex vertex, const Eigen::Vector2d &p,
                  double tolerance) {
  bool inside = false;
  for (std::size_t i = 1; i < n; ++i) {
    Eigen::Vector2d a = vertex(i - 1), b = vertex(i);
    if (segment_distance(p, a, b) <= tolerance)
      return 0;
    if ((a.y() > p.y()) != (b.y() > p.y()) &&
        p.x() < a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y()))
      inside = !inside;
  }
  return inside ? 1 : -1;
}
/// Native P/T solve accuracy in Pa and K, also used to compare junction states.
inline Eigen::Vector2d equilibrium_pt_tolerance() { return {.2, 1.e-6}; }
/// Renumber copies within each candidate model, retaining phase multiplicity.
/// Thus two feldspars still count as two phases, but exchanging their local
/// copy numbers does not change the assemblage identity.
std::vector<int> canonical_phase_ids(const std::vector<int> &, int copies);

/// Smallest endmember-composition distance between copies of one solution.
/// Infinity means there is no same-model pair; a small value suggests copies
/// are coalescing near a critical point or duplicating the same phase.
double solution_separation(const PhaseStates &);
} // namespace burnman::phase_diagrams::detail
