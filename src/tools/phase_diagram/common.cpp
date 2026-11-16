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

#include "burnman/tools/phase_diagram/common.hpp"
#include <cmath>
#include <map>

namespace burnman::phase_diagrams {
std::string Diagnostic::text() const {
  const auto id = std::to_string(entity_id);
  if (grid_point[0] >= 0) {
    const auto field =
        category == DiagnosticCategory::property && entity_id >= 0
            ? "Field " + id + ": "
            : "";
    return field + "grid [" + std::to_string(grid_point[0]) + ", " +
           std::to_string(grid_point[1]) + "]: " + message;
  }
  switch (category) {
  case DiagnosticCategory::trace:
  case DiagnosticCategory::boundary_note:
  case DiagnosticCategory::boundary:
  case DiagnosticCategory::spacing:
    return "Boundary " + id + ": " + message;
  case DiagnosticCategory::field:
    return "Field " + id + message;
  case DiagnosticCategory::junction:
    return "Unpaired internal junction at node " + id + ".";
  case DiagnosticCategory::critical_endpoint:
    return "Unpaired solution critical endpoint at node " + id +
           "; its adjoining field remains unresolved.";
  case DiagnosticCategory::missing_assemblage:
    return "No closed region resolved for sampled assemblage " + id +
           "; increase seed density or inspect failed solves.";
  default:
    return message;
  }
}
} // namespace burnman::phase_diagrams

namespace burnman::phase_diagrams::detail {
bool intersects_region(const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                       const std::array<std::array<double, 2>, 2> &ranges,
                       double parallel_tolerance) {
  double first = 0., last = 1.;
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    const auto &bounds = ranges[static_cast<std::size_t>(axis)];
    const double delta = b[axis] - a[axis];
    if (delta == 0. || std::abs(delta) < parallel_tolerance) {
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

std::optional<SegmentIntersection>
segment_intersection(const Eigen::Vector2d &direction,
                     const Eigen::Vector2d &other_direction,
                     const Eigen::Vector2d &offset) {
  const double determinant = cross(direction, other_direction);
  if (std::abs(determinant) < 1.e-30) {
    // Use cross-product roundoff, rather than a geometric exclusion band
    // that could swallow a narrow field between nearly parallel reactions.
    const double roundoff = 16. * std::numeric_limits<double>::epsilon() *
                            (std::abs(direction[0] * offset[1]) +
                             std::abs(direction[1] * offset[0]));
    if (direction.squaredNorm() > 0. && other_direction.squaredNorm() > 0. &&
        std::abs(cross(direction, offset)) <= roundoff) {
      const double first = offset.dot(direction) / direction.squaredNorm(),
                   last = (offset + other_direction).dot(direction) /
                          direction.squaredNorm(),
                   low = std::max(0., std::min(first, last)),
                   high = std::min(1., std::max(first, last));
      if (high - low > 1.e-9)
        return SegmentIntersection{low, high};
    }
    return std::nullopt;
  }
  const double along = cross(offset, other_direction) / determinant,
               across = cross(offset, direction) / determinant;
  if (along < -1.e-9 || along > 1. + 1.e-9 || across < -1.e-9 ||
      across > 1. + 1.e-9)
    return std::nullopt;
  return SegmentIntersection{along, along};
}

PhaseStates junction_states(const std::vector<Boundary> &lines,
                            const Node &event) {
  PhaseStates phases;
  std::pair<std::size_t, std::size_t> best{};
  for (const auto &peer : lines)
    if (!peer.points.empty() &&
        (peer.start_node == event.id || peer.end_node == event.id)) {
      const auto &states = (peer.start_node == event.id ? peer.points.front()
                                                        : peer.points.back())
                               .phases;
      std::map<int, std::size_t> counts;
      std::size_t size = 0, copies = 0;
      for (const auto &state : states)
        if (std::find(event.assemblage.begin(), event.assemblage.end(),
                      state.id) != event.assemblage.end()) {
          ++size;
          copies += counts[state.candidate_index]++;
        }
      const auto score = std::make_pair(size, copies);
      if (score > best) {
        best = score;
        phases = states;
      }
    }
  // Keep coexisting copies together. A single-copy arm may reuse the other
  // copy's ID after canonicalisation; combining that arm first can otherwise
  // reconstruct two identical compositions and lose the absent solution.
  for (const auto &peer : lines)
    if (!peer.points.empty() &&
        (peer.start_node == event.id || peer.end_node == event.id))
      for (const auto &p : (peer.start_node == event.id ? peer.points.front()
                                                        : peer.points.back())
                               .phases)
        if (std::none_of(phases.begin(), phases.end(),
                         [&](const auto &q) { return p.id == q.id; }))
          phases.push_back(p);
  erase_if(phases, [&](const auto &p) {
    return std::find(event.assemblage.begin(), event.assemblage.end(), p.id) ==
           event.assemblage.end();
  });
  return phases;
}

PhaseStates average_phase_states(PhaseStates states, const PhaseStates &other) {
  for (auto &state : states) {
    const auto match = find_phase(other, state.id);
    if (match == other.end())
      continue;
    const double amount = state.amount + match->amount;
    if (amount > 0.)
      state.composition = (state.amount * state.composition +
                           match->amount * match->composition) /
                          amount;
    else
      state.composition = .5 * (state.composition + match->composition);
    state.amount = .5 * amount;
  }
  return states;
}

std::vector<int> canonical_phase_ids(const std::vector<int> &ids, int copies) {
  std::map<int, int> counts;
  for (int id : ids)
    ++counts[id / copies];
  std::vector<int> canonical;
  for (auto [base, count] : counts)
    for (int copy = 0; copy < count; ++copy)
      canonical.push_back(base * copies + copy);
  return canonical;
}

std::set<std::vector<int>> boundary_assemblages(const Boundary &line,
                                                const Result &result) {
  std::set<std::vector<int>> choices;
  if (line.assemblage.empty())
    return choices;
  auto insert = [&](const std::vector<int> &phases) {
    if (!phases.empty())
      choices.insert(
          canonical_phase_ids(phases, result.settings.max_phase_instances));
  };
  if (line.reaction_kind == ReactionKind::phase_out && line.zero_phase >= 0) {
    insert(line.assemblage);
    auto absent = line.assemblage;
    erase_if(absent, [&](int id) { return id == line.zero_phase; });
    insert(absent);
  } else if (line.reaction_kind == ReactionKind::phase_out ||
             line.reaction_kind == ReactionKind::replacement) {
    insert(line.side_a);
    insert(line.side_b);
  }
  return choices;
}

ReactionPerimeter reaction_perimeter(const Result &result,
                                     const std::set<const Boundary *> &lines,
                                     const std::vector<int> &phases) {
  if (lines.empty())
    return {{}, "Incomplete or incompatible reaction perimeter"};
  std::map<int, const Node *> nodes;
  std::map<int, std::vector<int>> neighbours;
  for (const auto &node : result.nodes)
    nodes.emplace(node.id, &node);
  for (const auto *line : lines) {
    if (line->points.size() < 2 || line->start_node < 0 || line->end_node < 0 ||
        !boundary_assemblages(*line, result).count(phases))
      return {{}, "Incomplete or incompatible reaction perimeter"};
    const auto assemblage = canonical_phase_ids(
        line->assemblage, result.settings.max_phase_instances);
    for (int id : {line->start_node, line->end_node}) {
      const auto node = nodes.find(id);
      if (node == nodes.end())
        return {{},
                "Reaction endpoint does not reach its stated common vertex"};
      const auto event = canonical_phase_ids(
          node->second->assemblage, result.settings.max_phase_instances);
      if (!std::includes(event.begin(), event.end(), assemblage.begin(),
                         assemblage.end()))
        return {{},
                "Reaction assemblage is incompatible with its common vertex"};
    }
    neighbours[line->start_node].push_back(line->end_node);
    neighbours[line->end_node].push_back(line->start_node);
  }
  std::set<int> reached;
  std::vector<int> pending{neighbours.begin()->first};
  while (!pending.empty()) {
    const int vertex = pending.back();
    pending.pop_back();
    if (!reached.insert(vertex).second)
      continue;
    const auto &next = neighbours.at(vertex);
    pending.insert(pending.end(), next.begin(), next.end());
  }
  if (reached.size() != neighbours.size())
    return {{}, "Perimeter does not form one connected reaction loop"};
  ReactionPerimeter perimeter;
  for (const auto &[id, next] : neighbours) {
    if (next.size() != 2)
      return {{}, "Perimeter reactions do not meet at common vertices"};
    perimeter.nodes.push_back(nodes.at(id));
  }
  return perimeter;
}

double solution_separation(const PhaseStates &states) {
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < states.size(); ++i)
    for (std::size_t j = i + 1; j < states.size(); ++j)
      if (states[i].candidate_index == states[j].candidate_index)
        minimum = std::min(
            minimum, (states[i].composition - states[j].composition).norm());
  return minimum;
}
} // namespace burnman::phase_diagrams::detail
