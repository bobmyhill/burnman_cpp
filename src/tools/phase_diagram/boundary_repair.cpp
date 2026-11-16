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

#include "burnman/tools/phase_diagram/boundary_repair.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/diagram_verification.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/refinement.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/utils/index_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <set>

namespace burnman::phase_diagrams::detail {
std::set<int> reaction_overtraces(const Result &result) {
  std::set<int> redundant;
  if (result.section.type != DiagramType::PT)
    return redundant;
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]);
  const Eigen::Vector2d scale(ranges[0][1] - origin[0],
                              ranges[1][1] - origin[1]);
  auto coordinate = [&](const Eigen::Vector2d &point) {
    return (point - origin).cwiseQuotient(scale).eval();
  };
  std::map<int, const Node *> nodes;
  for (const auto &node : result.nodes)
    nodes.emplace(node.id, &node);
  std::map<int, std::vector<const Boundary *>> incident;
  for (const auto &line : result.boundaries)
    if (line.points.size() >= 2) {
      incident[line.start_node].push_back(&line);
      if (line.end_node != line.start_node)
        incident[line.end_node].push_back(&line);
    }
  auto same_assemblage = [&](const Boundary &a, const Boundary &b) {
    return canonical_phase_ids(a.assemblage,
                               result.settings.max_phase_instances) ==
           canonical_phase_ids(b.assemblage,
                               result.settings.max_phase_instances);
  };
  auto valid_sides = [&](const Boundary &line) {
    if (line.side_a.empty() || line.side_b.empty() ||
        line.side_a == line.side_b)
      return false;
    const auto full = canonical_phase_ids(line.assemblage,
                                          result.settings.max_phase_instances);
    for (const auto *side : {&line.side_a, &line.side_b}) {
      const auto phases =
          canonical_phase_ids(*side, result.settings.max_phase_instances);
      if (!std::includes(full.begin(), full.end(), phases.begin(),
                         phases.end()))
        return false;
    }
    return true;
  };
  auto at = [&](const Boundary &line, int node) {
    std::vector<Eigen::Vector2d> points;
    points.reserve(line.points.size());
    for (const auto &point : line.points)
      points.push_back(diagram_coordinates(point, result.section.type));
    if (line.end_node == node)
      std::reverse(points.begin(), points.end());
    return points;
  };
  for (const auto &long_line : result.boundaries) {
    if (long_line.points.size() < 2 || long_line.start_node < 0 ||
        long_line.end_node < 0 || long_line.start_node == long_line.end_node ||
        long_line.reaction_kind != ReactionKind::replacement ||
        !valid_sides(long_line))
      continue;
    const auto start = nodes.find(long_line.start_node);
    const auto end = nodes.find(long_line.end_node);
    if ((start == nodes.end() || start->second->kind != "domain_edge") &&
        (end == nodes.end() || end->second->kind != "domain_edge"))
      continue;
    std::vector<Eigen::Vector2d> long_points;
    for (const auto *first : incident[long_line.start_node]) {
      if (first->id == long_line.id ||
          first->reaction_kind != ReactionKind::replacement ||
          !valid_sides(*first) || !same_assemblage(long_line, *first))
        continue;
      const int middle = first->start_node == long_line.start_node
                             ? first->end_node
                             : first->start_node;
      const auto junction = nodes.find(middle);
      if (middle < 0 || middle == long_line.end_node ||
          junction == nodes.end() || junction->second->kind != "junction")
        continue;
      for (const auto *second : incident[middle]) {
        if (second->id == long_line.id || second->id == first->id ||
            (second->start_node != long_line.end_node &&
             second->end_node != long_line.end_node) ||
            second->reaction_kind != ReactionKind::replacement ||
            !valid_sides(*second) || !same_assemblage(long_line, *second) ||
            same_boundary_sides(*first, *second) ||
            (same_boundary_sides(long_line, *first) ==
             same_boundary_sides(long_line, *second)))
          continue;
        // Both branches must be certified by the saved junction's distinct
        // zero-phase conditions. The common limiting assemblage connects them.
        const auto &zeros = junction->second->zero_phases;
        if (first->zero_phase == second->zero_phase ||
            std::find(zeros.begin(), zeros.end(), first->zero_phase) ==
                zeros.end() ||
            std::find(zeros.begin(), zeros.end(), second->zero_phase) ==
                zeros.end())
          continue;
        const auto shared = [&](const std::vector<int> &a,
                                const std::vector<int> &b) {
          return canonical_phase_ids(a, result.settings.max_phase_instances) ==
                 canonical_phase_ids(b, result.settings.max_phase_instances);
        };
        if (!(shared(first->side_a, second->side_a) ||
              shared(first->side_a, second->side_b) ||
              shared(first->side_b, second->side_a) ||
              shared(first->side_b, second->side_b)))
          continue;
        auto chain = at(*first, long_line.start_node);
        auto tail = at(*second, middle);
        chain.insert(chain.end(), tail.begin() + 1, tail.end());
        if (long_points.empty())
          long_points = at(long_line, long_line.start_node);
        const double tolerance = result.settings.node_tolerance;
        if (follows_polyline(long_points, chain, coordinate, tolerance) &&
            follows_polyline(chain, long_points, coordinate, tolerance)) {
          redundant.insert(long_line.id);
          break;
        }
      }
      if (redundant.count(long_line.id))
        break;
    }
  }
  return redundant;
}

namespace {
struct SolvusBracket {
  struct Endpoint {
    double coordinate;
    WorkState work;
  };
  Engine &engine;
  std::vector<int> low, high;
  int high_side = 0; // Positive when the upper endpoint has more phases.

  SolvusBracket(Engine &e, const Boundary &arm) : engine(e) {
    const bool a_is_smaller = arm.side_a.size() < arm.side_b.size();
    low = engine.canonical_ids(a_is_smaller ? arm.side_a : arm.side_b);
    high = engine.canonical_ids(a_is_smaller ? arm.side_b : arm.side_a);
  }
  bool valid() const {
    return !low.empty() && high.size() == low.size() + 1 &&
           std::includes(high.begin(), high.end(), low.begin(), low.end());
  }
  int side(const WorkState &work) const {
    if (!work.state.success)
      return 0;
    const auto ids = engine.active_phases(work.state);
    return ids == high ? 1 : (ids == low ? -1 : 0);
  }
  bool brackets(const Endpoint &lower, const Endpoint &upper) {
    const int upper_side = side(upper.work);
    if (!upper_side || side(lower.work) != -upper_side ||
        (high_side && high_side != upper_side))
      return false;
    high_side = upper_side;
    return true;
  }
  // Refine an accepted bracket. Each caller supplies its midpoint coordinate:
  // requested offset for a gap, solved pressure for continuation to the frame.
  template <typename Evaluate>
  bool refine(Endpoint &lower, Endpoint &upper, int iterations,
              Evaluate at) const {
    try {
      for (int iteration = 0; iteration < iterations; ++iteration) {
        auto trial = at(.5 * (lower.coordinate + upper.coordinate));
        const int trial_side = side(trial.work);
        if (!trial_side)
          return false;
        (trial_side == high_side ? upper : lower) = std::move(trial);
      }
      return true;
    } catch (const std::exception &) {
      return false;
    }
  }
  const State &high_state(const Endpoint &lower, const Endpoint &upper) const {
    return (high_side > 0 ? upper : lower).work.state;
  }
};

BoundaryPoint boundary_point_from_state(const State &state) {
  BoundaryPoint point;
  copy_coordinates(point, state);
  point.phases = state.phases;
  point.mass_balance_error = state.mass_balance_error;
  point.minimum_affinity = state.minimum_affinity;
  point.residual = state.equilibrium_error;
  return point;
}

void mark_continued_boundary(Result &result, int line_id, int node_id) {
  for (auto &line : result.boundaries)
    if (line.id == line_id) {
      auto [back, forward] = boundary_statuses(line);
      if (line.start_node == node_id && back == "solution critical point")
        back = "boundary continuation";
      if (line.end_node == node_id && forward == "solution critical point")
        forward = "boundary continuation";
      line.termination = back + "; " + forward;
      return;
    }
}
} // namespace

std::size_t recover_solvus_branch_gaps(Result &result, Engine &engine) {
  if (result.section.type != DiagramType::PT)
    return 0;
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]);
  const Eigen::Vector2d span(ranges[0][1] - origin[0],
                             ranges[1][1] - origin[1]);
  auto position = [&](const Node &node) {
    return (diagram_coordinates(node, DiagramType::PT) - origin)
        .cwiseQuotient(span)
        .eval();
  };
  std::size_t added = 0;
  for (std::size_t i = 0; i < result.nodes.size(); ++i) {
    const auto &first = result.nodes[i];
    if (first.kind != "critical_point" ||
        !first.is_verified_critical_endpoint ||
        first.incident_lines.size() != 1)
      continue;
    for (std::size_t j = i + 1; j < result.nodes.size(); ++j) {
      const auto &second = result.nodes[j];
      if (second.kind != "critical_point" ||
          !second.is_verified_critical_endpoint ||
          second.incident_lines.size() != 1 ||
          first.temperature == second.temperature)
        continue;
      const auto *arm_a = [&]() -> const Boundary * {
        for (const auto &line : result.boundaries)
          if (line.id == first.incident_lines.front())
            return &line;
        return nullptr;
      }();
      const auto *arm_b = [&]() -> const Boundary * {
        for (const auto &line : result.boundaries)
          if (line.id == second.incident_lines.front())
            return &line;
        return nullptr;
      }();
      if (!arm_a || !arm_b || arm_a == arm_b ||
          arm_a->reaction_kind != ReactionKind::phase_out ||
          arm_b->reaction_kind != ReactionKind::phase_out ||
          arm_a->zero_phase < 0 || arm_b->zero_phase < 0 ||
          arm_a->zero_phase / engine.settings.max_phase_instances !=
              arm_b->zero_phase / engine.settings.max_phase_instances ||
          engine.canonical_ids(arm_a->assemblage) !=
              engine.canonical_ids(arm_b->assemblage) ||
          !same_boundary_sides(*arm_a, *arm_b))
        continue;
      const Node &start =
          first.temperature < second.temperature ? first : second;
      const Node &end = first.temperature < second.temperature ? second : first;
      const Boundary &start_arm =
          first.temperature < second.temperature ? *arm_a : *arm_b;
      const Boundary &end_arm =
          first.temperature < second.temperature ? *arm_b : *arm_a;
      SolvusBracket front(engine, start_arm);
      if (!front.valid())
        continue;
      const Eigen::Vector2d a = position(start), b = position(end);
      const Eigen::Vector2d tangent = b - a;
      const double length = tangent.norm();
      if (length <= 0.)
        continue;
      const Eigen::Vector2d right(tangent.y() / length, -tangent.x() / length);
      Boundary line;
      line.id = burnman::utils::checked_int(result.boundaries.size());
      // Copy numbers can exchange along this segment. Its stable two- and
      // three-copy sides are identified by equilibria, not by a fixed zero ID.
      line.zero_phase = -1;
      line.start_node = start.id;
      line.end_node = end.id;
      line.assemblage = front.high;
      line.reaction_kind = ReactionKind::phase_out;
      line.termination = "boundary continuation; boundary continuation";
      line.points.push_back(start_arm.start_node == start.id
                                ? start_arm.points.front()
                                : start_arm.points.back());
      bool accepted = true;
      // Locate the stable phase front on each cross-section. These points
      // approach the front from the high-phase side; they are not exact
      // coexistence solves, so retain the explicit phase labels on each side.
      for (int step = 1; step < 12 && accepted; ++step) {
        const auto centre =
            (a + (static_cast<double>(step) / 12.) * tangent).eval();
        const double offset = .1 * length;
        auto at = [&](double distance) {
          auto work = engine.stable_at(
              origin + (centre + distance * right).cwiseProduct(span));
          return SolvusBracket::Endpoint{distance, std::move(work)};
        };
        try {
          auto left = at(-offset), right_state = at(offset);
          if (!front.brackets(left, right_state) ||
              !front.refine(left, right_state, 18, at)) {
            accepted = false;
            break;
          }
          line.points.push_back(
              boundary_point_from_state(front.high_state(left, right_state)));
        } catch (const std::exception &) {
          accepted = false;
        }
      }
      if (!accepted)
        continue;
      line.points.push_back(end_arm.start_node == end.id
                                ? end_arm.points.front()
                                : end_arm.points.back());
      line.side_a = front.high_side > 0 ? front.high : front.low;
      line.side_b = front.high_side > 0 ? front.low : front.high;
      const int start_arm_id = start_arm.id, end_arm_id = end_arm.id;
      result.boundaries.push_back(std::move(line));
      rebuild_boundary_incidence(result);
      // A verified phase front is physical even when another unfinished edge
      // leaves the field graph provisional. Field-count changes cannot decide
      // whether this reaction continues.
      result.nodes[i].kind = "boundary_continuation";
      result.nodes[j].kind = "boundary_continuation";
      mark_continued_boundary(result, start_arm_id, start.id);
      mark_continued_boundary(result, end_arm_id, end.id);
      ++added;
      break;
    }
  }
  return added;
}

std::size_t continue_boundary_past_coalescence(Result &result, Engine &engine) {
  if (result.section.type != DiagramType::PT)
    return 0;
  const double p_min = result.pressure_range[0];
  const double p_max = result.pressure_range[1];
  const double p_span = p_max - p_min;
  const double t_min = result.temperature_range[0];
  const double t_max = result.temperature_range[1];
  std::size_t added = 0;
  const auto original_nodes = result.nodes.size();
  for (std::size_t i = 0; i < original_nodes; ++i) {
    const Node start = result.nodes[i];
    if (start.kind != "critical_point" ||
        !start.is_verified_critical_endpoint ||
        start.incident_lines.size() != 1 || start.gibbs_variance != 2)
      continue;
    const auto found =
        std::find_if(result.boundaries.begin(), result.boundaries.end(),
                     [&](const Boundary &line) {
                       return line.id == start.incident_lines[0];
                     });
    if (found == result.boundaries.end())
      continue;
    const Boundary arm = *found;
    if (arm.reaction_kind != ReactionKind::phase_out || arm.points.size() < 2 ||
        arm.side_a.empty() || arm.side_b.empty())
      continue;
    const bool arm_starts_here = arm.start_node == start.id;
    const auto &next =
        arm_starts_here ? arm.points[1] : arm.points[arm.points.size() - 2];
    const double dt = start.temperature - next.temperature;
    if (std::abs(dt) < 1.e-8)
      continue;
    const double frame = dt < 0. ? t_min : t_max;
    if ((frame - start.temperature) * dt <= 0.)
      continue;
    SolvusBracket front(engine, arm);
    if (!front.valid())
      continue;
    const int steps = std::clamp(
        static_cast<int>(std::ceil(std::abs(frame - start.temperature) /
                                   std::max(10., (t_max - t_min) / 80.))),
        4, 40);
    double previous_t = start.temperature;
    double previous_p = start.pressure;
    double slope = (start.pressure - next.pressure) / dt;
    Boundary line;
    line.id = burnman::utils::checked_int(result.boundaries.size());
    line.start_node = start.id;
    line.zero_phase = -1;
    line.assemblage = front.high;
    line.reaction_kind = ReactionKind::phase_out;
    line.side_a = arm_starts_here ? arm.side_b : arm.side_a;
    line.side_b = arm_starts_here ? arm.side_a : arm.side_b;
    line.termination = "boundary continuation; domain edge";
    line.points.push_back(arm_starts_here ? arm.points.front()
                                          : arm.points.back());
    // The saved reaction corrector becomes singular as solution copies merge.
    // Bracket the stable assemblage change at each temperature instead of
    // assuming that this coalescence ends the geometric phase boundary.
    bool accepted = true;
    for (int step = 1; step <= steps && accepted; ++step) {
      const double t =
          start.temperature +
          (frame - start.temperature) * (static_cast<double>(step) / steps);
      const double predicted = previous_p + slope * (t - previous_t);
      SolvusBracket::Endpoint below, above;
      auto at = [&](double pressure) {
        auto work = engine.stable_at({pressure, t});
        return SolvusBracket::Endpoint{work.state.pressure, std::move(work)};
      };
      bool bracketed = false;
      for (int trial = 0; trial < 7; ++trial) {
        const double offset = std::ldexp(p_span * .001, trial);
        const double left = std::max(p_min, predicted - offset);
        const double right = std::min(p_max, predicted + offset);
        if (left >= right)
          continue;
        try {
          below = at(left);
          above = at(right);
          if (front.brackets(below, above)) {
            bracketed = true;
            break;
          }
        } catch (const std::exception &) {
        }
      }
      if (!bracketed || !front.refine(below, above, 24, at)) {
        accepted = false;
        break;
      }
      const auto &state = front.high_state(below, above);
      line.points.push_back(boundary_point_from_state(state));
      slope = (state.pressure - previous_p) / (t - previous_t);
      previous_p = state.pressure;
      previous_t = t;
    }
    if (!accepted)
      continue;
    Node end;
    end.id = burnman::utils::checked_int(result.nodes.size());
    end.kind = "domain_edge";
    copy_coordinates(end, line.points.back());
    end.temperature = frame;
    end.assemblage = front.high;
    line.end_node = end.id;
    result.nodes.push_back(std::move(end));
    result.boundaries.push_back(std::move(line));
    rebuild_boundary_incidence(result);
    // The physical side-change checks above establish the reaction front.
    // Ring reconstruction must not decide whether to discard it: a missing
    // neighbouring trace can leave even a correct front unenclosed.
    result.nodes[i].kind = "boundary_continuation";
    mark_continued_boundary(result, arm.id, start.id);
    ++added;
  }
  return added;
}

namespace {
// Proximity proposes duplicate junctions; their independently constrained
// equilibria must agree at the native solver's accuracy before joining them.
void join_duplicate_junctions(Result &result, Engine &engine) {
  if (has_composition_axis(result.section.type))
    return;
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d range(ranges[0][1] - ranges[0][0],
                              ranges[1][1] - ranges[1][0]);
  std::map<int, int> aliases;
  std::set<std::pair<int, int>> checked;
  auto node = [&](int id) {
    while (aliases.count(id))
      id = aliases.at(id);
    return id;
  };
  auto compare = [&](const Node &a, const Node &z) {
    if (a.kind != "junction" || z.kind != "junction" || a.pt_nullity != 0 ||
        z.pt_nullity != 0 ||
        engine.canonical_ids(a.assemblage) !=
            engine.canonical_ids(z.assemblage) ||
        ((diagram_coordinates(a, result.section.type) -
          diagram_coordinates(z, result.section.type))
             .cwiseQuotient(range))
                .norm() > engine.settings.node_tolerance)
      return;
    if (!checked.emplace(std::min(a.id, z.id), std::max(a.id, z.id)).second)
      return;
    if (node(a.id) == node(z.id))
      return;
    try {
      auto first = junction_states(result.boundaries, a),
           last = junction_states(result.boundaries, z);
      if (first.size() != a.assemblage.size() ||
          last.size() != z.assemblage.size())
        return;
      auto equivalent = [&](const PhaseStates &left_seed,
                            const PhaseStates &right_seed) {
        auto left = junction_point_at(engine, a, first, left_seed,
                                      range * engine.settings.node_tolerance),
             right = junction_point_at(engine, z, last, right_seed,
                                       range * engine.settings.node_tolerance);
        if (!left || !right ||
            ((diagram_coordinates(*left, result.section.type) -
              diagram_coordinates(*right, result.section.type))
                 .cwiseAbs()
                 .array() >
             equilibrium_coordinate_tolerance(result.section.type, range)
                 .array())
                .any() ||
            (Eigen::Vector2d(left->pressure - right->pressure,
                             left->temperature - right->temperature)
                 .cwiseAbs()
                 .array() > equilibrium_pt_tolerance().array())
                .any() ||
            !same_phase_compositions(left->phases, right->phases,
                                     engine.settings.composition_tolerance,
                                     engine.settings.amount_tolerance))
          return false;
        return true;
      };
      auto compatible = first;
      const bool matched =
          seed_phase_compositions(compatible, last,
                                  engine.settings.composition_tolerance,
                                  engine.settings.amount_tolerance) &&
          (equivalent(first, first) || equivalent(last, last));
      if (!matched) {
        // Different active faces can hide one equilibrium junction. Open their
        // union, retaining each limiting state's own compositions; joining
        // still requires two independent roots at the existing accuracy.
        if (!share_composition_faces(first, last) || !equivalent(first, last))
          return;
      }
      const int x = node(a.id), y = node(z.id);
      if (x != y)
        aliases[std::max(x, y)] = std::min(x, y);
    } catch (const std::exception &error) {
      if (engine.settings.verbose)
        std::cerr << "Junction comparison: " << error.what() << '\n';
    }
  };
  // Separate recovered arms need not have a saved bridge or the same imposed
  // zero phases. The independent roots above, including plotted S/V, decide
  // whether these are two representations of the same equilibrium event.
  const auto active = active_boundary_nodes(result);
  for (auto first = active.begin(); first != active.end(); ++first)
    for (auto last = std::next(first); last != active.end(); ++last)
      compare(result.nodes.at(static_cast<std::size_t>(*first)),
              result.nodes.at(static_cast<std::size_t>(*last)));
  std::set<int> contracted;
  for (auto &b : result.boundaries) {
    const bool bridge = b.start_node != b.end_node;
    b.start_node = node(b.start_node);
    b.end_node = node(b.end_node);
    if (bridge && b.start_node >= 0 && b.start_node == b.end_node)
      contracted.insert(b.id);
  }
  erase_if(result.boundaries, [&](const Boundary &b) {
    return contracted.count(b.id) &&
           std::all_of(b.points.begin(), b.points.end(), [&](const auto &p) {
             return ((diagram_coordinates(p, result.section.type) -
                      diagram_coordinates(b.points.front(),
                                          result.section.type))
                         .cwiseQuotient(range))
                        .norm() <= engine.settings.node_tolerance;
           });
  });
}
// Curved traces can pass a saved stable node between corrector points. Locate
// it with an equilibrium solve before splitting; chord proximity is only a
// search bound. Different zero constraints can describe one univariant curve.
std::set<int> split_at_stable_nodes(Result &result, Engine &engine) {
  std::set<int> changed;
  if (has_composition_axis(result.section.type))
    return changed;
  const auto active = active_boundary_nodes(result);
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto normalise = [&](const auto &p) -> Eigen::Vector2d {
    return (diagram_coordinates(p, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  std::map<int, std::optional<BoundaryPoint>> equilibria;
  auto original = std::move(result.boundaries);
  for (const auto &line : original) {
    const int variance = boundary_variance(line, result);
    const auto phases = engine.canonical_ids(line.assemblage);
    Boundary arc = line;
    arc.points.clear();
    auto append = [&]() {
      arc.id = burnman::utils::checked_int(result.boundaries.size());
      result.boundaries.push_back(arc);
    };
    if (line.points.size() < 2 || line.zero_phase < 0) {
      arc.points = line.points;
      append();
      continue;
    }
    arc.points.push_back(line.points.front());
    for (std::size_t k = 1; k < line.points.size(); ++k) {
      const auto a = normalise(line.points[k - 1]),
                 b = normalise(line.points[k]);
      const Eigen::Vector2d delta = b - a;
      std::vector<std::pair<double, const Node *>> cuts;
      for (const auto &node : result.nodes) {
        if (!active.count(node.id) || node.id == arc.start_node ||
            node.id == line.start_node || node.id == line.end_node ||
            node.kind != "junction" ||
            (node.pt_nullity != 0 && result.section.type != DiagramType::PT) ||
            delta.squaredNorm() == 0.)
          continue;
        const auto event = engine.canonical_ids(node.assemblage);
        // A P-T invariant can span a latent S/V interval. Different zero
        // amounts then describe overlapping parts of the same plotted edge.
        // The constrained solve below, rather than the Gibbs phase count,
        // establishes whether this node actually lies on the curve.
        const bool same_reaction =
            result.section.type == DiagramType::PT
                ? variance == 1 && node.gibbs_variance >= 0 &&
                      node.gibbs_variance +
                              static_cast<int>(node.assemblage.size()) -
                              static_cast<int>(line.assemblage.size()) ==
                          1
                : event == phases;
        const bool same_zero = std::any_of(
            node.zero_phases.begin(), node.zero_phases.end(), [&](int zero) {
              return zero / engine.settings.max_phase_instances ==
                     line.zero_phase / engine.settings.max_phase_instances;
            });
        if ((!same_reaction && !same_zero) ||
            !std::includes(event.begin(), event.end(), phases.begin(),
                           phases.end()))
          continue;
        const auto p = normalise(node);
        const double along = (p - a).dot(delta) / delta.squaredNorm();
        if (along > 0. && along <= 1. &&
            segment_distance(p, a, b) <=
                std::max(engine.settings.node_tolerance, .5 * delta.norm()))
          cuts.emplace_back(along, &node);
      }
      std::sort(cuts.begin(), cuts.end(),
                [](const auto &x, const auto &y) { return x.first < y.first; });
      for (const auto &[along, node] : cuts)
        try {
          // Compare corrected equilibria, not a chord projection against
          // a loosely located saved junction. Reuse each junction solve.
          auto [entry, inserted] = equilibria.try_emplace(node->id);
          if (inserted) {
            auto states = junction_states(original, *node);
            entry->second =
                junction_point_at(engine, *node, states, states,
                                  range * engine.settings.node_tolerance);
          }
          if (!entry->second)
            continue;
          const auto &junction = *entry->second;
          std::optional<BoundaryPoint> point;
          for (const auto *warm : {&line.points[k - 1], &line.points[k]}) {
            auto states = warm->phases;
            erase_if(states, [&](const auto &p) {
              return std::find(line.assemblage.begin(), line.assemblage.end(),
                               p.id) == line.assemblage.end();
            });
            // Compare independent zero constraints on the same intensive
            // seed, including active solution faces. Preserve each trace's
            // instance IDs and amounts so reaction extents remain distinct.
            // A zero-amount solution still identifies this reaction branch.
            // Do not force it into another minimum merely to reach the node.
            if (!seed_phase_compositions(states, junction.phases) ||
                !same_solution_branches(engine, warm->phases, states,
                                        junction.pressure,
                                        junction.temperature))
              continue;
            point = boundary_point_at(
                engine, line, states,
                diagram_coordinates(junction, result.section.type),
                delta.normalized().cwiseQuotient(range),
                equilibrium_coordinate_tolerance(result.section.type, range));
            if (point &&
                same_phase_compositions(point->phases, junction.phases,
                                        engine.settings.composition_tolerance,
                                        engine.settings.amount_tolerance))
              break;
            point.reset();
          }
          if (!point)
            continue;
          arc.points.push_back(*point);
          arc.end_node = node->id;
          const auto [back, forward] = boundary_statuses(line);
          arc.termination =
              (arc.start_node == line.start_node ? back : "junction") +
              "; junction";
          append();
          if (result.section.type != DiagramType::PT || variance == 1)
            changed.insert(arc.id);
          arc.start_node = node->id;
          arc.points = {std::move(*point)};
        } catch (const std::exception &) {
          // Retain the original arc if this node cannot be verified on it.
        }
      arc.points.push_back(line.points[k]);
    }
    arc.end_node = line.end_node;
    arc.termination = arc.start_node == line.start_node
                          ? line.termination
                          : "junction; " + boundary_statuses(line).second;
    append();
    if (arc.start_node != line.start_node &&
        (result.section.type != DiagramType::PT || variance == 1))
      changed.insert(arc.id);
  }
  return changed;
}

// A finite strip between the same stable endpoints can acquire negative area
// when its two curved sides use different chords. Sample both branches at
// the same coordinate values only for these inconsistent two-arc loops.
void retain_thin_loops(Result &result, Engine &engine) {
  if (has_composition_axis(result.section.type))
    return;
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto q = [&](const auto &p) -> Eigen::Vector2d {
    return (diagram_coordinates(p, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  std::map<std::pair<int, int>, std::vector<std::size_t>> pairs;
  for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
    const auto &b = result.boundaries[i];
    if (b.start_node >= 0 && b.end_node >= 0 && b.start_node != b.end_node &&
        b.points.size() > 1 && !univariant_reaction(b, result))
      pairs[std::minmax(b.start_node, b.end_node)].push_back(i);
  }
  for (const auto &[ends, lines] : pairs)
    for (std::size_t i = 0; i < lines.size(); ++i)
      for (std::size_t j = i + 1; j < lines.size(); ++j) {
        auto &a = result.boundaries[lines[i]], &b = result.boundaries[lines[j]];
        if (a.zero_phase == b.zero_phase)
          continue;
        const bool forward = a.start_node == b.start_node;
        auto ids = [&](const auto &side) { return engine.canonical_ids(side); };
        const auto left_a = ids(a.side_b), right_a = ids(a.side_a),
                   left_b = ids(forward ? b.side_b : b.side_a),
                   right_b = ids(forward ? b.side_a : b.side_b);
        const bool left = !left_a.empty() && left_a == right_b,
                   right = !right_a.empty() && right_a == left_b;
        if (left == right)
          continue;
        auto integral = [&](const Boundary &line) {
          double value = 0.;
          for (std::size_t k = 1; k < line.points.size(); ++k)
            value += cross(q(line.points[k - 1]), q(line.points[k]));
          return value;
        };
        if ((left ? 1. : -1.) *
                (integral(a) - (forward ? 1. : -1.) * integral(b)) >
            0.)
          continue;
        Eigen::Index axis;
        (q(a.points.back()) - q(a.points.front())).cwiseAbs().maxCoeff(&axis);
        const double tolerance =
            equilibrium_coordinate_tolerance(result.section.type, range)[axis];
        auto value = [&](const auto &p) {
          return diagram_coordinates(p, result.section.type)[axis];
        };
        auto monotone = [&](const Boundary &line) {
          double sign =
              value(line.points.back()) > value(line.points.front()) ? 1. : -1.;
          for (std::size_t k = 1; k < line.points.size(); ++k)
            if (sign * (value(line.points[k]) - value(line.points[k - 1])) <
                -tolerance)
              return false;
          return true;
        };
        if (!monotone(a) || !monotone(b))
          continue;
        std::set<double> values;
        for (const auto *line : {&a, &b})
          for (const auto &p : line->points)
            values.insert(value(p));
        for (auto *line : {&a, &b}) {
          auto original = std::move(line->points);
          line->points = {original.front()};
          for (std::size_t k = 1; k < original.size(); ++k) {
            const auto p = diagram_coordinates(original[k - 1],
                                               result.section.type),
                       r = diagram_coordinates(original[k],
                                               result.section.type);
            std::vector<double> cuts;
            for (double v : values)
              if (v > std::min(p[axis], r[axis]) + tolerance &&
                  v < std::max(p[axis], r[axis]) - tolerance)
                cuts.push_back(v);
            if (r[axis] < p[axis])
              std::reverse(cuts.begin(), cuts.end());
            for (double v : cuts)
              try {
                const double fraction = (v - p[axis]) / (r[axis] - p[axis]);
                const Eigen::Vector2d target = p + fraction * (r - p);
                Eigen::Vector2d weights = Eigen::Vector2d::Zero(),
                                allowance = (r - p).cwiseAbs();
                weights[axis] = 1. / range[axis];
                allowance[axis] = tolerance;
                if (auto point =
                        boundary_point_at(engine, *line, original[k - 1].phases,
                                          target, weights, allowance))
                  line->points.push_back(std::move(*point));
              } catch (const std::exception &) {
              }
            line->points.push_back(original[k]);
          }
        }
      }
}
} // namespace

std::set<int> active_boundary_nodes(const Result &result) {
  std::set<int> nodes;
  for (const auto &line : result.boundaries)
    if (line.points.size() >= 2)
      for (int id : {line.start_node, line.end_node})
        if (id >= 0)
          nodes.insert(id);
  // Geometry-only files may specify junctions without endpoint references.
  if (nodes.empty())
    for (const auto &node : result.nodes)
      nodes.insert(node.id);
  return nodes;
}

int boundary_variance(const Boundary &line, const Result &result) {
  int variance = -1;
  const auto assemblage =
      canonical_phase_ids(line.assemblage, result.settings.max_phase_instances);
  for (const auto &node : result.nodes)
    if ((node.id == line.start_node || node.id == line.end_node) &&
        !node.kind.empty() && !node.assemblage.empty() &&
        node.gibbs_variance >= 0) {
      const auto phases = canonical_phase_ids(
          node.assemblage, result.settings.max_phase_instances);
      if (!std::includes(phases.begin(), phases.end(), assemblage.begin(),
                         assemblage.end()))
        continue;
      const int next = node.gibbs_variance +
                       burnman::utils::checked_int(node.assemblage.size()) -
                       burnman::utils::checked_int(line.assemblage.size());
      variance = variance < 0 ? next : std::min(variance, next);
    }
  return variance;
}

bool univariant_reaction(const Boundary &line, const Result &result) {
  const auto full =
      canonical_phase_ids(line.assemblage, result.settings.max_phase_instances);
  return result.section.type == DiagramType::PT &&
         ((line.reaction_kind == ReactionKind::replacement &&
           canonical_phase_ids(line.side_a,
                               result.settings.max_phase_instances) != full &&
           canonical_phase_ids(line.side_b,
                               result.settings.max_phase_instances) != full) ||
          boundary_variance(line, result) == 1);
}

void repair_field_boundaries(
    Result &result, Engine &engine,
    const std::function<void(Boundary &)> &label,
    const std::function<bool(Boundary &, std::size_t)> &refine) {
  const auto ranges = result.coordinate_ranges();
  for (auto &line : result.boundaries)
    trim_boundary_endpoints(
        line, engine,
        {ranges[0][1] - ranges[0][0], ranges[1][1] - ranges[1][0]});
  join_duplicate_junctions(result, engine);
  auto refresh_conflicting_labels = [&]() {
    std::set<int> conflicting;
    const auto graph = recover_boundary_labels(result);
    for (const auto &polygon : graph.polygons)
      if (polygon.is_provisional && !polygon.has_open_boundary &&
          !polygon.outside_model_domain)
        conflicting.insert(polygon.boundary_ids.begin(),
                           polygon.boundary_ids.end());
    // A stale label can order a neighbouring arc into a spur and hide it
    // from the perimeter. Refresh all reactions at the affected junctions.
    std::set<int> nodes;
    for (const auto &line : result.boundaries)
      if (conflicting.count(line.id))
        for (int id : {line.start_node, line.end_node})
          if (id >= 0)
            nodes.insert(id);
    for (auto &line : result.boundaries)
      if ((conflicting.count(line.id) || nodes.count(line.start_node) ||
           nodes.count(line.end_node)) &&
          !ambiguous_boundary(line))
        label(line);
  };
  refresh_conflicting_labels();
  auto selected = split_at_stable_nodes(result, engine);
  if (!selected.empty()) {
    // Splitting a recovered arc can expose a short bridge between two
    // representations of the same junction. Verify those new connections
    // before interpreting them as separate entropy/volume vertices.
    join_duplicate_junctions(result, engine);
    std::set<int> renumbered;
    for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
      auto &line = result.boundaries[i];
      const int id = burnman::utils::checked_int(i);
      if (selected.count(line.id))
        renumbered.insert(id);
      line.id = id;
    }
    selected = std::move(renumbered);
  }
  // A curve can pass through a verified junction and change limiting states
  // without creating a conflicting polygon label. Split and relabel those
  // segments before declaring the graph consistent.
  if (result.section.type == DiagramType::PT) {
    for (auto &line : result.boundaries)
      if (selected.count(line.id))
        label(line);
    rebuild_boundary_incidence(result);
    if (field_diagnostics(result, engine, field_polygons(result)).empty())
      return;
  }
  // A missing side label can itself reverse a thin field's graph fan. Recover
  // that thermodynamic evidence before treating its drawing as inconsistent.
  // This also avoids refining valid curves merely because a normal probe had
  // skipped their immediate neighbour.
  for (auto &line : result.boundaries)
    if (ambiguous_boundary(line) && !lies_on_calculation_frame(line, result))
      label(line);
  // Correct stale thermodynamic evidence before interpreting a conflicting
  // perimeter as a reason to subdivide every accepted segment.
  refresh_conflicting_labels();
  retain_thin_loops(result, engine);
  // Verify suspect drawn loops locally. Correctors reuse the ordinary branch
  // validation; intersections themselves never create graph connections.
  std::set<int> curved;
  for (const auto &polygon : field_polygons(result).polygons) {
    if (has_composition_axis(result.section.type) || !polygon.is_provisional ||
        (!polygon.phases.empty() && result.section.type != DiagramType::PT) ||
        polygon.has_open_boundary || polygon.outside_model_domain)
      continue;
    if (result.section.type == DiagramType::PT && !polygon.phases.empty()) {
      curved.insert(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
      std::set<int> perimeter_nodes;
      for (int id : polygon.boundary_ids) {
        const auto &line = result.boundaries.at(static_cast<std::size_t>(id));
        for (int node : {line.start_node, line.end_node})
          if (node >= 0)
            perimeter_nodes.insert(node);
      }
      // Coarse chords can turn a separating reaction into an apparent spur,
      // which is then omitted from the polygon's perimeter. Include reactions
      // joining its junctions when correcting the inconsistent drawing.
      for (const auto &line : result.boundaries)
        if (perimeter_nodes.count(line.start_node) &&
            perimeter_nodes.count(line.end_node))
          curved.insert(line.id);
    }
    if (polygon.boundary_ids.size() == 2) {
      const auto &a = result.boundaries.at(
                     static_cast<std::size_t>(polygon.boundary_ids[0])),
                 &b = result.boundaries.at(
                     static_cast<std::size_t>(polygon.boundary_ids[1]));
      if (a.zero_phase == b.zero_phase)
        curved.insert(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
    }
    for (int id : interior_boundaries(result, polygon)) {
      curved.insert(id);
      curved.insert(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
    }
  }
  std::set<int> rejected;
  for (auto &line : result.boundaries)
    if (curved.count(line.id))
      for (std::size_t k = line.points.size(); k-- > 1;)
        if (!refine(line, k))
          rejected.insert(line.id);
  for (const auto &polygon : field_polygons(result).polygons)
    if (polygon.is_provisional && !polygon.outside_model_domain)
      selected.insert(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
  for (auto &line : result.boundaries)
    if (ambiguous_boundary(line) ||
        (!rejected.count(line.id) && selected.count(line.id)))
      try {
        label(line);
      } catch (const std::exception &error) {
        if (engine.settings.verbose)
          std::cerr << "Field-label recovery: " << error.what() << '\n';
      }
}
// A trace can converge on a chord whose assemblage is metastable in its
// interior: the endpoints are stable nodes, so an endpoint test cannot reject
// it, but a lower-energy phase (typically a second solvus copy) appears in
// between. Probe interior points of lines that share both nodes with another
// line. Remove a record only when a corrected equilibrium proves instability
// and a parallel record verifies stable; failed corrections remain unresolved.
std::size_t remove_metastable_parallel_lines(Result &result, Engine &engine) {
  enum class Stability { stable, metastable, unresolved };
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto q = [&](const auto &p) -> Eigen::Vector2d {
    return (diagram_coordinates(p, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  auto stable_interior = [&](const Boundary &line) {
    if (line.points.size() < 2)
      return Stability::unresolved;
    std::vector<double> lengths{0.};
    for (std::size_t k = 1; k < line.points.size(); ++k)
      lengths.push_back(lengths.back() +
                        (q(line.points[k]) - q(line.points[k - 1])).norm());
    if (lengths.back() <= 0.)
      return Stability::unresolved;
    bool verified = true;
    for (double fraction : {.25, .5, .75}) {
      const double s = fraction * lengths.back();
      std::size_t k = 1;
      while (k + 1 < lengths.size() && lengths[k] < s)
        ++k;
      const double span = lengths[k] - lengths[k - 1];
      const double t = span > 0. ? (s - lengths[k - 1]) / span : 0.;
      const Eigen::Vector2d a = q(line.points[k - 1]), b = q(line.points[k]);
      const Eigen::Vector2d tangent = b - a;
      if (tangent.norm() < 1.e-12) {
        verified = false;
        continue;
      }
      const Eigen::Vector2d target =
          origin + (a + t * tangent).cwiseProduct(range);
      try {
        // The small solvus-affinity threshold below requires a more accurate
        // equilibrium than an ordinary continuation step. A failed corrector
        // is unresolved evidence, not proof of metastability.
        auto point = boundary_point_at(
            engine, line, line.points[k - 1].phases, target,
            tangent.normalized().cwiseQuotient(range), range * lengths.back(),
            engine.direct_coordinates() ? 1. : 1.e-4, false);
        // Competing arcs already share verified endpoints. Resolve small
        // solvus affinities against the alternative stable arc, even when
        // both lie inside the ordinary phase-discovery affinity tolerance.
        const double tolerance =
            engine.direct_coordinates()
                ? engine.settings.affinity_tolerance
                : std::min(engine.settings.affinity_tolerance, 1.e-5);
        if (!point) {
          verified = false;
        } else if (point->minimum_affinity < -tolerance)
          return Stability::metastable;
      } catch (const std::exception &) {
        verified = false;
      }
    }
    return verified ? Stability::stable : Stability::unresolved;
  };
  std::vector<Stability> states(result.boundaries.size(),
                                Stability::unresolved);
  std::vector<char> probed(result.boundaries.size(), 0);
  auto status = [&](std::size_t i) {
    if (!probed[i]) {
      probed[i] = 1;
      states[i] = stable_interior(result.boundaries[i]);
    }
    return states[i];
  };
  std::set<std::size_t> drop;
  for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
    const auto &line = result.boundaries[i];
    if (line.start_node < 0 || line.end_node < 0 ||
        line.start_node == line.end_node)
      continue;
    std::vector<std::size_t> peers;
    for (std::size_t j = 0; j < result.boundaries.size(); ++j) {
      const auto &other = result.boundaries[j];
      if (j != i && other.assemblage != line.assemblage &&
          ((other.start_node == line.start_node &&
            other.end_node == line.end_node) ||
           (other.start_node == line.end_node &&
            other.end_node == line.start_node)))
        peers.push_back(j);
    }
    if (peers.empty() || status(i) != Stability::metastable)
      continue;
    if (std::any_of(peers.begin(), peers.end(), [&](std::size_t j) {
          return status(j) == Stability::stable;
        }))
      drop.insert(i);
  }
  if (drop.empty())
    return 0;
  std::vector<Boundary> kept;
  for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
    if (drop.count(i)) {
      if (engine.settings.verbose)
        std::cerr << "Dropping metastable parallel boundary "
                  << result.boundaries[i].id << '\n';
      continue;
    }
    kept.push_back(std::move(result.boundaries[i]));
  }
  result.boundaries = std::move(kept);
  return drop.size();
}
} // namespace burnman::phase_diagrams::detail
