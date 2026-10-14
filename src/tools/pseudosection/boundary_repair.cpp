// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/utils/index_utils.hpp"
#include "internal.hpp"
#include <iostream>
#include <set>

namespace burnman::pseudosections::detail {
namespace {
std::vector<PhaseState> junction_states(const std::vector<Boundary> &lines,
                                        const Node &event,
                                        std::vector<PhaseState> phases = {}) {
  for (const auto &peer : lines)
    if (!peer.points.empty() && peer.assemblage == event.assemblage &&
        (peer.start_node == event.id || peer.end_node == event.id)) {
      phases = (peer.start_node == event.id ? peer.points.front()
                                            : peer.points.back())
                   .phases;
      break;
    }
  for (const auto &peer : lines)
    if (!peer.points.empty() &&
        (peer.start_node == event.id || peer.end_node == event.id))
      for (const auto &p : (peer.start_node == event.id ? peer.points.front()
                                                        : peer.points.back())
                               .phases)
        if (std::none_of(phases.begin(), phases.end(),
                         [&](const auto &q) { return p.id == q.id; }))
          phases.push_back(p);
  phases.erase(std::remove_if(phases.begin(), phases.end(),
                              [&](const auto &p) {
                                return std::find(event.assemblage.begin(),
                                                 event.assemblage.end(),
                                                 p.id) ==
                                       event.assemblage.end();
                              }),
               phases.end());
  return phases;
}

// Proximity proposes duplicate junctions; their independently constrained
// equilibria must agree at the native solver's accuracy before joining them.
void join_duplicate_junctions(Result &result, Engine &engine) {
  if (result.section.type != DiagramType::PT)
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
  for (const auto &b : result.boundaries) {
    if (b.start_node < 0 || b.end_node < 0 || b.start_node == b.end_node ||
        b.points.size() < 2)
      continue;
    const auto &a = result.nodes.at(static_cast<std::size_t>(b.start_node)),
               &z = result.nodes.at(static_cast<std::size_t>(b.end_node));
    if (a.kind != "junction" || z.kind != "junction" || a.pt_nullity != 0 ||
        z.pt_nullity != 0 ||
        engine.canonical_ids(a.assemblage) !=
            engine.canonical_ids(z.assemblage) ||
        ((diagram_coordinates(a, result.section.type) -
          diagram_coordinates(z, result.section.type))
             .cwiseQuotient(range))
                .norm() > engine.settings.node_tolerance)
      continue;
    if (!checked.emplace(std::min(a.id, z.id), std::max(a.id, z.id)).second)
      continue;
    if (node(a.id) == node(z.id))
      continue;
    try {
      auto first =
               junction_states(result.boundaries, a, b.points.front().phases),
           last = junction_states(result.boundaries, z, b.points.back().phases);
      if (first.size() != a.assemblage.size() ||
          last.size() != z.assemblage.size())
        continue;
      auto left = junction_point_at(engine, a, first, first,
                                    range * engine.settings.node_tolerance),
           right = junction_point_at(engine, z, last, first,
                                     range * engine.settings.node_tolerance);
      if (!left || !right ||
          ((diagram_coordinates(*left, result.section.type) -
            diagram_coordinates(*right, result.section.type))
               .cwiseAbs()
               .array() > equilibrium_pt_tolerance().array())
              .any() ||
          !same_phase_compositions(left->phases, right->phases,
                                   engine.settings.composition_tolerance,
                                   engine.settings.amount_tolerance))
        continue;
      const int x = node(a.id), y = node(z.id);
      if (x != y)
        aliases[std::max(x, y)] = std::min(x, y);
    } catch (const std::exception &error) {
      if (engine.settings.verbose)
        std::cerr << "Junction comparison: " << error.what() << '\n';
    }
  }
  std::set<int> contracted;
  for (auto &b : result.boundaries) {
    const bool bridge = b.start_node != b.end_node;
    b.start_node = node(b.start_node);
    b.end_node = node(b.end_node);
    if (bridge && b.start_node >= 0 && b.start_node == b.end_node)
      contracted.insert(b.id);
  }
  result.boundaries.erase(
      std::remove_if(
          result.boundaries.begin(), result.boundaries.end(),
          [&](const Boundary &b) {
            return contracted.count(b.id) &&
                   std::all_of(
                       b.points.begin(), b.points.end(), [&](const auto &p) {
                         return ((diagram_coordinates(p, result.section.type) -
                                  diagram_coordinates(b.points.front(),
                                                      result.section.type))
                                     .cwiseQuotient(range))
                                    .norm() <= engine.settings.node_tolerance;
                       });
          }),
      result.boundaries.end());
}
// Curved traces can pass a saved stable node between corrector points. Locate
// it with an equilibrium solve before splitting; chord proximity is only a
// search bound. Different zero constraints can describe one univariant curve.
std::set<int> split_at_stable_nodes(Result &result, Engine &engine) {
  std::set<int> changed;
  if (result.section.type != DiagramType::PT)
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
            node.kind != "junction" || node.pt_nullity != 0 ||
            delta.squaredNorm() == 0.)
          continue;
        const auto event = engine.canonical_ids(node.assemblage);
        const bool same_reaction =
            node.gibbs_variance == 1 && variance == 1 && event == phases;
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
            states.erase(std::remove_if(
                             states.begin(), states.end(),
                             [&](const auto &p) {
                               return std::find(line.assemblage.begin(),
                                                line.assemblage.end(),
                                                p.id) == line.assemblage.end();
                             }),
                         states.end());
            // Compare independent zero constraints on the same intensive
            // seed, including active solution faces. Preserve each trace's
            // instance IDs and amounts so reaction extents remain distinct.
            if (!seed_phase_compositions(states, junction.phases))
              continue;
            point = boundary_point_at(
                engine, line, states,
                diagram_coordinates(junction, result.section.type),
                delta.normalized().cwiseQuotient(range),
                equilibrium_pt_tolerance());
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
          if (variance == 1)
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
    if (variance == 1 && arc.start_node != line.start_node)
      changed.insert(arc.id);
  }
  return changed;
}

// A finite strip between the same stable endpoints can acquire negative area
// when its two curved sides use different chords. Sample both branches at
// the same coordinate values only for these inconsistent two-arc loops.
void retain_thin_loops(Result &result, Engine &engine) {
  if (result.section.type != DiagramType::PT)
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
        b.points.size() > 1 && boundary_variance(b, result) >= 2)
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
        const double tolerance = axis == 0 ? 1. : 1.e-6;
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
  for (const auto &node : result.nodes)
    if ((node.id == line.start_node || node.id == line.end_node) &&
        !node.kind.empty() && !node.assemblage.empty() &&
        node.gibbs_variance >= 0 &&
        std::includes(node.assemblage.begin(), node.assemblage.end(),
                      line.assemblage.begin(), line.assemblage.end())) {
      const int next = node.gibbs_variance +
                       burnman::utils::checked_int(node.assemblage.size()) -
                       burnman::utils::checked_int(line.assemblage.size());
      variance = variance < 0 ? next : std::min(variance, next);
    }
  return variance;
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
  auto selected = split_at_stable_nodes(result, engine);
  retain_thin_loops(result, engine);
  // Verify suspect drawn loops locally. Correctors reuse the ordinary branch
  // validation; intersections themselves never create graph connections.
  std::set<int> curved;
  for (const auto &polygon : field_polygons(result).polygons) {
    if (result.section.type != DiagramType::PT || !polygon.is_provisional ||
        !polygon.phases.empty() || polygon.has_open_boundary ||
        polygon.outside_model_domain)
      continue;
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
  auto skipped_neighbour = [&](const Boundary &line) {
    if (has_composition_axis(result.section.type))
      return false;
    const bool observed = std::any_of(
        result.fields.begin(), result.fields.end(), [&](const Field &f) {
          return f.phases == engine.canonical_ids(line.assemblage);
        });
    if (!line.points.empty() && observed) {
      const auto formulae = engine.phase_formulae(
          line.assemblage, line.points[line.points.size() / 2].phases);
      Eigen::FullPivLU<Eigen::MatrixXd> amounts(formulae);
      amounts.setThreshold(1.e-9);
      if (amounts.rank() == formulae.rows()) {
        auto absent = line.assemblage;
        absent.erase(std::find(absent.begin(), absent.end(), line.zero_phase));
        auto full = engine.canonical_ids(line.assemblage),
             missing = engine.canonical_ids(absent);
        auto immediate = [&](const std::vector<int> &side) {
          return side == full || side == missing;
        };
        return !immediate(line.side_a) || !immediate(line.side_b);
      }
    }
    const auto copies = std::count_if(
        line.assemblage.begin(), line.assemblage.end(), [&](int id) {
          return id / engine.settings.max_phase_instances ==
                 line.zero_phase / engine.settings.max_phase_instances;
        });
    return copies > 1 && std::min(line.side_a.size(), line.side_b.size()) + 1 <
                             line.assemblage.size();
  };
  for (auto &line : result.boundaries)
    if (line.side_a.empty() || line.side_b.empty() ||
        (line.side_a == line.side_b && !line.is_solution_replacement) ||
        (!rejected.count(line.id) &&
         (skipped_neighbour(line) || selected.count(line.id))))
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
// line, and drop those that fail while a parallel line passes.
std::size_t remove_metastable_parallel_lines(Result &result, Engine &engine) {
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto q = [&](const auto &p) -> Eigen::Vector2d {
    return (diagram_coordinates(p, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  auto stable_interior = [&](const Boundary &line) {
    if (line.points.size() < 2)
      return true;
    std::vector<double> lengths{0.};
    for (std::size_t k = 1; k < line.points.size(); ++k)
      lengths.push_back(lengths.back() +
                        (q(line.points[k]) - q(line.points[k - 1])).norm());
    if (lengths.back() <= 0.)
      return true;
    for (double fraction : {.25, .5, .75}) {
      const double s = fraction * lengths.back();
      std::size_t k = 1;
      while (k + 1 < lengths.size() && lengths[k] < s)
        ++k;
      const double span = lengths[k] - lengths[k - 1];
      const double t = span > 0. ? (s - lengths[k - 1]) / span : 0.;
      const Eigen::Vector2d a = q(line.points[k - 1]), b = q(line.points[k]);
      const Eigen::Vector2d tangent = b - a;
      if (tangent.norm() < 1.e-12)
        continue;
      const Eigen::Vector2d target =
          origin + (a + t * tangent).cwiseProduct(range);
      try {
        if (!boundary_point_at(engine, line, line.points[k - 1].phases, target,
                               tangent.normalized().cwiseQuotient(range),
                               range * lengths.back()))
          return false;
      } catch (const std::exception &) {
        return false;
      }
    }
    return true;
  };
  std::vector<char> failed(result.boundaries.size(), 0), probed(failed);
  auto status = [&](std::size_t i) {
    if (!probed[i]) {
      probed[i] = 1;
      failed[i] = !stable_interior(result.boundaries[i]);
    }
    return !failed[i];
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
    if (peers.empty() || status(i))
      continue;
    if (std::any_of(peers.begin(), peers.end(),
                    [&](std::size_t j) { return status(j); }))
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
} // namespace burnman::pseudosections::detail
