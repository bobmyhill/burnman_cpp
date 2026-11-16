// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/tools/phase_diagram/field_verification.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace burnman::phase_diagrams {
namespace detail {

std::string field_verification_signature(const Result &result) {
  // Bind a certificate to the complete saved reaction graph, including other
  // lines whose later insertion could invalidate the crossing check. Exclude
  // derived diagnostics and samples. The hash is an invalidation token, not a
  // cryptographic signature or a substitute for thermodynamic verification.
  std::uint64_t hash = 14695981039346656037ULL;
  auto word = [&](std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      hash ^= value & 255U;
      hash *= 1099511628211ULL;
      value >>= 8;
    }
  };
  auto number = [&](double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    word(bits);
  };
  auto ids = [&](const auto &values) {
    word(values.size());
    for (int value : values)
      word(static_cast<std::uint64_t>(value));
  };
  word(static_cast<std::uint64_t>(result.section.type));
  const std::map<std::string, double> ordered_bulk(
      result.composition_start.begin(), result.composition_start.end());
  for (const auto &[element, amount] : ordered_bulk) {
    word(element.size());
    for (char letter : element)
      word(static_cast<unsigned char>(letter));
    number(amount);
  }
  for (const auto &name : result.phase_names) {
    word(name.size());
    for (char letter : name)
      word(static_cast<unsigned char>(letter));
  }
  for (const auto &range : result.coordinate_ranges())
    for (double value : range)
      number(value);
  number(result.settings.affinity_tolerance);
  number(result.settings.mass_balance_tolerance);
  number(result.settings.amount_tolerance);
  number(result.settings.composition_tolerance);
  word(static_cast<std::uint64_t>(result.settings.max_phase_instances));
  for (const auto &line : result.boundaries) {
    word(static_cast<std::uint64_t>(line.id));
    word(static_cast<std::uint64_t>(line.zero_phase));
    word(static_cast<std::uint64_t>(line.start_node));
    word(static_cast<std::uint64_t>(line.end_node));
    word(static_cast<std::uint64_t>(line.reaction_kind));
    ids(line.assemblage);
    ids(line.side_a);
    ids(line.side_b);
    word(line.points.size());
    for (const auto &point : line.points) {
      number(point.pressure);
      number(point.temperature);
    }
  }
  for (const auto &node : result.nodes) {
    word(static_cast<std::uint64_t>(node.id));
    number(node.pressure);
    number(node.temperature);
    word(static_cast<std::uint64_t>(node.gibbs_variance));
    word(static_cast<std::uint64_t>(node.pt_nullity));
    word(node.kind.size());
    for (char letter : node.kind)
      word(static_cast<unsigned char>(letter));
    ids(node.assemblage);
    ids(node.zero_phases);
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

void verify_field_perimeters(Result &result, Engine &engine,
                             bool recheck_closed_rings) {
  if (result.section.type != DiagramType::PT)
    return;
  const auto geometry = field_polygons(result);
  result.field_verifications.clear();
  const auto signature = field_verification_signature(result);
  const Eigen::Vector2d range(
      result.pressure_range[1] - result.pressure_range[0],
      result.temperature_range[1] - result.temperature_range[0]);
  const auto accuracy = equilibrium_pt_tolerance();
  const Eigen::Vector2d junction_allowance =
      (range * engine.settings.node_tolerance).cwiseMax(accuracy * 4.);
  std::map<int, const Boundary *> lines;
  std::map<int, const Node *> nodes;
  for (const auto &line : result.boundaries)
    lines.emplace(line.id, &line);
  for (const auto &node : result.nodes)
    nodes.emplace(node.id, &node);
  auto location = [](const auto &point) {
    return Eigen::Vector2d(point.pressure, point.temperature);
  };
  for (const auto &polygon : geometry.polygons) {
    if (polygon.has_closed_reaction_ring && !recheck_closed_rings)
      continue;
    if ((!polygon.is_provisional && !polygon.has_unresolved_interior) ||
        polygon.has_open_boundary || polygon.outside_model_domain ||
        polygon.phases.empty() || polygon.boundary_ids.size() < 2 ||
        !polygon.holes.empty())
      continue;
    FieldVerification certificate;
    certificate.phases = polygon.phases;
    certificate.boundary_ids = polygon.boundary_ids;
    certificate.graph_signature = signature;
    auto record = [&](const BoundaryPoint &point) {
      certificate.maximum_mass_balance_error = std::max(
          certificate.maximum_mass_balance_error, point.mass_balance_error);
      certificate.maximum_reaction_error =
          std::max(certificate.maximum_reaction_error, point.residual);
      certificate.minimum_affinity =
          std::min(certificate.minimum_affinity, point.minimum_affinity);
    };
    try {
      std::set<int> perimeter(polygon.boundary_ids.begin(),
                              polygon.boundary_ids.end());
      std::map<int, int> degree;
      for (int id : perimeter) {
        const auto &line = *lines.at(id);
        if (line.points.size() < 2 || line.start_node < 0 ||
            line.end_node < 0 || line.start_node == line.end_node ||
            line.reaction_kind == ReactionKind::unknown ||
            (engine.canonical_ids(line.side_a) != polygon.phases &&
             engine.canonical_ids(line.side_b) != polygon.phases))
          throw std::runtime_error(
              "Incomplete or incompatible reaction perimeter");
        ++degree[line.start_node];
        ++degree[line.end_node];
        for (int end : {0, 1}) {
          const int node = end ? line.end_node : line.start_node;
          if (!nodes.count(node) ||
              ((location(end ? line.points.back() : line.points.front()) -
                location(*nodes.at(node)))
                   .cwiseAbs()
                   .array() > junction_allowance.array())
                  .any())
            throw std::runtime_error(
                "Reaction endpoint does not reach its stated common vertex");
          const auto event = engine.canonical_ids(nodes.at(node)->assemblage),
                     assemblage = engine.canonical_ids(line.assemblage);
          if (!std::includes(event.begin(), event.end(), assemblage.begin(),
                             assemblage.end()))
            throw std::runtime_error(
                "Reaction assemblage is incompatible with its common vertex");
        }
      }
      std::set<int> reached;
      std::vector<int> pending{degree.begin()->first};
      while (!pending.empty()) {
        const int node = pending.back();
        pending.pop_back();
        if (!reached.insert(node).second)
          continue;
        for (int boundary_id : perimeter) {
          const auto &line = *lines.at(boundary_id);
          if (line.start_node == node)
            pending.push_back(line.end_node);
          if (line.end_node == node)
            pending.push_back(line.start_node);
        }
      }
      if (reached.size() != degree.size())
        throw std::runtime_error(
            "Perimeter does not form one connected reaction loop");
      for (const auto &[id, count] : degree) {
        if (count != 2 || !nodes.count(id))
          throw std::runtime_error(
              "Perimeter reactions do not meet at common vertices");
        certificate.node_ids.push_back(id);
        const auto &node = *nodes.at(id);
        if (node.kind != "junction" || node.zero_phases.size() != 2)
          throw std::runtime_error(
              "Perimeter vertex lacks double-zero junction constraints");
        auto seed = junction_states(result.boundaries, node);
        auto point =
            junction_point_at(engine, node, seed, seed, junction_allowance);
        if (!point)
          throw std::runtime_error(
              "Common reaction vertex did not verify thermodynamically");
        record(*point);
        ++certificate.checked_junctions;
        // A saved node is an estimate, not an exactly rounded P,T equality.
        // Independently correct each incident reaction onto the re-solved
        // junction. Geometric proximity alone cannot establish a shared root.
        for (int boundary_id : perimeter) {
          const auto &line = *lines.at(boundary_id);
          if (line.start_node != node.id && line.end_node != node.id)
            continue;
          auto warm = (line.start_node == node.id ? line.points.front()
                                                  : line.points.back())
                          .phases;
          erase_if(warm, [&](const auto &phase) {
            return std::find(line.assemblage.begin(), line.assemblage.end(),
                             phase.id) == line.assemblage.end();
          });
          const auto original = warm;
          if (!seed_phase_compositions(warm, point->phases) ||
              !same_solution_branches(engine, original, warm, point->pressure,
                                      point->temperature))
            throw std::runtime_error(
                "Reaction branch cannot be matched to its common vertex");
          auto endpoint =
              boundary_point_at(engine, line, warm, location(*point),
                                {0., 1. / range[1]}, accuracy * 4.);
          if (!endpoint ||
              !same_phase_compositions(endpoint->phases, point->phases,
                                       engine.settings.composition_tolerance,
                                       engine.settings.amount_tolerance))
            throw std::runtime_error(
                "Incident reaction does not verify at its common vertex");
          record(*endpoint);
          ++certificate.checked_boundary_points;
        }
      }
      auto correct = [&](const Boundary &line, const Eigen::Vector2d &target,
                         const Eigen::Vector2d &weights,
                         const Eigen::Vector2d &allowance) {
        std::vector<const BoundaryPoint *> starts;
        for (const auto &point : line.points)
          starts.push_back(&point);
        std::sort(starts.begin(), starts.end(), [&](auto a, auto b) {
          return ((location(*a) - target).cwiseQuotient(range)).squaredNorm() <
                 ((location(*b) - target).cwiseQuotient(range)).squaredNorm();
        });
        for (std::size_t i = 0; i < std::min<std::size_t>(4, starts.size());
             ++i)
          if (auto point = boundary_point_at(engine, line, starts[i]->phases,
                                             target, weights, allowance)) {
            record(*point);
            ++certificate.checked_boundary_points;
            return *point;
          }
        throw std::runtime_error(
            "Bounding reaction failed equilibrium or stability checks");
      };
      for (int id : perimeter) {
        const auto &line = *lines.at(id);
        std::vector<double> lengths{0.};
        for (std::size_t k = 1; k < line.points.size(); ++k)
          lengths.push_back(lengths.back() + ((location(line.points[k]) -
                                               location(line.points[k - 1]))
                                                  .cwiseQuotient(range))
                                                 .norm());
        for (double fraction : {.25, .5, .75}) {
          const double position = lengths.back() * fraction;
          std::size_t k = 1;
          while (k + 1 < lengths.size() && lengths[k] < position)
            ++k;
          const Eigen::Vector2d first = location(line.points[k - 1]),
                                last = location(line.points[k]),
                                delta = last - first;
          const double span = lengths[k] - lengths[k - 1];
          if (span <= 0.)
            throw std::runtime_error("Zero-length reaction segment");
          correct(line, first + (position - lengths[k - 1]) / span * delta,
                  delta.cwiseQuotient(range).normalized().cwiseQuotient(range),
                  (range * lengths.back()).cwiseMax(accuracy * 4.));
        }
      }
      // A crossing of chords is a candidate, never a new physical vertex.
      // Re-solve both curves on common isotherms over the overlapping segment
      // interval. A change of ordering, coincident roots, or failed correction
      // leaves the label provisional. This checks existing lines as well as
      // the two sides of a thin strip, without imposing a minimum field width.
      for (int id : perimeter) {
        const auto &first = *lines.at(id);
        for (const auto &second : result.boundaries) {
          if (perimeter.count(second.id) && second.id < id)
            continue;
          for (std::size_t i = 1; i < first.points.size(); ++i)
            for (std::size_t j = 1; j < second.points.size(); ++j) {
              if (second.id == id && j <= i + 1)
                continue; // Adjacent segments of one curve share a drawing
                          // point.
              const Eigen::Vector2d a = location(first.points[i - 1]),
                                    b = location(first.points[i]),
                                    c = location(second.points[j - 1]),
                                    d = location(second.points[j]);
              if (((a.cwiseMax(b) - c.cwiseMin(d)).array() < 0.).any() ||
                  ((c.cwiseMax(d) - a.cwiseMin(b)).array() < 0.).any())
                continue;
              const Eigen::Vector2d u = (b - a).cwiseQuotient(range),
                                    v = (d - c).cwiseQuotient(range),
                                    offset = (c - a).cwiseQuotient(range);
              auto shared_vertex_at = [&](const Eigen::Vector2d &point) {
                for (int node : {first.start_node, first.end_node})
                  if (nodes.count(node) &&
                      (node == second.start_node || node == second.end_node) &&
                      ((location(*nodes.at(node)) - point).cwiseAbs().array() <=
                       (accuracy * 4.).array())
                          .all())
                    return true;
                return false;
              };
              const auto intersection = segment_intersection(u, v, offset);
              if (!intersection)
                continue;
              if (intersection->first != intersection->last) {
                if (!shared_vertex_at(
                        a + .5 * (intersection->first + intersection->last) *
                                (b - a))) {
                  ++certificate.checked_crossings;
                  throw std::runtime_error(
                      "Overlapping reaction lines require a separate "
                      "crossing verification");
                }
                continue;
              }
              // Interior drawing points are not junctions: an intersection
              // exactly at a sampled point still requires verification.
              const Eigen::Vector2d crossing =
                  a + intersection->first * (b - a);
              if (shared_vertex_at(crossing))
                continue;
              ++certificate.checked_crossings;
              const double low = std::max(std::min(a[1], b[1]),
                                          std::min(c[1], d[1])),
                           high = std::min(std::max(a[1], b[1]),
                                           std::max(c[1], d[1]));
              if (high - low <= accuracy[1] * 4.)
                throw std::runtime_error(
                    "Possible intervening reaction crossing is unresolved");
              int previous_sign = 0;
              for (double fraction : {.1, .3, .5, .7, .9}) {
                const double temperature = low + fraction * (high - low);
                auto at_temperature = [&](const Boundary &line,
                                          const Eigen::Vector2d &x,
                                          const Eigen::Vector2d &y) {
                  Eigen::Vector2d target =
                      x + (temperature - x[1]) / (y[1] - x[1]) * (y - x);
                  target[1] = temperature;
                  return correct(
                      line, target, Eigen::Vector2d(0., 1. / range[1]),
                      Eigen::Vector2d(std::max(std::abs(y[0] - x[0]), 4.),
                                      accuracy[1]));
                };
                const auto left = at_temperature(first, a, b),
                           right = at_temperature(second, c, d);
                const double difference = left.pressure - right.pressure,
                             roundoff = 32. *
                                        std::numeric_limits<double>::epsilon() *
                                        std::max(std::abs(left.pressure),
                                                 std::abs(right.pressure));
                if (std::abs(difference) <= roundoff)
                  throw std::runtime_error("Reaction crossing cannot be "
                                           "distinguished from coincidence");
                const int sign = difference > 0. ? 1 : -1;
                if (previous_sign && sign != previous_sign)
                  throw std::runtime_error("A bounding reaction passes through "
                                           "another existing reaction");
                previous_sign = sign;
              }
            }
        }
      }
      certificate.success = true;
      certificate.message =
          "Assemblage verified from stable bounding reactions, common "
          "junctions and corrected crossing checks; field width is not "
          "certified";
    } catch (const std::exception &error) {
      certificate.message = error.what();
    }
    result.field_verifications.push_back(std::move(certificate));
  }
}
} // namespace detail

} // namespace burnman::phase_diagrams
