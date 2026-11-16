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

#include "burnman/tools/phase_diagram/diagram_verification.hpp"
#include "burnman/tools/phase_diagram/consistency.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/critical.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/field_verification.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

namespace burnman::phase_diagrams {
namespace detail {
namespace {
bool terminal_critical(Engine &engine, const Node &node, const Boundary &line) {
  const bool replacement = line.is_solution_replacement &&
                           !line.side_a.empty() && line.side_a == line.side_b;
  const bool phase_out = line.reaction_kind == ReactionKind::phase_out;
  if ((!replacement && !phase_out) || line.points.size() < 2)
    return false;
  const auto &endpoint =
      node.id == line.start_node ? line.points.front() : line.points.back();
  const Eigen::Vector2d target(node.pressure, node.temperature);
  const auto accuracy = equilibrium_pt_tolerance();
  if (((diagram_coordinates(endpoint, DiagramType::PT) - target)
           .cwiseAbs()
           .array() > accuracy.array())
          .any())
    return false;
  auto states = endpoint.phases;
  auto zero = find_phase(states, line.zero_phase);
  if (zero == states.end() ||
      node.critical_mode.size() != zero->composition.size() ||
      !node.critical_mode.allFinite() || node.critical_mode.norm() == 0.)
    return false;
  const double total = total_amount(states);
  if (total <= 0. || zero->amount > total * engine.settings.amount_tolerance)
    return false;
  auto survivor =
      std::find_if(states.begin(), states.end(), [&](const auto &p) {
        return p.id != zero->id && p.candidate_index == zero->candidate_index &&
               p.amount > total * engine.settings.amount_tolerance &&
               (p.composition - zero->composition).norm() <=
                   engine.settings.composition_tolerance;
      });
  if (survivor == states.end())
    return false;
  const int other = survivor->id, candidate = survivor->candidate_index;
  survivor->amount += zero->amount;
  states.erase(zero);
  auto ids = line.assemblage;
  auto absent = std::find(ids.begin(), ids.end(), line.zero_phase);
  if (absent == ids.end())
    return false;
  ids.erase(absent);
  auto a = engine.make_assemblage(ids, states, target[0], target[1]);
  if (field_variance(*a, line.assemblage.size()) != node.gibbs_variance ||
      node.gibbs_variance != (replacement ? 1 : 2))
    return false;
  const int phase =
      static_cast<int>(std::find(ids.begin(), ids.end(), other) - ids.begin());
  auto solution = a->get_phase<Solution>(static_cast<std::size_t>(phase));
  if (!solution)
    return false;
  const auto basis = engine.composition_basis(*solution, candidate);
  Eigen::VectorXd direction =
      basis.transpose().completeOrthogonalDecomposition().solve(
          node.critical_mode);
  const auto prm = engine.parameters(*a);
  auto c = constraints(
      std::make_unique<CriticalConstraint>(*a, phase, prm, direction, false),
      std::make_unique<CriticalConstraint>(*a, phase, prm, direction, true));
  auto critical = corrected_point(engine, *a, ids, c, target, accuracy);
  if (!critical ||
      !same_phase_compositions(states, critical->phases,
                               engine.settings.composition_tolerance,
                               engine.settings.amount_tolerance) ||
      !static_cast<CriticalConstraint *>(c[0][0].get())->stable(*a, phase))
    return false;

  // A phase-out line can end when its zero-amount solution copy merges into a
  // surviving copy. It has one arm, unlike a univariant replacement line.
  // The corrected intrinsic critical point and an accepted separated state
  // on that arm certify the termination without inventing a second branch.
  if (phase_out) {
    for (std::size_t j = 1; j < line.points.size(); ++j) {
      const auto &seed = node.id == line.start_node
                             ? line.points[j]
                             : line.points[line.points.size() - 1 - j];
      const auto missing = find_phase(seed.phases, line.zero_phase);
      const auto present = find_phase(seed.phases, other);
      if (missing == seed.phases.end() || present == seed.phases.end() ||
          missing->candidate_index != candidate ||
          (missing->composition - present->composition).norm() <=
              engine.settings.composition_tolerance)
        continue;
      const double amount = total_amount(seed.phases);
      if (amount > 0. &&
          std::abs(missing->amount) <=
              amount * engine.settings.amount_tolerance &&
          present->amount > amount * engine.settings.amount_tolerance) {
        try {
          auto corrected = boundary_point_at(
              engine, line, seed.phases,
              diagram_coordinates(seed, DiagramType::PT), {0., 1.}, accuracy);
          if (corrected &&
              same_phase_compositions(seed.phases, corrected->phases,
                                      engine.settings.composition_tolerance,
                                      engine.settings.amount_tolerance))
            return true;
        } catch (const std::exception &) {
          // A point too close to the critical endpoint can be ill-conditioned;
          // another accepted point along the arm may still certify it.
        }
      }
    }
    return false;
  }

  // On a univariant replacement, the two zero-amount choices are limiting
  // states on one P-T curve. Re-solve both, rather than inferring this from
  // phase names or joining the apparent dangling end to an arbitrary node.
  for (std::size_t j = 1; j < line.points.size(); ++j) {
    const auto &seed = node.id == line.start_node
                           ? line.points[j]
                           : line.points[line.points.size() - 1 - j];
    try {
      const auto location = diagram_coordinates(seed, DiagramType::PT);
      auto left = boundary_point_at(engine, line, seed.phases, location,
                                    {0., 1.}, accuracy);
      if (!left)
        continue;
      auto opposite = line;
      opposite.zero_phase = other;
      auto warm = seed.phases;
      auto first = find_phase(warm, line.zero_phase);
      auto second = find_phase(warm, other);
      if (first == warm.end() || second == warm.end())
        continue;
      std::swap(first->amount, second->amount);
      auto right = boundary_point_at(engine, opposite, warm, location, {0., 1.},
                                     accuracy);
      if (!right ||
          !same_phase_compositions(left->phases, right->phases,
                                   engine.settings.composition_tolerance,
                                   engine.settings.amount_tolerance))
        continue;
      State l, r;
      l.phases = left->phases;
      r.phases = right->phases;
      const auto active = engine.active_phases(l);
      if (active.size() + 1 == line.assemblage.size() &&
          active == engine.active_phases(r) &&
          active == engine.canonical_ids(line.side_a))
        return true;
    } catch (const std::exception &) {
      // A difficult point does not certify an endpoint; try the next saved
      // accepted state, without changing the diagram or its tolerances.
    }
  }
  return false;
}
} // namespace

void verify_critical_endpoints(Result &result, Engine &engine) {
  for (auto &node : result.nodes) {
    node.is_verified_critical_endpoint =
        false; // Never reuse a stale certificate.
    if (result.section.type != DiagramType::PT ||
        node.kind != "critical_point" ||
        (node.gibbs_variance != 1 && node.gibbs_variance != 2) ||
        !node.critical_mode.size())
      continue;
    bool verified = true, found = false;
    for (const auto &line : result.boundaries)
      if (line.start_node == node.id || line.end_node == node.id) {
        found = true;
        try {
          verified = verified && terminal_critical(engine, node, line);
        } catch (const std::exception &error) {
          verified = false;
          if (engine.settings.verbose)
            std::cerr << "Critical endpoint " << node.id << ": " << error.what()
                      << '\n';
        }
      }
    node.is_verified_critical_endpoint = found && verified;
    // An intrinsic critical state need not terminate the plotted phase front.
    // Two separately verified phase-out arms can pass through the coalescence
    // while exchanging which solution copy has zero amount. Replacement
    // limits (variance one) can instead be two records of one terminal arm.
    if (node.is_verified_critical_endpoint && node.gibbs_variance == 2) {
      const auto ranges = result.coordinate_ranges();
      const Eigen::Vector2d range(ranges[0][1] - ranges[0][0],
                                  ranges[1][1] - ranges[1][0]);
      auto direction = [&](const Boundary &line) -> Eigen::Vector2d {
        for (std::size_t k = 1; k < line.points.size(); ++k) {
          const auto &point = node.id == line.start_node
                                  ? line.points[k]
                                  : line.points[line.points.size() - 1 - k];
          const Eigen::Vector2d delta(point.pressure - node.pressure,
                                      point.temperature - node.temperature);
          if ((delta.cwiseAbs().array() >
               equilibrium_pt_tolerance().array() * 4.)
                  .any())
            return delta.cwiseQuotient(range).normalized();
        }
        return Eigen::Vector2d::Zero();
      };
      bool continued = false;
      std::vector<std::pair<const Boundary *, Eigen::Vector2d>> arms;
      for (const auto &line : result.boundaries)
        if ((line.start_node == node.id || line.end_node == node.id) &&
            line.reaction_kind == ReactionKind::phase_out) {
          const auto current = direction(line);
          // Compare only with earlier arms: each pair and direction is used
          // without repeating the endpoint search for the reverse pair.
          for (const auto &[other, previous] : arms)
            if (other->id != line.id &&
                engine.canonical_ids(other->assemblage) ==
                    engine.canonical_ids(line.assemblage) &&
                same_boundary_sides(*other, line) &&
                previous.dot(current) < -.5)
              continued = true;
          arms.emplace_back(&line, current);
        }
      if (continued) {
        node.kind = "boundary_continuation";
        node.is_verified_critical_endpoint = false;
        for (auto &line : result.boundaries) {
          auto [back, forward] = boundary_statuses(line);
          if (line.start_node == node.id)
            back = "boundary continuation";
          if (line.end_node == node.id)
            forward = "boundary continuation";
          line.termination = back + "; " + forward;
        }
      }
    }
    if (engine.settings.verbose)
      std::cerr << "Critical endpoint " << node.id << ": "
                << (node.is_verified_critical_endpoint
                        ? "verified critical termination"
                        : "not certified")
                << '\n';
  }
}

FieldPolygons recover_boundary_labels(Result &result) {
  auto geometry = field_polygons(result);
  for (auto &line : result.boundaries)
    if (const auto it = geometry.boundary_sides.find(line.id);
        it != geometry.boundary_sides.end()) {
      line.side_a = it->second[0];
      line.side_b = it->second[1];
    }
  return geometry;
}

void refresh_diagram_diagnostics(Result &result,
                                 const FieldPolygons &geometry) {
  std::set<int> verified_interiors;
  for (const auto &polygon : geometry.polygons)
    if (!polygon.is_provisional &&
        (polygon.has_verified_reaction_perimeter ||
         polygon.has_closed_reaction_ring || polygon.sample_index >= 0))
      verified_interiors.insert(polygon.interior_failure_indices.begin(),
                                polygon.interior_failure_indices.end());
  erase_if(result.coordinate_diagnostics, [](const Diagnostic &diagnostic) {
    return diagnostic.category == DiagnosticCategory::field ||
           diagnostic.category == DiagnosticCategory::sample;
  });
  result.coordinate_diagnostics.insert(result.coordinate_diagnostics.end(),
                                       geometry.coordinate_diagnostics.begin(),
                                       geometry.coordinate_diagnostics.end());
  erase_if(result.diagnostics, [](const Diagnostic &diagnostic) {
    return diagnostic.category == DiagnosticCategory::sample ||
           diagnostic.category == DiagnosticCategory::trace;
  });
  // Repair can remove or reindex traces; regenerate their messages from the
  // retained graph instead of carrying stale boundary IDs through the passes.
  for (const auto &line : result.boundaries) {
    const auto [back, forward] = boundary_statuses(line);
    if (unfinished_boundary(back) || unfinished_boundary(forward))
      result.diagnostics.push_back(
          {DiagnosticCategory::trace, line.id, line.termination});
  }
  for (std::size_t index = 0; index < result.samples.size(); ++index) {
    const auto &state = result.samples[index];
    if (state.success || state.outside_model_domain)
      continue;
    auto &messages = verified_interiors.count(static_cast<int>(index))
                         ? result.coordinate_diagnostics
                         : result.diagnostics;
    std::ostringstream message;
    message << "Unresolved seed P=" << state.pressure
            << " Pa, T=" << state.temperature << " K: " << state.message;
    messages.push_back(
        {DiagnosticCategory::sample, static_cast<int>(index), message.str()});
  }
}

void erase_field_diagnostics(Result &result) {
  // Rebuild only graph-owned categories; wording never determines ownership.
  erase_if(result.diagnostics, [](const Diagnostic &diagnostic) {
    const auto category = diagnostic.category;
    return category == DiagnosticCategory::field ||
           category == DiagnosticCategory::boundary ||
           category == DiagnosticCategory::junction ||
           category == DiagnosticCategory::critical_endpoint ||
           category == DiagnosticCategory::missing_assemblage;
  });
}

Diagnostics field_diagnostics(const Result &result, Engine &engine,
                              const FieldPolygons &geometry) {
  auto messages = geometry.diagnostics;
  for (const auto &line : result.boundaries)
    if (ambiguous_boundary(line) && !lies_on_calculation_frame(line, result))
      messages.push_back({DiagnosticCategory::boundary, line.id,
                          "neighbouring fields could not be distinguished "
                          "within tolerances."});
  for (const auto &node : result.nodes) {
    if (node.kind == "critical_point" && node.incident_lines.size() == 1 &&
        !node.is_verified_critical_endpoint)
      messages.push_back({DiagnosticCategory::critical_endpoint, node.id});
    if (node.kind == "junction" && node.incident_lines.size() == 1)
      messages.push_back({DiagnosticCategory::junction, node.id});
  }
  std::set<std::vector<int>> bordered;
  for (const auto &p : geometry.polygons)
    if (!p.has_open_boundary && !p.outside_model_domain && !p.phases.empty())
      bordered.insert(p.phases);
  // A field sampled only at nodes is a limiting assemblage of a junction
  // cluster, not an areal region.
  const double dp =
      std::max(result.pressure_range[1] - result.pressure_range[0],
               std::numeric_limits<double>::min());
  const double dt =
      std::max(result.temperature_range[1] - result.temperature_range[0],
               std::numeric_limits<double>::min());
  auto only_at_nodes = [&](const Field &field) {
    if (result.section.type != DiagramType::PT || field.sample_indices.empty())
      return false;
    for (int index : field.sample_indices) {
      const auto &sample = result.samples.at(static_cast<std::size_t>(index));
      bool near = false;
      for (const auto &node : result.nodes)
        if (std::abs(sample.pressure - node.pressure) / dp <=
                result.settings.node_tolerance &&
            std::abs(sample.temperature - node.temperature) / dt <=
                result.settings.node_tolerance) {
          near = true;
          break;
        }
      if (!near)
        return false;
    }
    return true;
  };
  auto only_on_latent_edges = [&](const Field &field) {
    if (engine.direct_coordinates() || field.sample_indices.empty())
      return false;
    const auto ranges = result.coordinate_ranges();
    const Eigen::Vector2d range(ranges[0][1] - ranges[0][0],
                                ranges[1][1] - ranges[1][0]);
    for (int index : field.sample_indices) {
      const auto &sample = result.samples.at(static_cast<std::size_t>(index));
      const auto target = diagram_coordinates(sample, result.section.type);
      bool found = false;
      for (const auto &line : result.boundaries) {
        if (engine.canonical_ids(line.assemblage) != field.phases)
          continue;
        for (std::size_t k = 1; k < line.points.size(); ++k) {
          const auto a = diagram_coordinates(line.points[k - 1],
                                             result.section.type),
                     b = diagram_coordinates(line.points[k],
                                             result.section.type);
          const Eigen::Vector2d along = (b - a).cwiseQuotient(range);
          if (along.norm() == 0. ||
              segment_distance(Eigen::Vector2d::Zero(),
                               (a - target).cwiseQuotient(range),
                               (b - target).cwiseQuotient(range)) >
                  result.settings.node_tolerance)
            continue;
          try {
            auto point = boundary_point_at(
                engine, line, sample.phases, target,
                along.normalized().cwiseQuotient(range),
                equilibrium_coordinate_tolerance(result.section.type, range));
            if (point &&
                (Eigen::Vector2d(point->pressure - sample.pressure,
                                 point->temperature - sample.temperature)
                     .cwiseAbs()
                     .array() <= equilibrium_pt_tolerance().array())
                    .all() &&
                same_phase_compositions(point->phases, sample.phases,
                                        result.settings.composition_tolerance,
                                        result.settings.amount_tolerance)) {
              found = true;
              break;
            }
          } catch (const std::exception &) {
          }
        }
        if (found)
          break;
      }
      if (!found)
        return false;
    }
    return true;
  };
  if (result.fields.size() > 1)
    for (const auto &field : result.fields)
      if (!bordered.count(field.phases) && !only_at_nodes(field) &&
          !only_on_latent_edges(field)) {
        if (!engine.free_vectors.empty()) {
          Eigen::MatrixXd span(0, engine.bulk_direction.size());
          for (int id : field.phases) {
            const auto &a = engine.phases
                                .at(static_cast<std::size_t>(
                                    id / engine.settings.max_phase_instances))
                                .a;
            span.conservativeResize(span.rows() + a.rows(), Eigen::NoChange);
            span.bottomRows(a.rows()) = a;
          }
          auto coefficients = span.transpose()
                                  .completeOrthogonalDecomposition()
                                  .solve(engine.bulk_direction)
                                  .eval();
          if ((span.transpose() * coefficients - engine.bulk_direction).norm() >
              1.e-10 * engine.bulk_direction.norm())
            continue;
        }
        messages.push_back({DiagnosticCategory::missing_assemblage, field.id});
      }
  return messages;
}
} // namespace detail

namespace {
// Both public verification operations preserve unrelated diagnostics and use
// the same field-status refresh and accounting after their specialised checks.
template <typename Verify>
Result reverify_result(const Result &previous,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       Verify verify) {
  Result result = previous;
  detail::Engine engine(result.composition_start, candidates, result.settings,
                        result.section);
  engine.pressure_bounds = result.pressure_range;
  engine.temperature_bounds = result.temperature_range;
  detail::erase_field_diagnostics(result);
  const auto graph = verify(result, engine);
  detail::refresh_diagram_diagnostics(result, graph);
  const auto messages = detail::field_diagnostics(result, engine, graph);
  result.diagnostics.insert(result.diagnostics.end(), messages.begin(),
                            messages.end());
  result.equilibrium_solves += engine.equilibrium_solves;
  result.minimization_calls += engine.minimization_calls;
  result.resolved = result.diagnostics.empty();
  detail::audit_thermodynamic_samples(result);
  return result;
}
} // namespace

Result verify_critical_endpoints(
    const Result &previous,
    const std::vector<std::shared_ptr<Material>> &candidates) {
  return reverify_result(
      previous, candidates, [](Result &result, detail::Engine &engine) {
        // Full native snapshots index their nodes contiguously. Cropped
        // networks can retain original IDs and connections to omitted exterior
        // boundaries; their incidence cannot be reconstructed from this partial
        // record alone.
        bool indexed_graph = true;
        for (std::size_t i = 0; i < result.nodes.size(); ++i)
          indexed_graph =
              indexed_graph && result.nodes[i].id == static_cast<int>(i);
        for (const auto &line : result.boundaries)
          for (int id : {line.start_node, line.end_node})
            indexed_graph =
                indexed_graph &&
                (id < 0 || static_cast<std::size_t>(id) < result.nodes.size());
        if (indexed_graph)
          detail::rebuild_boundary_incidence(result);
        detail::verify_critical_endpoints(result, engine);
        return detail::recover_boundary_labels(result);
      });
}

Result verify_field_perimeters(
    const Result &previous,
    const std::vector<std::shared_ptr<Material>> &candidates) {
  return reverify_result(
      previous, candidates, [](Result &result, detail::Engine &engine) {
        detail::verify_field_perimeters(result, engine, true);
        return field_polygons(result);
      });
}

} // namespace burnman::phase_diagrams
