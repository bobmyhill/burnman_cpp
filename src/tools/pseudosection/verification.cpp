// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "critical.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>

namespace burnman::pseudosections {
namespace detail {
namespace {
bool terminal_critical(Engine &engine, const Node &node, const Boundary &line) {
  if (!line.is_solution_replacement || line.side_a.empty() ||
      line.side_a != line.side_b || line.points.size() < 2)
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
  auto zero = std::find_if(states.begin(), states.end(), [&](const auto &p) {
    return p.id == line.zero_phase;
  });
  if (zero == states.end() ||
      node.critical_mode.size() != zero->composition.size() ||
      !node.critical_mode.allFinite() || node.critical_mode.norm() == 0.)
    return false;
  double total = 0.;
  for (const auto &p : states)
    total += p.amount;
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
  if (field_variance(*a, line.assemblage.size()) != 1)
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
      auto first = std::find_if(warm.begin(), warm.end(), [&](const auto &p) {
        return p.id == line.zero_phase;
      });
      auto second = std::find_if(warm.begin(), warm.end(),
                                 [&](const auto &p) { return p.id == other; });
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
        node.kind != "critical_point" || node.gibbs_variance != 1 ||
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
    if (engine.settings.verbose)
      std::cerr << "Critical endpoint " << node.id << ": "
                << (node.is_verified_critical_endpoint
                        ? "verified univariant termination"
                        : "not certified")
                << '\n';
  }
}

std::vector<std::string> field_diagnostics(const Result &result,
                                           Engine &engine) {
  auto geometry = field_polygons(result, 1.e-8, true, false);
  auto messages = geometry.diagnostics;
  for (const auto &node : result.nodes) {
    if (node.kind == "critical_point" && node.incident_lines.size() == 1 &&
        !node.is_verified_critical_endpoint)
      messages.push_back("Unpaired solution critical endpoint at node " +
                         std::to_string(node.id) +
                         "; its adjoining field remains unresolved.");
    if (node.kind == "junction" && node.incident_lines.size() == 1)
      messages.push_back("Unpaired internal junction at node " +
                         std::to_string(node.id) + ".");
  }
  std::set<std::vector<int>> bordered;
  for (const auto &p : geometry.polygons)
    if (!p.has_open_boundary && !p.outside_model_domain && !p.phases.empty())
      bordered.insert(p.phases);
  // A field sampled only at nodes is a limiting assemblage of a junction
  // cluster, not an areal region.
  const double dp = std::max(result.pressure_range[1] - result.pressure_range[0],
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
  if (result.fields.size() > 1)
    for (const auto &field : result.fields)
      if (!bordered.count(field.phases) && !only_at_nodes(field)) {
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
        messages.push_back("No closed region resolved for sampled assemblage " +
                           std::to_string(field.id) +
                           "; increase seed density or inspect failed solves.");
      }
  return messages;
}
} // namespace detail

Result verify_critical_endpoints(
    const Result &previous,
    const std::vector<std::shared_ptr<Material>> &candidates) {
  Result result = previous;
  detail::Engine engine(result.composition_start, candidates, result.settings,
                        result.section);
  engine.pressure_bounds = result.pressure_range;
  engine.temperature_bounds = result.temperature_range;
  // Regenerate the same derived diagnostics used after tracing. Preserve
  // every unrelated message, without matching hard-coded warning prefixes.
  const auto old = detail::field_diagnostics(result, engine);
  result.diagnostics.erase(
      std::remove_if(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const auto &message) {
                       return std::find(old.begin(), old.end(), message) !=
                              old.end();
                     }),
      result.diagnostics.end());
  detail::verify_critical_endpoints(result, engine);
  auto messages = detail::field_diagnostics(result, engine);
  result.diagnostics.insert(result.diagnostics.end(), messages.begin(),
                            messages.end());
  result.equilibrium_solves += engine.equilibrium_solves;
  result.minimization_calls += engine.minimization_calls;
  result.resolved = result.diagnostics.empty();
  return result;
}
} // namespace burnman::pseudosections
