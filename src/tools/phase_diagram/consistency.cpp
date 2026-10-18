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

#include "burnman/tools/phase_diagram/consistency.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>

namespace burnman::phase_diagrams::detail {
namespace {
std::optional<DiagramType> natural_diagram(const CompositionSection &section) {
  auto axes = diagram_axes(section.type);
  if (axes[1] == Coordinate::X) {
    if (!section.fixed_coordinate)
      return std::nullopt;
    axes[1] = *section.fixed_coordinate;
  }
  auto has = [&](Coordinate c) { return axes[0] == c || axes[1] == c; };
  if (has(Coordinate::P) && has(Coordinate::T))
    return DiagramType::PT;
  if (has(Coordinate::P) && has(Coordinate::S))
    return DiagramType::PS;
  if (has(Coordinate::T) && has(Coordinate::V))
    return DiagramType::TV;
  if (has(Coordinate::S) && has(Coordinate::V))
    return DiagramType::SV;
  return std::nullopt;
}
double conjugate(const State &state, Coordinate coordinate) {
  switch (coordinate) {
  case Coordinate::P:
    return state.volume;
  case Coordinate::T:
    return -state.entropy;
  case Coordinate::S:
    return state.temperature;
  case Coordinate::V:
    return -state.pressure;
  case Coordinate::X:
    break;
  }
  return std::numeric_limits<double>::quiet_NaN();
}
} // namespace

double thermodynamic_potential(const State &state,
                               const CompositionSection &section) {
  const auto type = natural_diagram(section);
  if (!type)
    return std::numeric_limits<double>::quiet_NaN();
  double energy = state.gibbs;
  if (*type == DiagramType::PS || *type == DiagramType::SV)
    energy += state.temperature * state.entropy;
  if (*type == DiagramType::TV || *type == DiagramType::SV)
    energy -= state.pressure * state.volume;
  return energy;
}
double thermodynamic_energy_tolerance(const State &a, const State &b) {
  double tolerance = 0.;
  for (const auto *s : {&a, &b}) {
    const double moles = total_amount(s->phases);
    const double scale =
        std::max({std::abs(s->gibbs), std::abs(s->temperature * s->entropy),
                  std::abs(s->pressure * s->volume), 1.e5 * moles});
    tolerance += std::abs(s->equilibrium_error) * moles +
                 (std::abs(s->mass_balance_error) +
                  64. * std::numeric_limits<double>::epsilon()) *
                     scale;
  }
  return tolerance;
}
double thermodynamic_violation(const State &a, const State &b,
                               const CompositionSection &section,
                               std::size_t axis) {
  const auto coordinate = diagram_axes(section.type)[axis];
  if (!a.success || !b.success || !natural_diagram(section) ||
      coordinate == Coordinate::X ||
      a.composition_coordinate != b.composition_coordinate)
    return 0.;
  const double delta =
      coordinate_value(b, coordinate) - coordinate_value(a, coordinate);
  if (delta == 0.)
    return 0.;
  const double change =
      thermodynamic_potential(b, section) - thermodynamic_potential(a, section);
  const double sign =
      coordinate == Coordinate::S || coordinate == Coordinate::V ? 1. : -1.;
  const double excess =
      std::max({0., sign * (conjugate(a, coordinate) * delta - change),
                sign * (change - conjugate(b, coordinate) * delta)});
  return std::max(0., excess - thermodynamic_energy_tolerance(a, b));
}

State equilibrium_state(const BoundaryPoint &point, double gibbs) {
  State state;
  state.success = true;
  state.pressure = point.pressure;
  state.temperature = point.temperature;
  state.entropy = point.entropy;
  state.volume = point.volume;
  state.composition_coordinate = point.composition_coordinate;
  state.phases = point.phases;
  state.gibbs = gibbs;
  state.mass_balance_error = point.mass_balance_error;
  state.minimum_affinity = point.minimum_affinity;
  state.equilibrium_error = point.residual;
  return state;
}
State equilibrium_state(Engine &engine, const Assemblage &a,
                        const std::vector<int> &ids) {
  BoundaryPoint point;
  engine.record_coordinates(point, a);
  point.phases = engine.snapshot(a, ids);
  point.mass_balance_error = engine.mass_error(a);
  point.residual = reaction_residual(a);
  return equilibrium_state(point, a.get_n_moles() * a.get_molar_gibbs());
}

WorkState recover_thermodynamic_state(Engine &engine, WorkState best,
                                      const std::vector<State> &neighbours) {
  const auto q = diagram_coordinates(best.state, engine.section.type);
  if (!std::isfinite(thermodynamic_potential(best.state, engine.section)))
    return best;
  best.state.minimum_affinity = engine.stability(*best.assemblage);
  best.state.success =
      best.state.minimum_affinity >= -engine.settings.affinity_tolerance;
  auto consider = [&](WorkState trial) {
    if (trial.state.success && engine.locally_stable(*trial.assemblage) &&
        engine.distinct(*trial.assemblage, trial.ids) &&
        (!best.state.success ||
         thermodynamic_potential(trial.state, engine.section) <
             thermodynamic_potential(best.state, engine.section) -
                 thermodynamic_energy_tolerance(best.state, trial.state)))
      best = std::move(trial);
  };
  for (const auto &neighbour : neighbours)
    if (neighbour.success)
      consider(engine.fixed_at(
          phase_ids(neighbour.phases), neighbour.phases, q,
          Eigen::Vector2d(neighbour.pressure, neighbour.temperature)));
  // A neighbour can occupy the same metastable basin. Independent discovery
  // must not simply inherit its Newton start or its physical temperature.
  const auto physical_seed = engine.physical_seed;
  engine.physical_seed << .5 * (engine.pressure_bounds[0] +
                                engine.pressure_bounds[1]),
      .5 * (engine.temperature_bounds[0] + engine.temperature_bounds[1]);
  consider(engine.stable_at(q));
  engine.physical_seed = physical_seed;
  return best;
}
std::vector<std::string>
recover_thermodynamic_grid(Engine &engine, std::vector<WorkState> &grid,
                           int columns,
                           const std::function<void(std::size_t)> &updated) {
  std::vector<std::string> diagnostics;
  const auto start = std::chrono::steady_clock::now();
  const int solves = engine.equilibrium_solves;
  auto pairs = [&](auto visit) {
    for (std::size_t k = 0; k < grid.size(); ++k) {
      if (k + static_cast<std::size_t>(columns) < grid.size())
        visit(k, k + static_cast<std::size_t>(columns), std::size_t{0});
      if ((k + 1) % static_cast<std::size_t>(columns))
        visit(k, k + 1, std::size_t{1});
    }
  };
  for (int pass = 0; pass < engine.settings.max_recovery_passes; ++pass) {
    std::set<std::size_t> flagged;
    pairs([&](auto a, auto b, auto axis) {
      if (thermodynamic_violation(grid[a].state, grid[b].state, engine.section,
                                  axis) > 0.) {
        flagged.insert(a);
        flagged.insert(b);
      }
    });
    if (flagged.empty())
      break;
    for (auto k : flagged) {
      std::vector<State> neighbours;
      for (int offset : {-columns, -1, 1, columns}) {
        const auto n = static_cast<std::ptrdiff_t>(k) + offset;
        if (n >= 0 && static_cast<std::size_t>(n) < grid.size() &&
            (std::abs(offset) != 1 ||
             n / columns == static_cast<std::ptrdiff_t>(k) / columns))
          neighbours.push_back(grid[static_cast<std::size_t>(n)].state);
      }
      grid[k] = recover_thermodynamic_state(engine, grid[k], neighbours);
      updated(k);
    }
  }
  pairs([&](auto a, auto b, auto axis) {
    if (thermodynamic_violation(grid[a].state, grid[b].state, engine.section,
                                axis) > 0.)
      diagnostics.push_back("Thermodynamic inconsistency between seed " +
                            std::to_string(a) + " and " + std::to_string(b) +
                            ".");
  });
  if (engine.settings.verbose)
    std::cerr << "Thermodynamic seed checks: "
              << engine.equilibrium_solves - solves << " extra solves; "
              << diagnostics.size() << " unresolved pairs ("
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " s).\n";
  return diagnostics;
}
} // namespace burnman::phase_diagrams::detail
