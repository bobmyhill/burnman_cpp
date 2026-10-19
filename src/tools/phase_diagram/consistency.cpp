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
double affinity_uncertainty(double amount_a, double affinity_a, double amount_b,
                            double affinity_b) {
  // Candidate affinities are per mole of the competing phase. The chemical
  // potential bound therefore uses the other assemblage's mole amount, e.g.
  // two moles of A + B can replace one mole of AB at the same bulk composition.
  return amount_b * std::max(0., -affinity_a) +
         amount_a * std::max(0., -affinity_b);
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
  return tolerance +
         affinity_uncertainty(total_amount(a.phases), a.minimum_affinity,
                              total_amount(b.phases), b.minimum_affinity);
}
ThermodynamicSample thermodynamic_sample(const State &state,
                                         const CompositionSection &section) {
  ThermodynamicSample sample;
  sample.position = diagram_coordinates(state, section.type);
  sample.energy = thermodynamic_potential(state, section);
  sample.uncertainty = thermodynamic_energy_tolerance(state, State{});
  sample.amount = total_amount(state.phases);
  sample.minimum_affinity = state.minimum_affinity;
  sample.bulk_coordinate = state.composition_coordinate;
  sample.available = state.success && !state.outside_model_domain;
  const auto axes = diagram_axes(section.type);
  for (std::size_t axis = 0; axis < 2; ++axis) {
    sample.gradient[static_cast<Eigen::Index>(axis)] =
        conjugate(state, axes[axis]);
    sample.applicable[axis] =
        natural_diagram(section).has_value() && axes[axis] != Coordinate::X;
    sample.convex[axis] =
        axes[axis] == Coordinate::S || axes[axis] == Coordinate::V;
  }
  return sample;
}
double audit_thermodynamic_pair(const ThermodynamicSample &a,
                                const ThermodynamicSample &b, std::size_t axis,
                                ThermodynamicValidation &coverage) {
  if (!a.available || !b.available || !a.applicable[axis] ||
      !b.applicable[axis] || a.bulk_coordinate != b.bulk_coordinate) {
    ++coverage.skipped_pairs;
    return 0.;
  }
  ++coverage.checked_pairs;
  const auto k = static_cast<Eigen::Index>(axis), other = 1 - k;
  double excess = std::numeric_limits<double>::infinity();
  if (std::isfinite(a.energy) && std::isfinite(b.energy) &&
      a.position.allFinite() && b.position.allFinite() &&
      std::isfinite(a.gradient[k]) && std::isfinite(b.gradient[k]) &&
      std::isfinite(a.amount) && std::isfinite(b.amount) &&
      std::isfinite(a.minimum_affinity) && std::isfinite(b.minimum_affinity) &&
      std::isfinite(a.uncertainty) && std::isfinite(b.uncertainty)) {
    const double delta = b.position[k] - a.position[k],
                 change = b.energy - a.energy;
    const double sign = a.convex[axis] ? 1. : -1.;
    // Allow roundoff in the nominally fixed orthogonal coordinate. Diagram
    // sampling below only groups coordinates that agree to roundoff.
    double tolerance = a.uncertainty + b.uncertainty +
                       affinity_uncertainty(a.amount, a.minimum_affinity,
                                            b.amount, b.minimum_affinity);
    if (a.applicable[static_cast<std::size_t>(other)] &&
        a.position[other] != b.position[other])
      tolerance +=
          std::abs(b.position[other] - a.position[other]) *
          std::max(std::abs(a.gradient[other]), std::abs(b.gradient[other]));
    excess = std::max(
        0.,
        (delta == 0. ? std::abs(change)
                     : std::max({0., sign * (a.gradient[k] * delta - change),
                                 sign * (change - b.gradient[k] * delta)})) -
            tolerance);
  }
  if (excess > 0.)
    ++coverage.violations;
  return excess;
}
double thermodynamic_violation(const State &a, const State &b,
                               const CompositionSection &section,
                               std::size_t axis) {
  ThermodynamicValidation coverage;
  return audit_thermodynamic_pair(thermodynamic_sample(a, section),
                                  thermodynamic_sample(b, section), axis,
                                  coverage);
}

void audit_thermodynamic_samples(Result &result) {
  auto &coverage = result.thermodynamic_validation;
  coverage = {"aligned_samples", 0, 0, 0};
  erase_if(result.diagnostics, [](const auto &message) {
    return message.rfind("Thermodynamic audit: ", 0) == 0 ||
           message.rfind("Thermodynamic inconsistency between seed ", 0) == 0;
  });
  std::vector<ThermodynamicSample> samples;
  for (const auto &state : result.samples)
    samples.push_back(thermodynamic_sample(state, result.section));
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const auto other = static_cast<Eigen::Index>(1 - axis);
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < samples.size(); ++i)
      if (samples[i].position.allFinite())
        order.push_back(i);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
      return samples[a].position[other] < samples[b].position[other];
    });
    for (std::size_t first = 0; first < order.size();) {
      std::size_t last = first + 1;
      const double fixed = samples[order[first]].position[other];
      const double tolerance =
          64. * std::numeric_limits<double>::epsilon() * std::abs(fixed);
      while (last < order.size() &&
             samples[order[last]].position[other] - fixed <= tolerance)
        ++last;
      std::sort(order.begin() + static_cast<std::ptrdiff_t>(first),
                order.begin() + static_cast<std::ptrdiff_t>(last),
                [&](auto a, auto b) {
                  return samples[a].position[static_cast<Eigen::Index>(axis)] <
                         samples[b].position[static_cast<Eigen::Index>(axis)];
                });
      for (auto i = first + 1; i < last; ++i)
        if (audit_thermodynamic_pair(samples[order[i - 1]], samples[order[i]],
                                     axis, coverage) > 0.)
          result.diagnostics.push_back(
              "Thermodynamic audit: inconsistent saved samples " +
              std::to_string(order[i - 1]) + " and " +
              std::to_string(order[i]) + ".");
      first = last;
    }
  }
  if (!natural_diagram(result.section))
    coverage.scope = "not_applicable";
  result.resolved = result.resolved && result.diagnostics.empty();
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
