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

#include "burnman/tools/phase_diagram/calculation.hpp"
#include "burnman/tools/phase_diagram/boundary_repair.hpp"
#include "burnman/tools/phase_diagram/consistency.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/tools/phase_diagram/tracing.hpp"
#include "burnman/tools/phase_diagram/verification.hpp"
#include "burnman/utils/index_utils.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace burnman::phase_diagrams::detail {
// Master sequence for new diagrams and saved-diagram refinement.
Result Tracer::calculate(const Result *previous,
                         const std::optional<LineResolution> &resolution) {
  if (previous && previous->resolved && previous->diagnostics.empty()) {
    validate_previous_identity(*previous);
    result = *previous;
    result.settings = engine.settings;
    line_resolution = resolution;
    if (resolution) {
      validate_line_resolution(result, *resolution);
      result.spacing_diagnostics = detail::refine_line_spacing(
          result, engine, *resolution, [&](Boundary &line, std::size_t k) {
            return refine_segment(line, k);
          });
    }
    // Preserve accepted connections, labels, original points, nodes, samples
    // and exclusions; point insertion needs no recovery or recertification.
    result.equilibrium_solves += engine.equilibrium_solves;
    result.minimization_calls += engine.minimization_calls;
    audit_thermodynamic_samples(result);
    return result;
  }
  if (previous) {
    const bool search_junctions = prepare_previous(*previous, resolution);
    recover();
    if (search_junctions)
      search();
  } else {
    seed_boundaries();
    search();
    recover();
  }
  apply_model_domain();
  remove_retraced_spurs();
  recover_failed_seeds();
  resolve_boundaries();
  // Recovery can add accepted endpoints after the first domain pass.
  detail::join_frame_limits(result, engine);
  detail::join_model_limits(result, engine);
  if (detail::remove_metastable_parallel_lines(result, engine))
    index_boundaries();
  collect_boundary_diagnostics();
  verify_field_interiors();
  return certify_result();
}

Tracer::Tracer(const types::FormulaMap &bulk,
               const std::vector<std::shared_ptr<Material>> &phases,
               const std::array<double, 2> &pr, const std::array<double, 2> &tr,
               const Settings &settings, const CompositionSection &section)
    : engine(bulk, phases, settings, section) {
  result.section = engine.section;
  result.composition_start = engine.bulk_start;
  result.pressure_range = pr;
  result.temperature_range = tr;
  engine.pressure_bounds = pr;
  engine.temperature_bounds = tr;
  engine.physical_seed << (pr[0] + pr[1]) * .5, (tr[0] + tr[1]) * .5;
  auto axes = diagram_axes(section.type);
  if (has_composition_axis(section.type)) {
    if (!engine.section.fixed_coordinate)
      throw std::invalid_argument(
          "X diagrams require one fixed thermodynamic coordinate.");
    if (*engine.section.fixed_coordinate == axes[0] ||
        *engine.section.fixed_coordinate == Coordinate::X ||
        !std::isfinite(engine.section.fixed_value))
      throw std::invalid_argument(
          "Fix one thermodynamic coordinate outside the diagram axes.");
    if (section.composition_range[0] < 0. || section.composition_range[1] > 1.)
      throw std::invalid_argument(
          "Composition range must lie between 0 and 1.");
  } else if (section.fixed_coordinate)
    throw std::invalid_argument(
        "Only X diagrams take an additional fixed coordinate.");
  result.section = engine.section;
  const auto ranges = result.coordinate_ranges();
  for (auto r : ranges)
    if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || r[1] <= r[0])
      throw std::invalid_argument(
          "Diagram ranges must be finite and increasing.");
  for (std::size_t k = 0; k < axes.size(); ++k)
    if (axes[k] == Coordinate::V && ranges[k][0] <= 0.)
      throw std::invalid_argument("Volume must be positive.");
  if (engine.section.fixed_coordinate) {
    auto axis = *engine.section.fixed_coordinate;
    double value = engine.section.fixed_value;
    if ((axis == Coordinate::V && value <= 0.) ||
        ((axis == Coordinate::P || axis == Coordinate::T) && value < 0.))
      throw std::invalid_argument(
          "Fixed volume must be positive; P and T must be nonnegative.");
  }
  for (auto r : {pr, tr})
    if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || r[1] < r[0])
      throw std::invalid_argument(
          "Physical seed bounds must be finite and ordered.");
  if (pr[0] < 0 || tr[0] < 0 || (tr[0] == 0 && !settings.active_solution_faces))
    throw std::invalid_argument("Phase diagrams require P>=0 and T>0 (T=0 "
                                "requires active_solution_faces).");
  origin << ranges[0][0], ranges[1][0];
  range << ranges[0][1] - ranges[0][0], ranges[1][1] - ranges[1][0];
  result.settings = engine.settings;
  for (auto &phase : engine.phases)
    for (int j = 0; j < settings.max_phase_instances; ++j)
      result.phase_names.push_back(phase.material->get_name() +
                                   (j ? " #" + std::to_string(j + 1) : ""));
}

void Tracer::seed_boundaries() {
  auto &opts = engine.settings;
  auto seed_count = [&](Coordinate axis) {
    switch (axis) {
    case Coordinate::P:
      return opts.pressure_seeds;
    case Coordinate::T:
      return opts.temperature_seeds;
    case Coordinate::S:
      return opts.entropy_seeds;
    case Coordinate::V:
      return opts.volume_seeds;
    case Coordinate::X:
      return opts.composition_seeds;
    }
    return 7;
  };
  const auto axes = diagram_axes(result.section.type);
  const int first_seeds = seed_count(axes[0]),
            second_seeds = seed_count(axes[1]);
  const auto grid_start = std::chrono::steady_clock::now();
  for (int i = 0; i < first_seeds; ++i)
    for (int j = 0; j < second_seeds; ++j) {
      auto actual = (origin + Eigen::Vector2d(double(i) / (first_seeds - 1),
                                              double(j) / (second_seeds - 1))
                                  .cwiseProduct(range))
                        .eval();
      // Reuse the same second coordinate in the preceding row, including
      // the cold start after each hot end-of-row seed. Use the preceding
      // column when that row's seed was unsuccessful.
      const WorkState *warm = nullptr;
      const auto columns = static_cast<std::size_t>(second_seeds);
      if (i > 0 && grid[grid.size() - columns].state.success)
        warm = &grid[grid.size() - columns];
      else if (j > 0 && grid.back().state.success)
        warm = &grid.back();
      const auto start = std::chrono::steady_clock::now();
      auto state = engine.stable_at(actual, warm);
      add_sample(state.state);
      grid.push_back(std::move(state));
      if (opts.verbose)
        std::cerr << "Seed " << grid.size() << '/' << first_seeds * second_seeds
                  << " " << grid.back().state.message << " ("
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << " s)\n";
    }
  // An excluded cold frame must not leave an entire temperature row without
  // starting equilibria. Seed the valid model edge, retaining the original
  // exclusion and its reason, before bracketing neighbouring assemblages.
  if (result.section.type == DiagramType::PT &&
      !opts.required_thermal_phases.empty())
    for (int i = 0; i < first_seeds; ++i) {
      const auto k = static_cast<std::size_t>(i * second_seeds);
      if (!grid[k].state.outside_model_domain)
        continue;
      for (int j = 1; j < second_seeds; ++j) {
        const auto &warm = grid[k + static_cast<std::size_t>(j)];
        if (!warm.state.success)
          continue;
        const double pressure = warm.state.pressure;
        double low = grid[k].state.temperature, high = warm.state.temperature;
        for (int pass = 0;
             pass < 40 && high - low > equilibrium_pt_tolerance()[1]; ++pass) {
          const double mid = .5 * (low + high);
          if (engine.eos_bulk_feasible(pressure, mid))
            high = mid;
          else
            low = mid;
        }
        auto edge = engine.stable_at({pressure, high}, &warm);
        if (edge.state.success) {
          add_sample(grid[k].state);
          grid[k] = std::move(edge);
          add_sample(grid[k].state, burnman::utils::checked_int(k));
        }
        break;
      }
    }
  if (opts.verbose)
    std::cerr << "Seed grid completed in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - grid_start)
                     .count()
              << " s. Discovering boundaries between neighbouring seeds.\n";
  thermodynamic_diagnostics = recover_thermodynamic_grid(
      engine, grid, second_seeds, [&](std::size_t k) {
        add_sample(grid[k].state, burnman::utils::checked_int(k));
      });
  const auto bracket_start = std::chrono::steady_clock::now();
  for (int i = 0; i < first_seeds; ++i)
    for (int j = 0; j < second_seeds; ++j) {
      int k = i * second_seeds + j;
      if (i + 1 < first_seeds)
        bracket(grid[static_cast<std::size_t>(k)],
                grid[static_cast<std::size_t>(k + second_seeds)]);
      if (j + 1 < second_seeds)
        bracket(grid[static_cast<std::size_t>(k)],
                grid[static_cast<std::size_t>(k + 1)]);
    }
  if (opts.verbose)
    std::cerr << "Boundary bracketing completed in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - bracket_start)
                     .count()
              << " s.\n";
}

void Tracer::validate_previous_identity(const Result &previous) const {
  if (!previous.composition_start.empty() &&
      previous.composition_start != result.composition_start)
    throw std::invalid_argument("Resume with the same composition path start.");
  if (previous.phase_names != result.phase_names)
    throw std::invalid_argument(
        "Resume with the same candidate names and max_phase_instances.");
}

void Tracer::collect_boundary_diagnostics() {
  for (auto &line : result.boundaries) {
    auto [back, forward] = boundary_statuses(line);
    if (unfinished_boundary(back))
      line.start_node = -1;
    if (unfinished_boundary(forward))
      line.end_node = -1;
    if (unfinished_boundary(back) || unfinished_boundary(forward))
      result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                   ": " + line.termination);
  }
  rebuild_boundary_incidence(result);
  for (auto &state : result.samples)
    if (!state.success && !state.outside_model_domain)
      result.diagnostics.push_back("Unresolved seed: " + state.message);
}

void Tracer::verify_field_interiors() {
  if (engine.settings.verbose)
    std::cerr
        << "Checking phase field consistency from saved boundary labels.\n";
  // Seek the interior point with greatest clearance from outer ring and
  // holes; a centroid can lie outside a concave field or inside one of its
  // holes.
  {
    std::vector<Eigen::Vector2d> centres;
    for (const auto &polygon : field_polygons(result).polygons)
      if ((polygon.is_provisional || polygon.phases.empty()) &&
          !polygon.outside_model_domain && !polygon.has_open_boundary &&
          polygon.label_clearance > 0.)
        centres.push_back(polygon.label_position);
    for (const auto &centre : centres) {
      auto state = field_state(centre);
      if (!state.success)
        continue;
      state.is_field_verification = true;
      add_sample(state);
    }
  }
}

Result Tracer::certify_result() {
  // Domain attachment and removal of metastable records can change the
  // final perimeter after tracing's repair pass. Reuse the same local repair
  // before certification; relabelling a subset of its arcs can leave a spur
  // with stale side evidence or an artificial hole between coarse chords.
  repair_field_boundaries(
      result, engine, [&](Boundary &line) { label(line); },
      [&](Boundary &line, std::size_t k) {
        return refine_segment(line, k, true);
      });
  index_boundaries();
  // The final repair can merge nodes, split arcs or remove boundary records
  // after collect_boundary_diagnostics() rebuilt these connections. Certify
  // the retained graph rather than stale tracing history.
  rebuild_boundary_incidence(result);
  // Verify endpoint diagnostics after the final geometry repair.
  detail::verify_critical_endpoints(result, engine);
  const auto graph = detail::recover_boundary_labels(result);
  auto messages = detail::field_diagnostics(result, engine, graph);
  result.diagnostics.insert(result.diagnostics.end(), messages.begin(),
                            messages.end());
  result.diagnostics.insert(result.diagnostics.end(),
                            thermodynamic_diagnostics.begin(),
                            thermodynamic_diagnostics.end());
  if (!queue.empty())
    result.diagnostics.push_back(
        "Boundary search limit reached; increase max_lines.");
  if (result.fields.size() > 1 && result.boundaries.empty())
    result.diagnostics.push_back(
        "Field changes detected but no boundaries converged.");
  result.equilibrium_solves += engine.equilibrium_solves;
  result.minimization_calls += engine.minimization_calls;
  result.resolved = result.diagnostics.empty();
  audit_thermodynamic_samples(result);
  return result;
}
} // namespace burnman::phase_diagrams::detail

namespace burnman::phase_diagrams {
Result phase_diagram(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const std::array<double, 2> &pr,
                     const std::array<double, 2> &tr, const Settings &settings,
                     const CompositionSection &section) {
  return detail::Tracer(bulk, phases, pr, tr, settings, section).calculate();
}
Result
refine_phase_diagram(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const Result &previous, const Settings &settings,
                     const std::optional<LineResolution> &resolution) {
  return detail::Tracer(bulk, phases, previous.pressure_range,
                        previous.temperature_range, settings, previous.section)
      .calculate(&previous, resolution);
}
Result
refine_phase_diagram(const types::FormulaMap &bulk,
                     const std::vector<std::shared_ptr<Material>> &phases,
                     const Result &previous,
                     const std::optional<LineResolution> &resolution) {
  return refine_phase_diagram(bulk, phases, previous, previous.settings,
                              resolution);
}
State stable_equilibrium_at(
    const types::FormulaMap &bulk,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Eigen::Vector2d &coordinates, DiagramType type,
    const Settings &settings, const std::array<double, 2> &pressure_range,
    const std::array<double, 2> &temperature_range) {
  return equilibrium_path(bulk, candidates, {coordinates}, type, settings,
                          pressure_range, temperature_range)
      .states.front();
}

EquilibriumPath
equilibrium_path(const types::FormulaMap &bulk,
                 const std::vector<std::shared_ptr<Material>> &candidates,
                 const std::vector<Eigen::Vector2d> &coordinates,
                 DiagramType type, const Settings &settings,
                 const std::array<double, 2> &pressure_range,
                 const std::array<double, 2> &temperature_range,
                 const std::vector<std::string> &properties, bool relaxed,
                 types::AveragingType averaging_scheme) {
  const auto supported = thermodynamic_property_names();
  for (const auto &name : properties)
    if (std::find(supported.begin(), supported.end(), name) == supported.end())
      throw std::invalid_argument("Unknown scalar thermodynamic property: " +
                                  name);
  const auto axes = diagram_axes(type);
  if (axes[1] == Coordinate::X)
    throw std::invalid_argument(
        "Select two of P, T, S and V for a fixed bulk.");
  if (coordinates.empty())
    throw std::invalid_argument(
        "An equilibrium path needs at least one point.");
  for (const auto &point : coordinates)
    for (int i = 0; i < 2; ++i) {
      const auto axis = axes[static_cast<std::size_t>(i)];
      const double value = point[i];
      if (!std::isfinite(value) ||
          ((axis == Coordinate::P || axis == Coordinate::T) && value < 0.) ||
          (axis == Coordinate::T && value == 0. &&
           !settings.active_solution_faces) ||
          (axis == Coordinate::V && value <= 0.))
        throw std::invalid_argument(
            "Equilibrium coordinates must be finite: P,T >= 0 and V > 0; "
            "T=0 requires active_solution_faces.");
    }
  for (const auto &bounds : {pressure_range, temperature_range})
    if (!std::isfinite(bounds[0]) || !std::isfinite(bounds[1]) ||
        bounds[0] < 0. || bounds[1] < bounds[0])
      throw std::invalid_argument("P/T search ranges must be finite, "
                                  "nonnegative and ordered.");
  CompositionSection section;
  section.type = type;
  detail::Engine engine(bulk, candidates, settings, section);
  engine.pressure_bounds = pressure_range;
  engine.temperature_bounds = temperature_range;
  engine.physical_seed << .5 * (pressure_range[0] + pressure_range[1]),
      .5 * (temperature_range[0] + temperature_range[1]);
  EquilibriumPath result;
  result.states.reserve(coordinates.size());
  for (const auto &name : properties)
    result.values[name] =
        Eigen::VectorXd::Constant(static_cast<Eigen::Index>(coordinates.size()),
                                  std::numeric_limits<double>::quiet_NaN());
  detail::WorkState warm;
  const auto interval = std::max(std::size_t{1}, coordinates.size() / 20);
  for (const auto &point : coordinates) {
    auto next = engine.stable_at(point, warm.state.success ? &warm : nullptr);
    const auto index = static_cast<Eigen::Index>(result.states.size());
    result.states.push_back(next.state);
    if (next.state.success && !next.state.outside_model_domain) {
      next.assemblage->set_averaging_scheme(averaging_scheme);
      for (const auto &name : properties)
        try {
          result.values[name][index] =
              assemblage_properties(*next.assemblage, {name}, relaxed).at(name);
        } catch (const std::exception &e) {
          result.property_diagnostics.push_back(
              "Point " + std::to_string(index) + ": " + name + ": " + e.what());
        }
    }
    if (next.state.success)
      warm = std::move(next);
    if (settings.verbose && (result.states.size() % interval == 0 ||
                             result.states.size() == coordinates.size()))
      std::cout << "Path point " << result.states.size() << "/"
                << coordinates.size() << "; " << engine.equilibrium_solves
                << " equilibrium solves" << std::endl;
  }
  result.equilibrium_solves = engine.equilibrium_solves;
  return result;
}

State stable_equilibrium(
    const types::FormulaMap &bulk,
    const std::vector<std::shared_ptr<Material>> &candidates, double p,
    double t, const Settings &settings) {
  if (!std::isfinite(p) || p < 0 || !std::isfinite(t) || t < 0 ||
      (t == 0 && !settings.active_solution_faces))
    throw std::invalid_argument("Equilibrium requires finite P>=0 and T>0 (T=0 "
                                "requires active_solution_faces).");
  detail::Engine engine(bulk, candidates, settings);
  return engine.stable(p, t).state;
}
} // namespace burnman::phase_diagrams
