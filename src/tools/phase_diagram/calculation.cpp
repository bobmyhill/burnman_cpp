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
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/diagram_verification.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/field_verification.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/tools/phase_diagram/tracing.hpp"
#include "burnman/utils/index_utils.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <tuple>

namespace burnman::phase_diagrams::detail {
// Master sequence for new diagrams and saved-diagram refinement.
Result Tracer::calculate(const Result *previous,
                         const std::optional<LineResolution> &resolution) {
  if (previous) {
    validate_previous_identity(*previous);
    result = *previous;
    result.settings = engine.settings;
    if (resolution) {
      validate_line_resolution(result, *resolution);
      result.spacing_diagnostics = detail::refine_line_spacing(
          result, engine, *resolution, [&](Boundary &line, std::size_t k) {
            return refine_segment(line, k);
          });
      record_limiting_curves(result, engine);
    }
    // Preserve the saved graph and evidence, including unresolved problems.
    // Refinement only inserts points; recovery belongs to diagram creation.
    result.equilibrium_solves += engine.equilibrium_solves;
    result.minimization_calls += engine.minimization_calls;
    audit_thermodynamic_samples(result);
    return result;
  }
  seed_boundaries();
  search();
  recover();
  apply_model_domain();
  remove_retraced_spurs();
  recover_failed_seeds();
  resolve_boundaries();
  // Recovery can add accepted endpoints after the first domain pass.
  detail::join_frame_limits(result, engine);
  detail::join_model_limits(result, engine);
  const auto overtraces = detail::reaction_overtraces(result);
  erase_if(result.boundaries,
           [&](const Boundary &line) { return overtraces.count(line.id); });
  const auto metastable =
      detail::remove_metastable_parallel_lines(result, engine);
  if (!overtraces.empty() || metastable)
    index_boundaries();
  detach_unfinished_endpoints();
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

void Tracer::detach_unfinished_endpoints() {
  for (auto &line : result.boundaries) {
    auto [back, forward] = boundary_statuses(line);
    if (unfinished_boundary(back))
      line.start_node = -1;
    if (unfinished_boundary(forward))
      line.end_node = -1;
  }
  rebuild_boundary_incidence(result);
}

void Tracer::sample_unlabelled_fields() {
  if (engine.settings.verbose)
    std::cerr << "Sampling unlabelled phase fields.\n";
  // A point of greatest clearance avoids the concave exterior and holes that
  // can contain a polygon's centroid.
  for (const auto &polygon : field_polygons(result).polygons) {
    if ((!polygon.is_provisional && !polygon.phases.empty()) ||
        polygon.outside_model_domain || polygon.has_open_boundary ||
        !(polygon.label_clearance > 0.))
      continue;
    if (auto state = field_state(polygon.label_position); state.success) {
      state.is_field_verification = true;
      add_sample(state);
    }
  }
}

std::vector<Tracer::FieldCut>
Tracer::correct_field_section(const FieldPolygon &polygon, Eigen::Index axis,
                              double slice) {
  const auto accuracy =
      equilibrium_coordinate_tolerance(result.section.type, range);
  std::vector<FieldCut> cuts;
  for (int id : polygon.boundary_ids) {
    const auto &line = result.boundaries.at(static_cast<std::size_t>(id));
    for (std::size_t k = 1; k < line.points.size(); ++k) {
      const auto left = coordinates(line.points[k - 1]),
                 right = coordinates(line.points[k]);
      if (slice < std::min(left[axis], right[axis]) ||
          slice > std::max(left[axis], right[axis]) ||
          std::abs(right[axis] - left[axis]) <= accuracy[axis])
        continue;
      const double fraction = (slice - left[axis]) / (right[axis] - left[axis]);
      const Eigen::Vector2d target = left + fraction * (right - left);
      Eigen::Vector2d weights = Eigen::Vector2d::Zero(),
                      allowance =
                          range * ((right - left).cwiseQuotient(range).norm() +
                                   engine.settings.node_tolerance);
      weights[axis] = 1. / range[axis];
      allowance[axis] = accuracy[axis];
      for (const auto *warm :
           {&line.points[k - 1].phases, &line.points[k].phases})
        try {
          if (auto point = boundary_point_at(engine, line, *warm, target,
                                             weights, allowance)) {
            cuts.push_back(
                {static_cast<std::size_t>(id), k, std::move(*point)});
            break;
          }
        } catch (const std::exception &) {
        }
    }
  }
  const Eigen::Index other = 1 - axis;
  std::sort(cuts.begin(), cuts.end(), [&](const auto &a, const auto &b) {
    return coordinates(a.point)[other] < coordinates(b.point)[other];
  });
  return cuts;
}

void Tracer::verify_field_interiors() {
  if (result.section.type != DiagramType::PT)
    return;
  if (engine.settings.verbose)
    std::cerr << "Verifying field interiors and corrected sections.\n";
  const auto geometry = field_polygons(result);
  const auto accuracy =
      equilibrium_coordinate_tolerance(result.section.type, range);
  for (const auto &polygon : geometry.polygons) {
    if (polygon.outside_model_domain || polygon.has_open_boundary ||
        polygon.phases.empty() || polygon.is_provisional ||
        polygon.boundary_ids.empty())
      continue;
    struct Start {
      double distance;
      PhaseStates phases;
    };
    std::vector<Start> starts;
    std::map<int, int> counts;
    for (int id : polygon.phases)
      ++counts[id / engine.settings.max_phase_instances];
    auto subset = [&](const PhaseStates &phases) {
      PhaseStates selected;
      for (const auto &[candidate, count] : counts) {
        PhaseStates group;
        for (const auto &p : phases)
          if (p.candidate_index == candidate)
            group.push_back(p);
        if (group.size() < static_cast<std::size_t>(count))
          return PhaseStates{};
        std::stable_sort(
            group.begin(), group.end(),
            [](const auto &a, const auto &b) { return a.amount > b.amount; });
        for (int copy = 0; copy < count; ++copy) {
          auto p = group[static_cast<std::size_t>(copy)];
          p.id = candidate * engine.settings.max_phase_instances + copy;
          selected.push_back(std::move(p));
        }
      }
      return selected;
    };
    auto consider = [&](const auto &point) {
      auto selected = subset(point.phases);
      if (!selected.empty())
        starts.push_back(
            {(normalise(point) - normalise(polygon.label_position)).norm(),
             std::move(selected)});
    };
    for (const auto &state : result.samples)
      if (state.success)
        consider(state);
    for (int id : polygon.boundary_ids)
      for (const auto &point :
           result.boundaries.at(static_cast<std::size_t>(id)).points)
        consider(point);
    std::sort(starts.begin(), starts.end(), [](const auto &a, const auto &b) {
      return a.distance < b.distance;
    });
    auto solve =
        [&](const Eigen::Vector2d &location,
            const std::vector<PhaseStates> &extra) -> std::optional<State> {
      auto attempt = [&](const PhaseStates &selected) -> std::optional<State> {
        if (selected.empty())
          return std::nullopt;
        try {
          auto work = engine.equilibrate_phase_set(polygon.phases, selected,
                                                   location[0], location[1],
                                                   std::nullopt, true);
          if (work.state.success &&
              engine.active_phases(work.state) == polygon.phases &&
              engine.locally_stable(*work.assemblage) &&
              engine.distinct(*work.assemblage, work.ids)) {
            work.state.is_field_verification = true;
            work.state.message =
                "Interior verified against the field's equilibrium assemblage";
            return work.state;
          }
        } catch (const std::exception &) {
        }
        return std::nullopt;
      };
      for (const auto &seed : extra)
        if (auto state = attempt(subset(seed)))
          return state;
      for (std::size_t i = 0; i < std::min<std::size_t>(8, starts.size()); ++i)
        if (auto state = attempt(starts[i].phases))
          return state;
      return std::nullopt;
    };
    if (auto state = solve(polygon.label_position, {})) {
      add_sample(*state);
      continue;
    }
    // A fixed assemblage need not determine every elemental potential. A cold
    // phase search supplies a feasible tangent-plane gauge before treating a
    // failed conditional affinity check as evidence of a drawing error.
    auto independent =
        engine.stable(polygon.label_position[0], polygon.label_position[1]);
    if (independent.state.success &&
        engine.active_phases(independent.state) == polygon.phases &&
        engine.locally_stable(*independent.assemblage)) {
      independent.state.is_field_verification = true;
      add_sample(independent.state);
      continue;
    }
    const Eigen::Vector2d low = polygon.vertices.colwise().minCoeff(),
                          high = polygon.vertices.colwise().maxCoeff();
    Eigen::Index axis;
    ((high - low).cwiseQuotient(range)).maxCoeff(&axis);
    bool below_accuracy = false;
    // A strip can be unresolved on fixed-P cuts yet have a resolvable width
    // in pressure. Try both coordinates before treating its interior as below
    // the solver's accuracy; the dominant drawing axis is only a first choice.
    auto verify_sections = [&](Eigen::Index fixed_axis) {
      const Eigen::Index other = 1 - fixed_axis;
      for (double slice :
           {polygon.label_position[fixed_axis],
            low[fixed_axis] + .25 * (high[fixed_axis] - low[fixed_axis]),
            low[fixed_axis] + .5 * (high[fixed_axis] - low[fixed_axis]),
            low[fixed_axis] + .75 * (high[fixed_axis] - low[fixed_axis])}) {
        auto cuts = correct_field_section(polygon, fixed_axis, slice);
        for (std::size_t k = 1; k < cuts.size(); ++k) {
          auto &a = cuts[k - 1], &b = cuts[k];
          const Eigen::Vector2d left = coordinates(a.point),
                                right = coordinates(b.point);
          if (right[other] - left[other] <= accuracy[other] * 4.) {
            below_accuracy = true;
            continue;
          }
          auto blended = average_phase_states(a.point.phases, b.point.phases);
          auto state = solve(.5 * (left + right),
                             {blended, a.point.phases, b.point.phases});
          if (!state)
            continue;
          // Insert independently solved boundary states at the same section.
          // Their midpoint is now inside the drawing as well as the field.
          std::array<FieldCut *, 2> insertions{&a, &b};
          std::sort(insertions.begin(), insertions.end(),
                    [](const auto *first, const auto *second) {
                      return std::tie(first->line, first->segment) >
                             std::tie(second->line, second->segment);
                    });
          for (auto *cut : insertions) {
            auto &points = result.boundaries[cut->line].points;
            if (std::none_of(
                    points.begin(), points.end(), [&](const auto &saved) {
                      return ((coordinates(saved) - coordinates(cut->point))
                                  .cwiseAbs()
                                  .array() <= accuracy.array())
                          .all();
                    }))
              points.insert(points.begin() +
                                static_cast<std::ptrdiff_t>(cut->segment),
                            cut->point);
          }
          add_sample(*state);
          return true;
        }
      }
      return false;
    };
    if (!verify_sections(axis) && !verify_sections(1 - axis)) {
      State failed;
      failed.pressure = polygon.label_position[0];
      failed.temperature = polygon.label_position[1];
      failed.is_field_verification = true;
      failed.message =
          below_accuracy
              ? "Field width could not be resolved above the equilibrium "
                "coordinate accuracy"
              : "No equilibrium interior could be verified for the drawn field";
      add_sample(failed);
    }
  }
}

/// Run one local perimeter repair with this tracer's labelling and corrector
/// callbacks. The repair can join junctions, split arcs, recover side labels and
/// refine suspect segments. Both tracing recovery and final verification use
/// this pass; their callers decide whether to repeat it.
void Tracer::repair_field_perimeters() {
  repair_field_boundaries(
      result, engine, [&](Boundary &line) { label(line); },
      [&](Boundary &line, std::size_t k) { return refine_segment(line, k, true); });
}

/// Final graph repair after domain attachment and metastable-trace removal.
/// Repeat the local pass at most four times, stopping when no eligible field
/// remains provisional, the point count stops changing, or growth reaches 4096.
/// Non-PT sections receive one pass. Then remove overtraces, reindex boundaries
/// and rebuild incidence before certifying the retained graph.
void Tracer::repair_field_graph() {
  const auto point_count = [&]() {
    std::size_t count = 0;
    for (const auto &line : result.boundaries)
      count += line.points.size();
    return count;
  };
  const auto initial_points = point_count();
  for (int pass = 0; pass < 4; ++pass) {
    const auto before = point_count();
    repair_field_perimeters();
    const auto after = point_count();
    if (engine.settings.verbose)
      std::cerr << "Field repair pass " << pass + 1 << "/4: " << before << " -> "
                << after << " boundary points; " << engine.equilibrium_solves
                << " equilibrium solves.\n";
    const auto geometry = field_polygons(result);
    if (result.section.type != DiagramType::PT ||
        std::none_of(geometry.polygons.begin(), geometry.polygons.end(),
                     [](const auto &p) {
                       return p.is_provisional && !p.has_open_boundary &&
                              !p.outside_model_domain && !p.phases.empty();
                     }) ||
        after == before || after >= initial_points + 4096)
      break;
  }
  const auto overtraces = detail::reaction_overtraces(result);
  erase_if(result.boundaries,
           [&](const Boundary &line) { return overtraces.count(line.id); });
  index_boundaries();
  // Repair can merge nodes, split arcs or remove traces. Certify their current
  // incidence rather than the earlier tracing history.
  rebuild_boundary_incidence(result);
}

void Tracer::refine_incompatible_perimeters() {
  // Refine around incompatible perimeter arcs, including their neighbours'
  // solved extent: coarse chords can put unrelated reactions on a perimeter.
  if (result.section.type == DiagramType::PT) {
    const auto geometry = field_polygons(result);
    std::map<int, std::set<int>> neighbours;
    for (const auto &polygon : geometry.polygons)
      if (polygon.is_provisional && !polygon.has_open_boundary &&
          !polygon.outside_model_domain && !polygon.phases.empty())
        for (int id : polygon.boundary_ids) {
          const auto &line = result.boundaries.at(static_cast<std::size_t>(id));
          const auto choices = boundary_assemblages(line, result);
          if (!choices.empty() && !choices.count(polygon.phases))
            neighbours[id].insert(polygon.boundary_ids.begin(),
                                  polygon.boundary_ids.end());
        }
    for (const auto &[id, adjacent_lines] : neighbours) {
      const auto &source = result.boundaries.at(static_cast<std::size_t>(id));
      if (source.points.empty())
        continue;
      Eigen::Vector2d low = coordinates(source.points.front()), high = low;
      auto include = [&](const Boundary &line) {
        for (const auto &point : line.points) {
          const auto q = coordinates(point);
          low = low.cwiseMin(q);
          high = high.cwiseMax(q);
        }
      };
      include(source);
      // Neighbouring perimeter arcs can curve outside the incompatible arc's
      // own bounding box. Include their solved extent at this junction.
      for (int adjacent : adjacent_lines) {
        const auto &line =
            result.boundaries.at(static_cast<std::size_t>(adjacent));
        if (line.start_node != source.start_node &&
            line.start_node != source.end_node &&
            line.end_node != source.start_node &&
            line.end_node != source.end_node)
          continue;
        include(line);
      }
      const Eigen::Vector2d padding =
          ((high - low) * .05).cwiseMax(range * 1.e-7);
      low = (low - padding).cwiseMax(origin);
      high = (high + padding).cwiseMin(origin + range);
      LineResolution local;
      local.axis_points = {65, 65};
      local.region = std::array<std::array<double, 2>, 2>{
          {{low[0], high[0]}, {low[1], high[1]}}};
      if (engine.settings.verbose)
        std::cerr << "Refining incompatible perimeter near boundary " << id
                  << ".\n";
      detail::refine_line_spacing(result, engine, local,
                                  [&](Boundary &line, std::size_t k) {
                                    return refine_segment(line, k);
                                  });
    }
  }
}

Result Tracer::certify_result() {
  sample_unlabelled_fields();
  repair_field_graph();
  // Verify endpoint diagnostics after the final geometry repair. Gap recovery
  // precedes continuation to the frame; both can change the retained graph.
  detail::verify_critical_endpoints(result, engine);
  const auto gaps = detail::recover_solvus_branch_gaps(result, engine);
  const auto extensions =
      detail::continue_boundary_past_coalescence(result, engine);
  if (gaps + extensions) {
    detail::verify_critical_endpoints(result, engine);
    sample_unlabelled_fields();
  }
  refine_incompatible_perimeters();
  verify_field_interiors();
  detail::verify_critical_endpoints(result, engine);
  detail::recover_boundary_labels(result);
  // A sub-resolution drawing does not invalidate a label established by
  // stable reactions meeting at common vertices without intervening curves.
  detail::verify_field_perimeters(result, engine);
  const auto graph = field_polygons(result);
  detail::refresh_diagram_diagnostics(result, graph);
  auto messages = detail::field_diagnostics(result, engine, graph);
  result.diagnostics.insert(result.diagnostics.end(), messages.begin(),
                            messages.end());
  result.diagnostics.insert(result.diagnostics.end(),
                            thermodynamic_diagnostics.begin(),
                            thermodynamic_diagnostics.end());
  if (!queue.empty())
    result.diagnostics.push_back(
        {DiagnosticCategory::search, -1,
         "Boundary search limit reached; increase max_lines."});
  if (result.fields.size() > 1 && result.boundaries.empty())
    result.diagnostics.push_back(
        {DiagnosticCategory::search, -1,
         "Field changes detected but no boundaries converged."});
  record_limiting_curves(result, engine);
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
