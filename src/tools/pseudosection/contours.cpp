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

#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace burnman::pseudosections {
namespace {
using namespace detail;
using V2 = Eigen::Vector2d;
struct TraceState {
  std::shared_ptr<Assemblage> a;
  optim::roots::DampedNewtonResult solve;
};

class Contours {
  const Result &source;
  Engine engine;
  const ContourConstraintFactory &factory;
  ContourSettings settings;
  ContourResult result;
  V2 origin, range;
  std::array<Coordinate, 2> axes;
  FieldPolygon field;
  // Closed rings after rescaling each diagram axis to 0..1, exterior first.
  std::vector<Eigen::MatrixXd> rings;
  std::vector<int> ids;
  std::unique_ptr<EqualityConstraint> constraint;
  double reference_x = 0., constraint_scale = 1.;
  std::shared_ptr<Assemblage> warm;
  std::size_t first_line = 0;

  V2 normalized(const Assemblage &a) const {
    return (engine.coordinates(a) - origin).cwiseQuotient(range);
  }
  bool inside(const V2 &p) const {
    auto location = [&](const Eigen::MatrixXd &ring) {
      return ring_location(
          static_cast<std::size_t>(ring.rows()),
          [&](std::size_t i) -> V2 {
            return ring.row(static_cast<Eigen::Index>(i));
          },
          p, 1.e-9);
    };
    return location(rings[0]) >= 0 &&
           std::none_of(rings.begin() + 1, rings.end(),
                        [&](const auto &ring) { return location(ring) > 0; });
  }
  double edge_distance(const V2 &p) const {
    double distance = std::numeric_limits<double>::infinity();
    for (const auto &ring : rings)
      for (Eigen::Index i = 1; i < ring.rows(); ++i)
        distance = std::min(distance,
                            segment_distance(p, ring.row(i - 1), ring.row(i)));
    return distance;
  }
  std::unique_ptr<EqualityConstraint> target(const Assemblage &a,
                                             bool scaled = true) const {
    auto c = constraint->clone();
    // Implemented constraints involving the free X parameter are affine.
    // Rebase their intercept, keeping the factory's original reference.
    if (!engine.free_vectors.empty() &&
        dynamic_cast<LinearXConstraint *>(c.get())) {
      auto x = get_parameter_vector(a, 1);
      auto row = c->derivative(x, a, x.size());
      double b =
          row.dot(x) - c->evaluate(x, a) -
          row[x.size() - 1] * (engine.composition_coordinate(a) - reference_x);
      c = std::make_unique<LinearXConstraint>(row, b);
    }
    return scaled ? std::make_unique<SectionConstraint>(
                        std::move(c), std::make_unique<PressureConstraint>(0.),
                        V2(1. / constraint_scale, 0.))
                  : std::move(c);
  }
  std::unique_ptr<EqualityConstraint> plane(const Assemblage &a, const V2 &u,
                                            const V2 &normal) const {
    auto q = (origin + u.cwiseProduct(range)).eval();
    auto n = engine.parameters(a).n_parameters;
    return continuation_plane(engine, a, n, q, normal.cwiseQuotient(range));
  }
  TraceState solve(const Assemblage &initial, ConstraintList c) {
    // Keep the accepted warm state intact when a trial corrector fails.
    TraceState state{engine.copy_assemblage(initial), {}};
    try {
      state.solve = engine.solve(*state.a, c);
    } catch (const std::exception &) {
      state.solve.success = false;
    }
    return state;
  }
  TraceState solve_at(const Assemblage &initial, const V2 &u) {
    auto q = (origin + u.cwiseProduct(range)).eval();
    auto n = engine.parameters(initial).n_parameters;
    double x = engine.composition_coordinate(initial);
    return solve(
        initial,
        constraints(engine.coordinate_constraint(axes[0], q[0], n, x, true),
                    engine.coordinate_constraint(axes[1], q[1], n, x, true)));
  }
  TraceState correct(const Assemblage &initial, const V2 &u, const V2 &normal) {
    return solve(initial,
                 constraints(target(initial), plane(initial, u, normal)));
  }
  double residual(const Assemblage &a) const {
    auto x = get_parameter_vector(a, engine.free_vectors.empty() ? 0 : 1);
    return target(a)->evaluate(x, a);
  }
  BoundaryPoint point(const Assemblage &a) const {
    BoundaryPoint p;
    engine.record_coordinates(p, a);
    p.phases = engine.snapshot(a, ids);
    p.mass_balance_error = engine.mass_error(a);
    auto x = get_parameter_vector(a, engine.free_vectors.empty() ? 0 : 1);
    p.residual = std::abs(target(a, false)->evaluate(x, a));
    return p;
  }
  void issue(const std::string &message) {
    result.resolved = false;
    auto diagnostic =
        "Field " + std::to_string(field.field_id) + ": " + message;
    if (std::find(result.diagnostics.begin(), result.diagnostics.end(),
                  diagnostic) == result.diagnostics.end())
      result.diagnostics.push_back(std::move(diagnostic));
  }
  bool covered(const V2 &p) const {
    for (std::size_t k = first_line; k < result.lines.size(); ++k) {
      const auto &points = result.lines[k].points;
      for (std::size_t i = 1; i < points.size(); ++i) {
        auto a =
            (diagram_coordinates(points[i - 1], source.section.type) - origin)
                .cwiseQuotient(range)
                .eval();
        auto b = (diagram_coordinates(points[i], source.section.type) - origin)
                     .cwiseQuotient(range)
                     .eval();
        if (segment_distance(p, a, b) < settings.step * .05)
          return true;
      }
    }
    return false;
  }
  // First intersection with the saved field exterior or a hole.
  bool clip(const V2 &u, const V2 &delta, double &fraction, V2 &normal) const {
    bool hit = false;
    fraction = 1.;
    for (const auto &ring : rings)
      for (Eigen::Index i = 1; i < ring.rows(); ++i) {
        V2 a = ring.row(i - 1), b = ring.row(i), edge = b - a;
        double det = cross(delta, edge);
        if (std::abs(det) < 1.e-20)
          continue;
        double t = cross(a - u, edge) / det, v = cross(a - u, delta) / det;
        if (t > 1.e-9 && t <= fraction && v >= -1.e-9 && v <= 1. + 1.e-9) {
          hit = true;
          fraction = t;
          normal = V2(-edge[1], edge[0]).normalized();
        }
      }
    return hit;
  }
  std::vector<BoundaryPoint> follow(const TraceState &seed, Eigen::VectorXd d,
                                    bool &closed, std::string &termination) {
    auto a = engine.copy_assemblage(*seed.a);
    std::vector<BoundaryPoint> points{point(*a)};
    V2 start = normalized(*a);
    double step = settings.step, travelled = 0.;
    for (int iteration = 0; iteration < settings.max_trace_steps; ++iteration) {
      V2 u = normalized(*a);
      V2 direction = engine.project_direction(d, *a).cwiseQuotient(range);
      if (edge_distance(u) < 1.e-9 &&
          !inside(u + settings.min_step * direction)) {
        termination = "field boundary";
        return points;
      }
      V2 normal = direction;
      double fraction;
      bool hit = clip(u, step * direction, fraction, normal);
      double h = step * fraction;
      auto inequalities =
          calculate_constraints(*a, engine.free_vectors.empty() ? 0 : 1);
      auto base = get_parameter_vector(*a, engine.free_vectors.empty() ? 0 : 1);
      Eigen::VectorXd slack =
          -(inequalities.first * base + inequalities.second);
      Eigen::VectorXd rate = inequalities.first * d;
      for (Eigen::Index i = 0; i < slack.size(); ++i)
        if (rate[i] > 1.e-14 && slack[i] / rate[i] < h) {
          h = std::max(0., .99 * slack[i] / rate[i]);
          hit = false;
          normal = direction;
        }
      if (h < settings.min_step && !hit) {
        termination = edge_distance(u) < settings.step * .1 ? "field boundary"
                                                            : "physical limit";
        if (termination == "physical limit")
          issue("contour reached a physical limit inside its saved field");
        return points;
      }
      V2 predicted = u + h * direction;
      auto guess = engine.copy_assemblage(*a);
      try {
        set_composition_and_state_from_parameters(*guess, base + h * d);
        if (!engine.free_vectors.empty())
          engine.set_coordinate(*guess, engine.composition_coordinate(*a) +
                                            h * d[d.size() - 1]);
      } catch (const std::exception &) {
        step *= .5;
        continue;
      }
      auto next = correct(*guess, predicted, normal);
      if (!next.solve.success || !inside(normalized(*next.a)) ||
          (normalized(*next.a) - predicted).norm() > h) {
        step *= .5;
        if (step < settings.min_step) {
          termination = "corrector failed";
          issue("contour corrector failed before reaching the field boundary");
          return points;
        }
        continue;
      }
      auto next_d = continuation_tangent(engine, next.solve, *next.a, range);
      if (!next_d.size()) {
        termination = "singular constraint";
        issue("constraint no longer defines a single contour direction");
        return points;
      }
      V2 next_direction =
          engine.project_direction(next_d, *next.a).cwiseQuotient(range);
      if (next_direction.dot(direction) < 0.)
        next_d = -next_d;
      travelled += (normalized(*next.a) - u).norm();
      points.push_back(point(*next.a));
      if (points.size() > 6 && travelled > settings.step * 4. &&
          (normalized(*next.a) - start).norm() < h * .75) {
        points.push_back(points.front());
        closed = true;
        termination = "closed loop";
        return points;
      }
      a = std::move(next.a);
      d = std::move(next_d);
      if (hit) {
        termination = "field boundary";
        return points;
      }
      step = std::min(settings.step, step * 1.3);
    }
    termination = "trace step limit";
    issue("contour reached max_trace_steps");
    return points;
  }
  void trace(const TraceState &seed) {
    if (!seed.solve.success || !inside(normalized(*seed.a)) ||
        covered(normalized(*seed.a)))
      return;
    auto d = continuation_tangent(engine, seed.solve, *seed.a, range);
    if (!d.size()) {
      issue("constraint does not define a one-dimensional contour");
      return;
    }
    ContourLine line;
    line.field_id = field.field_id;
    line.phases = ids;
    std::string forward, backward;
    line.points = follow(seed, d, line.closed, forward);
    if (!line.closed) {
      bool closed = false;
      auto back = follow(seed, -d, closed, backward);
      std::reverse(back.begin(), back.end());
      back.pop_back();
      back.insert(back.end(), line.points.begin(), line.points.end());
      line.points = std::move(back);
    }
    line.termination = line.closed ? forward : backward + "; " + forward;
    if (line.points.size() > 1)
      result.lines.push_back(std::move(line));
  }
  // Sweeps in both directions find boundary crossings and interior loops.
  // Intersections with holes split sweeps; no segment spans different fields.
  void seeds() {
    int successful_seeds = 0;
    for (int axis = 0; axis < 2; ++axis) {
      double low = rings[0].col(axis).minCoeff(),
             high = rings[0].col(axis).maxCoeff();
      std::vector<double> planes;
      for (int i = 0; i < settings.seed_grid; ++i)
        planes.push_back(low +
                         (high - low) * (double(i) + .5) / settings.seed_grid);
      planes.push_back(
          ((field.label_position - origin).cwiseQuotient(range))[axis]);
      for (double value : planes) {
        std::vector<double> cuts;
        for (const auto &ring : rings)
          for (Eigen::Index i = 1; i < ring.rows(); ++i) {
            V2 a = ring.row(i - 1), b = ring.row(i);
            if ((a[axis] <= value && b[axis] > value) ||
                (b[axis] <= value && a[axis] > value))
              cuts.push_back(a[1 - axis] + (value - a[axis]) *
                                               (b[1 - axis] - a[1 - axis]) /
                                               (b[axis] - a[axis]));
          }
        std::sort(cuts.begin(), cuts.end());
        for (std::size_t k = 1; k < cuts.size(); ++k) {
          V2 u;
          u[axis] = value;
          u[1 - axis] = .5 * (cuts[k - 1] + cuts[k]);
          if (!inside(u) || cuts[k] - cuts[k - 1] < 1.e-10)
            continue;
          int count =
              std::max(2, static_cast<int>(std::ceil((cuts[k] - cuts[k - 1]) *
                                                     settings.seed_grid)));
          TraceState previous;
          double old_f = 0.;
          for (int j = 0; j <= count; ++j) {
            double t = std::clamp(double(j) / count, 1.e-7, 1. - 1.e-7);
            u[1 - axis] = cuts[k - 1] + t * (cuts[k] - cuts[k - 1]);
            auto state = solve_at(*warm, u);
            if (!state.solve.success) {
              previous = {};
              continue;
            }
            warm = state.a;
            ++successful_seeds;
            double f = residual(*state.a);
            if (!std::isfinite(f))
              throw std::runtime_error(
                  "Contour constraint returned a nonfinite residual.");
            if (std::abs(f) < 1.e-9 || (previous.a && old_f * f < 0.)) {
              if (covered(normalized(*state.a)) && std::abs(f) < 1.e-9) {
                previous = std::move(state);
                old_f = f;
                continue;
              }
              auto &initial = previous.a && std::abs(old_f) < std::abs(f)
                                  ? previous.a
                                  : state.a;
              V2 normal = V2::Zero();
              normal[axis] = 1.;
              auto root = correct(*initial, u, normal);
              if (!root.solve.success && std::abs(f) < 1.e-9) {
                normal = V2::Zero();
                normal[1 - axis] = 1.;
                root = correct(*state.a, u, normal);
              }
              if (root.solve.success)
                trace(root);
              else
                issue("a bracketed contour seed could not be equilibrated");
            }
            previous = std::move(state);
            old_f = f;
          }
        }
      }
    }
    if (!successful_seeds)
      issue("no seed could be equilibrated in the saved field");
  }
  bool prepare() {
    ids = field.phases;
    double best = std::numeric_limits<double>::infinity(), saved_x = 0.;
    const std::vector<PhaseState> *states = nullptr;
    V2 physical;
    auto consider = [&](const auto &saved) {
      if (!std::all_of(ids.begin(), ids.end(), [&](int id) {
            return std::any_of(saved.phases.begin(), saved.phases.end(),
                               [&](const PhaseState &p) { return p.id == id; });
          }))
        return;
      double distance = (diagram_coordinates(saved, source.section.type) -
                         field.label_position)
                            .cwiseQuotient(range)
                            .norm();
      if (distance < best) {
        best = distance;
        states = &saved.phases;
        physical << saved.pressure, saved.temperature;
        saved_x = saved.composition_coordinate;
      }
    };
    // Polygon construction already selects a verified interior warm state.
    if (field.sample_index >= 0)
      consider(source.samples.at(static_cast<std::size_t>(field.sample_index)));
    if (!states)
      for (const auto &line : source.boundaries)
        for (const auto &saved : line.points)
          consider(saved);
    if (!states) {
      issue("no saved equilibrium state contains the field assemblage");
      return false;
    }
    for (const auto &state : *states)
      if (std::find(ids.begin(), ids.end(), state.id) != ids.end() &&
          (state.candidate_index !=
               state.id / source.settings.max_phase_instances ||
           state.candidate_index < 0 ||
           static_cast<std::size_t>(state.candidate_index) >=
               engine.phases.size() ||
           state.composition.size() !=
               engine.phases[static_cast<std::size_t>(state.candidate_index)]
                   .vertices.cols() ||
           !state.composition.allFinite() || !std::isfinite(state.amount)))
        throw std::invalid_argument("Contours require valid saved phase "
                                    "compositions and candidate_index.");
    if (!engine.free_vectors.empty())
      engine.set_bulk(saved_x);
    warm = engine.make_assemblage(ids, *states, physical[0], physical[1]);
    engine.set_coordinate(*warm, saved_x);
    reference_x = engine.composition_coordinate(*warm);
    auto context = engine.copy_assemblage(*warm);
    constraint = factory(*context, engine.parameters(*context), ids);
    if (!constraint)
      return false;
    if (field.has_open_boundary) {
      issue("unfinished field boundary; contours were skipped");
      return false;
    }
    auto x = get_parameter_vector(*warm, engine.free_vectors.empty() ? 0 : 1);
    auto derivative = constraint->derivative(x, *warm, x.size());
    if (derivative.size() != x.size())
      throw std::invalid_argument("Invalid contour constraint derivative.");
    double scale =
        derivative.cwiseProduct(engine.parameter_scales(*warm, x.size(), range))
            .norm();
    // A nonlinear constraint may have a singular derivative at the warm
    // state (e.g. an ellipse centre); its contour seeds need not be singular.
    constraint_scale = std::isfinite(scale) && scale > 0. ? scale : 1.;
    return true;
  }

public:
  Contours(const Result &previous,
           const std::vector<std::shared_ptr<Material>> &candidates,
           const ContourConstraintFactory &make_constraint,
           const ContourSettings &options)
      : source(previous), engine(previous.composition_start, candidates,
                                 previous.settings, previous.section),
        factory(make_constraint), settings(options),
        axes(diagram_axes(previous.section.type)) {
    result.diagram_type = previous.section.type;
    auto bounds = previous.coordinate_ranges();
    origin << bounds[0][0], bounds[1][0];
    range << bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0];
    engine.pressure_bounds = previous.pressure_range;
    engine.temperature_bounds = previous.temperature_range;
  }
  ContourResult run() {
    auto geometry = field_polygons(source);
    for (const auto &polygon : geometry.polygons) {
      field = polygon;
      if (field.outside_model_domain)
        continue;
      if (field.phases.empty()) {
        issue("unidentified assemblage; contours were skipped");
        continue;
      }
      rings.clear();
      auto add = [&](const Eigen::MatrixXd &ring) {
        rings.push_back((ring.rowwise() - origin.transpose()) *
                        range.cwiseInverse().asDiagonal());
      };
      add(field.vertices);
      for (const auto &hole : field.holes)
        add(hole);
      first_line = result.lines.size();
      if (prepare())
        seeds();
      if (source.settings.verbose)
        std::cerr << "Contour field " << field.field_id << ": "
                  << result.lines.size() - first_line << " lines\n";
    }
    result.equilibrium_solves = engine.equilibrium_solves;
    return result;
  }
};
} // namespace

ContourResult
pseudosection_contours(const Result &previous,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const ContourConstraintFactory &constraint,
                       const ContourSettings &settings) {
  if (settings.seed_grid < 2 || settings.max_trace_steps < 1 ||
      !std::isfinite(settings.step) || !std::isfinite(settings.min_step) ||
      settings.step <= 0. || settings.step > 1. || settings.min_step <= 0. ||
      settings.min_step > settings.step)
    throw std::invalid_argument("Invalid contour seed/continuation settings.");
  if (previous.composition_start.empty())
    throw std::invalid_argument(
        "Contour calculation requires composition_start.");
  if (previous.settings.max_phase_instances < 1 || !constraint)
    throw std::invalid_argument(
        "Invalid saved phase-ID stride or constraint factory.");
  if (previous.phase_names.size() !=
      candidates.size() *
          static_cast<std::size_t>(previous.settings.max_phase_instances))
    throw std::invalid_argument(
        "Contour candidate names/order must match the saved pseudosection.");
  for (std::size_t i = 0; i < candidates.size(); ++i)
    if (!candidates[i] ||
        candidates[i]->get_name() !=
            previous
                .phase_names[i * static_cast<std::size_t>(
                                     previous.settings.max_phase_instances)])
      throw std::invalid_argument(
          "Contour candidate names/order must match the saved pseudosection.");
  auto bounds = previous.coordinate_ranges();
  for (auto r : bounds)
    if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || r[0] >= r[1])
      throw std::invalid_argument(
          "Contour diagram ranges must be finite and increasing.");
  return Contours(previous, candidates, constraint, settings).run();
}
} // namespace burnman::pseudosections
