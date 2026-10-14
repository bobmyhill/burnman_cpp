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

#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/utils/index_utils.hpp"

namespace burnman::phase_diagrams::detail {
bool unfinished_boundary(const std::string &status) {
  return status.find("failed") != std::string::npos ||
         status.find("unresolved") != std::string::npos ||
         status == "trace step limit";
}
std::pair<std::string, std::string> boundary_statuses(const Boundary &line) {
  auto split = line.termination.find(';');
  if (split == std::string::npos)
    throw std::invalid_argument(
        "Boundary termination must describe both directions.");
  auto second = line.termination.substr(split + 1);
  second.erase(0, second.find_first_not_of(' '));
  return {line.termination.substr(0, split), second};
}
void join_model_limits(Result &result) {
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]);
  const Eigen::Vector2d range(ranges[0][1] - ranges[0][0],
                              ranges[1][1] - ranges[1][0]);
  auto normalise = [&](double first, double second) -> Eigen::Vector2d {
    return (Eigen::Vector2d(first, second) - origin).cwiseQuotient(range);
  };
  for (auto &line : result.boundaries) {
    auto [back, forward] = boundary_statuses(line);
    for (bool start : {true, false}) {
      auto &id = start ? line.start_node : line.end_node;
      if (line.points.empty() ||
          (id >= 0 && result.nodes.at(static_cast<std::size_t>(id)).kind !=
                          "model_domain_limit"))
        continue;
      auto &point = start ? line.points.front() : line.points.back();
      const auto q = diagram_coordinates(point, result.section.type);
      auto u = normalise(q[0], q[1]);
      double distance = std::numeric_limits<double>::infinity();
      int region_index = -1, segment = -1;
      for (std::size_t r = 0; r < result.excluded_regions.size(); ++r) {
        auto &ring = result.excluded_regions[r];
        // Exclude the top/frame closure: only the actual EOS limit can
        // receive a terminated phase line. Keep the last thermodynamically
        // verified point, adjusting the plotted domain envelope within node
        // tolerance.
        for (int i = 1; i < ring.rows() - 2; ++i) {
          auto a = normalise(ring(i - 1, 0), ring(i - 1, 1)),
               b = normalise(ring(i, 0), ring(i, 1));
          auto delta = (b - a).eval();
          double f =
              (delta.squaredNorm() != 0.)
                  ? std::clamp((u - a).dot(delta) / delta.squaredNorm(), 0., 1.)
                  : 0.;
          double d = (u - a - f * delta).norm();
          if (d < distance) {
            distance = d;
            region_index = burnman::utils::checked_int(r);
            segment = i;
          }
        }
      }
      if (region_index < 0 || distance > result.settings.node_tolerance * 2.)
        continue;
      auto &ring =
          result.excluded_regions[static_cast<std::size_t>(region_index)];
      bool attached = false;
      for (int i = 0; i < ring.rows(); ++i)
        if ((normalise(ring(i, 0), ring(i, 1)) - u).norm() < 1.e-12)
          attached = true;
      if (!attached) {
        Eigen::MatrixXd updated(ring.rows() + 1, 2);
        updated.topRows(segment) = ring.topRows(segment);
        updated.row(segment) =
            diagram_coordinates(point, result.section.type).transpose();
        updated.bottomRows(ring.rows() - segment) =
            ring.bottomRows(ring.rows() - segment);
        ring = std::move(updated);
      }
      // Recomputed EOS envelopes must retain connections from a saved run
      // too.
      if (id < 0) {
        Node n;
        n.id = burnman::utils::checked_int(result.nodes.size());
        n.kind = "model_domain_limit";
        n.pressure = point.pressure;
        n.temperature = point.temperature;
        n.entropy = point.entropy;
        n.volume = point.volume;
        n.composition_coordinate = point.composition_coordinate;
        n.zero_phases = {line.zero_phase};
        n.assemblage = line.assemblage;
        result.nodes.push_back(n);
        id = n.id;
      }
      (start ? back : forward) = "model domain limit";
    }
    line.termination = back + "; " + forward;
  }
}
// Find the lower-pressure EOS feasibility limit. This is a boundary of
// the admissible model domain, not a zero-amount phase-equilibrium line.
std::vector<Eigen::MatrixXd>
excluded_model_regions(Engine &engine, const Eigen::Vector2d &origin,
                       const Eigen::Vector2d &range) {
  std::vector<Eigen::MatrixXd> regions;
  if (!engine.settings.exclude_invalid_eos)
    return regions;
  auto feasible = [&](double pn, double tn) {
    return engine.eos_bulk_feasible_at(
        origin + Eigen::Vector2d(pn, tn).cwiseProduct(range));
  };
  auto lower_pressure = [&](double tn) {
    if (feasible(0., tn))
      return 0.;
    if (!feasible(1., tn))
      return 1.;
    double low = 0., high = 1.;
    for (int i = 0; i < 35; ++i) {
      double middle = .5 * (low + high);
      if (feasible(middle, tn))
        high = middle;
      else
        low = middle;
    }
    return .5 * (low + high);
  };
  std::vector<Eigen::Vector2d> curve;
  auto flush = [&]() {
    if (curve.empty())
      return;
    std::vector<Eigen::Vector2d> ring{{0., curve.front()[1]}};
    ring.insert(ring.end(), curve.begin(), curve.end());
    ring.push_back({0., curve.back()[1]});
    ring.push_back(ring.front());
    Eigen::MatrixXd physical(ring.size(), 2);
    for (std::size_t i = 0; i < ring.size(); ++i)
      physical.row(static_cast<Eigen::Index>(i)) =
          (origin + ring[i].cwiseProduct(range)).transpose();
    regions.push_back(std::move(physical));
    curve.clear();
  };
  double previous = lower_pressure(0.);
  if (previous > 0.)
    curve.push_back({previous, 0.});
  for (int j = 1; j <= 64; ++j) {
    double t = double(j) / 64., left = double(j - 1) / 64.,
           p = lower_pressure(t);
    if ((previous == 0.) != (p == 0.)) {
      double low = left, high = t;
      for (int i = 0; i < 32; ++i) {
        double mid = .5 * (low + high);
        bool exists = lower_pressure(mid) > 0.;
        if (exists == (p > 0.))
          high = mid;
        else
          low = mid;
      }
      Eigen::Vector2d edge(0., .5 * (low + high));
      if (p > 0.)
        curve.push_back(edge);
      else {
        curve.push_back(edge);
        flush();
      }
    }
    if (p > 0.)
      curve.push_back({p, t});
    previous = p;
  }
  flush();
  return regions;
}

int field_variance(const Assemblage &a, std::size_t n_phases) {
  // Subtract signed counts: an overcomplete assemblage has negative variance.
  return burnman::utils::checked_int(
      static_cast<std::ptrdiff_t>(a.get_independent_element_indices().size()) -
      static_cast<std::ptrdiff_t>(n_phases) + 2);
}
void join_frame_limits(Result &result, Engine &engine) {
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]);
  const Eigen::Vector2d range(ranges[0][1] - ranges[0][0],
                              ranges[1][1] - ranges[1][0]);
  auto normalise = [&](const auto &point) -> Eigen::Vector2d {
    return (diagram_coordinates(point, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  auto inside = [](const Eigen::Vector2d &u, double margin) {
    return (u.array() >= -margin).all() && (u.array() <= 1. + margin).all();
  };

  // A corrector can already have accepted a frame point before an amount
  // null direction obscures the outgoing P,T tangent (especially at T=0).
  // Classify that accepted endpoint; do not extend an interior curve to it.
  for (auto &line : result.boundaries) {
    if (line.points.empty())
      continue;
    auto [back, forward] = boundary_statuses(line);
    for (bool start : {true, false}) {
      auto &status = start ? back : forward;
      auto &id = start ? line.start_node : line.end_node;
      if (!unfinished_boundary(status))
        continue;
      auto &point = start ? line.points.front() : line.points.back();
      auto u = normalise(point);
      if (!u.allFinite() || !inside(u, 1.e-10) ||
          std::min(u.cwiseAbs().minCoeff(),
                   (u - Eigen::Vector2d::Ones()).cwiseAbs().minCoeff()) >
              1.e-10 ||
          point.mass_balance_error > engine.settings.mass_balance_tolerance ||
          point.minimum_affinity < -engine.settings.affinity_tolerance)
        continue;
      id = -1;
      for (auto &previous : result.nodes)
        if (previous.kind == "domain_edge" &&
            (normalise(previous) - u).norm() < 1.e-10) {
          id = previous.id;
          break;
        }
      if (id < 0) {
        Node n;
        n.id = burnman::utils::checked_int(result.nodes.size());
        n.kind = "domain_edge";
        n.pressure = point.pressure;
        n.temperature = point.temperature;
        n.entropy = point.entropy;
        n.volume = point.volume;
        n.composition_coordinate = point.composition_coordinate;
        n.assemblage = line.assemblage;
        n.zero_phases = {line.zero_phase};
        auto a =
            engine.make_at(line.assemblage, point.phases,
                           diagram_coordinates(point, result.section.type));
        n.gibbs_variance = field_variance(*a, line.assemblage.size());
        result.nodes.push_back(n);
        id = n.id;
      }
      status = "domain edge";
    }
    line.termination = back + "; " + forward;
  }
}
} // namespace burnman::phase_diagrams::detail
