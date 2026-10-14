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

#include "burnman/tools/pseudosection.hpp"
#include "burnman/utils/index_utils.hpp"
#include "burnman/utils/polygon_utils.hpp"
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace burnman::pseudosections {
namespace {
using Point = Eigen::Vector2d;
using Index = std::size_t;
constexpr Index no_index = std::numeric_limits<Index>::max();
// Rescale each diagram axis to 0..1 internally; rings repeat their first index.
using Ring = std::vector<Index>;
using Phases = std::vector<int>;
using detail::cross;
using detail::segment_distance;

using Assemblages = std::set<Phases>;
using detail::canonical_phase_ids;

Assemblages boundary_assemblages(const Boundary &line, const Result &result,
                                 bool limiting = false) {
  Assemblages choices;
  if (line.assemblage.empty() || line.zero_phase < 0)
    return choices;
  const int variance = detail::boundary_variance(line, result);
  auto finite_area = [&](const Phases &ids) {
    return !ids.empty() &&
           (result.section.type != DiagramType::PT || variance < 0 ||
            static_cast<int>(ids.size()) <=
                static_cast<int>(line.assemblage.size()) + variance - 2);
  };
  auto insert = [&](const Phases &ids) {
    if (!finite_area(ids))
      return;
    choices.insert(
        canonical_phase_ids(ids, result.settings.max_phase_instances));
  };
  // A composition edge or compound reaction can replace phases absent from
  // the source assemblage. Its verified neighbours supply the alternatives.
  const auto source =
      canonical_phase_ids(line.assemblage, result.settings.max_phase_instances);
  auto outside_source = [&](const Phases &ids) {
    const auto phases =
        canonical_phase_ids(ids, result.settings.max_phase_instances);
    return !std::includes(source.begin(), source.end(), phases.begin(),
                          phases.end());
  };
  const bool replacement =
      limiting || has_composition_axis(result.section.type) ||
      !finite_area(line.assemblage) || outside_source(line.side_a) ||
      outside_source(line.side_b);
  if (replacement) {
    insert(line.side_a);
    insert(line.side_b);
  } else {
    insert(line.assemblage);
    Phases absent = line.assemblage;
    auto zero = std::find(absent.begin(), absent.end(), line.zero_phase);
    if (zero != absent.end()) {
      absent.erase(zero);
      insert(absent);
    }
  }
  return choices;
}

// Every physical arc constrains the same field. No edge gets extra authority
// because it is longer or has more stored corrector points.
void intersect_assemblages(Assemblages &common, const Assemblages &choices,
                           bool &initialized) {
  // A reaction spur with no finite limiting assemblage adds no field evidence.
  if (choices.empty())
    return;
  if (!initialized) {
    common = choices;
    initialized = true;
  } else {
    Assemblages intersection;
    std::set_intersection(common.begin(), common.end(), choices.begin(),
                          choices.end(),
                          std::inserter(intersection, intersection.end()));
    common = std::move(intersection);
  }
}

struct Edge {
  // Store both directions consecutively; index ^ 1 reverses the direction.
  // Labels describe the region on the left when following the edge.
  Index from, to, next = no_index;
  Ring path;
  std::set<Phases> labels{};
  bool solution_replacement = false;
  bool phase_boundary = false;
  bool latent_invariant = false;
  bool frame_boundary = false;
  // One retained polyline, with all reaction records merged into it.
  const Boundary *source = nullptr;
  std::map<const Boundary *, bool> observations{}; // follows record direction
};

// Infer one cyclic fan from stable side assemblages. Coarse endpoint chords
// can reverse the apparent angular order of neighbouring branches.
void link_junction(const std::vector<Index> &list, std::vector<Edge> &edges,
                   bool frame, int copies) {
  if (list.empty())
    return;
  std::vector<Index> order(list.size());
  std::iota(order.begin(), order.end(), 0);
  auto score = [&](const auto &fan) {
    std::pair<int, Index> value{0, 0};
    for (Index i = 0; i < fan.size(); ++i) {
      Index reverse = fan[i], next = fan[(i + fan.size() - 1) % fan.size()];
      const auto &a = edges[list[reverse] ^ 1].labels,
                 &b = edges[list[next]].labels;
      if (!a.empty() && !b.empty() &&
          std::none_of(a.begin(), a.end(),
                       [&](const auto &phases) { return b.count(phases); }))
        ++value.first;
      value.second += (reverse + fan.size() - next - 1) % fan.size();
    }
    return value;
  };
  auto best = order;
  auto best_score = score(best);
  // Old files may contain stale side labels. They cannot reorder stable arcs
  // when they name phases outside the boundary's own equilibrium assemblage.
  const bool reliable = std::all_of(list.begin(), list.end(), [&](Index index) {
    const auto *line = edges[index].source;
    if (!line)
      return true;
    const auto source = canonical_phase_ids(line->assemblage, copies);
    for (Index direction : {index, index ^ 1})
      for (const auto &phases : edges[direction].labels)
        if (!std::includes(source.begin(), source.end(), phases.begin(),
                           phases.end()))
          return false;
    return true;
  });
  // Parallel arcs can bound distinct thin fields. Their side assemblages,
  // rather than coarse endpoint chords, constrain the same cyclic fan. Pin
  // frame arcs so the exterior stays connected; elsewhere pin one arc to
  // avoid checking cyclic rotations of the same ordering.
  std::vector<Index> slots;
  for (Index i = 0; i < order.size(); ++i)
    if (frame ? !edges[list[i]].frame_boundary : i != 0)
      slots.push_back(i);
  auto choices = slots;
  if (reliable && order.size() <= 8)
    while (std::next_permutation(choices.begin(), choices.end())) {
      for (Index i = 0; i < slots.size(); ++i)
        order[slots[i]] = choices[i];
      auto candidate = score(order);
      if (candidate < best_score) {
        best = order;
        best_score = candidate;
      }
    }
  // A latent S/V edge can change limiting assemblage at its endpoints. Other
  // conflicting side evidence cannot justify a new cyclic ordering.
  const bool latent = std::any_of(list.begin(), list.end(), [&](Index edge) {
    return edges[edge].latent_invariant;
  });
  if (best_score.first != 0 && !latent)
    std::iota(best.begin(), best.end(), 0);
  for (Index i = 0; i < best.size(); ++i)
    edges[list[best[i]] ^ 1].next =
        list[best[(i + best.size() - 1) % best.size()]];
}

struct Walk {
  std::vector<Ring> rings;
  bool dangling = false;
};

struct Face {
  Ring exterior;
  std::vector<Ring> holes;
  double area;
  bool dangling;
};

struct Hole {
  Ring ring;
  bool dangling;
};

// Snap numerical duplicates, rather than connecting nearby unfinished traces.
class Vertices {
  double tolerance;
  std::map<std::pair<long long, long long>, std::vector<Index>> bins;

public:
  std::vector<Point> points;
  explicit Vertices(double tol) : tolerance(tol) {}
  Index find(const Point &p) const {
    long long x = static_cast<long long>(std::floor(p.x() / tolerance)),
              y = static_cast<long long>(std::floor(p.y() / tolerance));
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy) {
        auto it = bins.find({x + dx, y + dy});
        if (it != bins.end())
          for (Index index : it->second)
            if ((points[index] - p).norm() <= tolerance)
              return index;
      }
    return no_index;
  }
  Index append(const Point &p) {
    points.push_back(p);
    return points.size() - 1;
  }
  Index endpoint(const Point &p) {
    long long x = static_cast<long long>(std::floor(p.x() / tolerance)),
              y = static_cast<long long>(std::floor(p.y() / tolerance));
    Index index = append(p);
    bins[{x, y}].push_back(index);
    return index;
  }
  Index insert(const Point &p) {
    Index existing = find(p);
    return existing == no_index ? endpoint(p) : existing;
  }
};

double area(const Ring &ring, const std::vector<Point> &points) {
  double total = 0.;
  const Point origin = points[ring.front()];
  for (std::size_t i = 1; i < ring.size(); ++i)
    total += cross(Point(points[ring[i - 1]] - origin),
                   Point(points[ring[i]] - origin));
  return .5 * total;
}

// Boundary points are excluded for classification: zero-amount states can
// have either neighbouring assemblage, and chord approximation adds roundoff.
int contains(const Ring &ring, const std::vector<Point> &points, const Point &p,
             double tolerance) {
  return detail::ring_location(
      ring.size(), [&](std::size_t i) { return points[ring[i]]; }, p,
      tolerance);
}

bool contains(const Face &face, const std::vector<Point> &points,
              const Point &p, double tolerance) {
  if (contains(face.exterior, points, p, tolerance) != 1)
    return false;
  for (auto &hole : face.holes)
    if (contains(hole, points, p, tolerance) != -1)
      return false;
  return true;
}

std::vector<Ring> rings(const Ring &walk) {
  // A dangling branch is traversed twice. Remove its zero-area excursion,
  // retaining any actual exterior/hole cycles in the same face walk.
  std::vector<Ring> result;
  Ring stack;
  std::map<Index, Index> positions;
  for (Index vertex : walk) {
    auto it = positions.find(vertex);
    if (it == positions.end()) {
      positions[vertex] = stack.size();
      stack.push_back(vertex);
    } else {
      Index start = it->second;
      Ring ring(stack.begin() + static_cast<Ring::difference_type>(start),
                stack.end());
      ring.push_back(vertex);
      if (ring.size() >= 4)
        result.push_back(std::move(ring));
      for (std::size_t i = start + 1; i < stack.size(); ++i)
        positions.erase(stack[i]);
      stack.resize(start + 1);
    }
  }
  return result;
}

Eigen::MatrixXd coordinates(const Ring &ring, const std::vector<Point> &points,
                            const Point &origin, const Point &scale) {
  Eigen::MatrixXd output(ring.size(), 2);
  for (std::size_t i = 0; i < ring.size(); ++i)
    output.row(static_cast<Eigen::Index>(i)) =
        (origin + points[ring[i]].cwiseProduct(scale)).transpose();
  return output;
}

// Find an interior label point far from the edges. A centroid can lie outside
// a concave region or in a hole, so use the pole of inaccessibility.
std::pair<Point, double> label_point(const Face &face,
                                     const std::vector<Point> &points,
                                     const Point *sample = nullptr) {
  auto matrix = [&](const Ring &ring) {
    // Rings repeat their first index; the utility closes them itself.
    Eigen::MatrixXd output(static_cast<Eigen::Index>(ring.size()) - 1, 2);
    for (std::size_t i = 0; i + 1 < ring.size(); ++i)
      output.row(static_cast<Eigen::Index>(i)) = points[ring[i]].transpose();
    return output;
  };
  std::vector<Eigen::MatrixXd> rings{matrix(face.exterior)};
  for (const auto &hole : face.holes)
    rings.push_back(matrix(hole));
  std::vector<Point> seeds;
  if (sample)
    seeds.push_back(*sample);
  const auto centre = utils::visual_center_of_polygon(rings, 0., seeds);
  return {centre.point, centre.distance};
}
} // namespace

bool FieldPolygon::contains_rectangle(const Point &centre,
                                      const Point &half_size) const {
  if (!centre.allFinite() || !half_size.allFinite() ||
      (half_size.array() <= 0.).any())
    throw std::invalid_argument("Label rectangle must have finite coordinates "
                                "and positive half-sizes.");
  // Work in rectangle units: diagram coordinates can differ by many
  // orders of magnitude. Corner inclusion alone misses concave edges/holes.
  auto ring_points = [&](const Eigen::MatrixXd &ring) {
    std::vector<Point> points;
    for (int i = 0; i < ring.rows(); ++i)
      points.push_back(
          (ring.row(i).transpose() - centre).cwiseQuotient(half_size));
    return points;
  };
  auto exterior = ring_points(vertices);
  Ring indices;
  for (std::size_t i = 0; i < exterior.size(); ++i)
    indices.push_back(i);
  for (double p : {-1., 1.})
    for (double t : {-1., 1.})
      if (contains(indices, exterior, Point(p, t), 1.e-10) != 1)
        return false;
  auto crosses = [&](const std::vector<Point> &ring) {
    for (std::size_t i = 1; i < ring.size(); ++i) {
      Point delta = ring[i] - ring[i - 1];
      double low = 0., high = 1.;
      for (int k = 0; k < 2; ++k) {
        if (std::abs(delta[k]) < 1.e-15) {
          if (std::abs(ring[i - 1][k]) > 1.) {
            low = 1.;
            high = 0.;
            break;
          }
        } else {
          double a = (-1. - ring[i - 1][k]) / delta[k],
                 b = (1. - ring[i - 1][k]) / delta[k];
          low = std::max(low, std::min(a, b));
          high = std::min(high, std::max(a, b));
        }
      }
      if (low <= high)
        return true;
    }
    return false;
  };
  if (crosses(exterior))
    return false;
  for (auto &hole : holes) {
    auto points = ring_points(hole);
    indices.clear();
    for (std::size_t i = 0; i < points.size(); ++i)
      indices.push_back(i);
    if (contains(indices, points, Point::Zero(), 1.e-10) >= 0 ||
        crosses(points))
      return false;
  }
  return true;
}

FieldPolygons field_polygons(const Result &result, double tolerance,
                             bool close_domain, bool merge_fields) {
  if (!std::isfinite(tolerance) || tolerance < 1.e-12 || tolerance > 1.e-3)
    throw std::invalid_argument(
        "Polygon tolerance must be between 1e-12 and 1e-3 of the P/T domain.");
  const auto ranges = result.coordinate_ranges();
  Point origin(ranges[0][0], ranges[1][0]);
  Point scale(ranges[0][1] - origin.x(), ranges[1][1] - origin.y());
  if (!origin.allFinite() || !scale.allFinite() || (scale.array() <= 0.).any())
    throw std::invalid_argument(
        "Polygon diagram ranges must be finite and increasing.");
  auto normalise = [&](double p, double t) -> Point {
    Point value = (Point(p, t) - origin).cwiseQuotient(scale);
    if (!value.allFinite())
      throw std::invalid_argument("Polygon coordinates must be finite.");
    // Corrector roundoff at an exact frame constraint must not form slivers
    // between T=0 and a temperature a few machine epsilons above it.
    constexpr double roundoff = 64. * std::numeric_limits<double>::epsilon();
    for (int k = 0; k < 2; ++k) {
      if (std::abs(value[k]) < roundoff)
        value[k] = 0.;
      if (std::abs(value[k] - 1.) < roundoff)
        value[k] = 1.;
    }
    return value;
  };
  auto coordinate = [&](const auto &point) {
    const auto q = diagram_coordinates(point, result.section.type);
    return normalise(q[0], q[1]);
  };
  auto canonical = [](Phases values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
  };
  // Reaction identity depends on phase multiplicity, not local instance IDs.
  auto canonical_assemblage = [&](const Phases &values) {
    return canonical_phase_ids(values, result.settings.max_phase_instances);
  };
  // Calculate saved labels once; polygon reconstruction never changes samples.
  std::vector<Phases> sample_labels;
  std::set<Phases> sampled_assemblages;
  bool has_domain_sample = false;
  for (const auto &state : result.samples) {
    Phases phases;
    if (state.success && !state.outside_model_domain) {
      double total = 0.;
      for (const auto &phase : state.phases)
        total += phase.amount;
      for (const auto &phase : state.phases)
        if (phase.amount > total * result.settings.amount_tolerance)
          phases.push_back(phase.id);
    }
    sample_labels.push_back(canonical(phases));
    if (!phases.empty()) {
      sampled_assemblages.insert(sample_labels.back());
      const auto q = coordinate(state);
      // Inverse S/V solves can place frame seeds just outside by roundoff.
      has_domain_sample =
          has_domain_sample || ((q.array() >= -tolerance).all() &&
                                (q.array() <= 1. + tolerance).all());
    }
  }
  const bool uniform = has_domain_sample && sampled_assemblages.size() == 1;
  // Only saved junctions and trace endpoints connect boundaries. Interior
  // chords may cross without creating a thermodynamic vertex or another field.
  Vertices vertices(tolerance);
  std::map<int, Index> nodes;
  std::set<Index> complete_critical;
  const auto active = detail::active_boundary_nodes(result);
  for (const auto &node : result.nodes)
    if (active.count(node.id)) {
      Index vertex = vertices.insert(coordinate(node));
      nodes.emplace(node.id, vertex);
      if (node.is_verified_critical_endpoint && node.kind == "critical_point" &&
          result.section.type == DiagramType::PT && node.gibbs_variance == 1)
        complete_critical.insert(vertex);
    }
  std::vector<Edge> edges;
  std::map<std::pair<Index, Index>, Index> edge_index;
  std::set<Index> endpoints, stable_endpoints;
  auto add_edge = [&](Ring path, const Phases &left, const Phases &right,
                      bool phase_boundary, bool replacement = false,
                      bool latent_invariant = false,
                      const Boundary *source = nullptr) {
    if (path.size() < 2)
      return;
    double length = 0.;
    for (Index i = 1; i < path.size(); ++i)
      length +=
          (vertices.points[path[i]] - vertices.points[path[i - 1]]).norm();
    if (length <= tolerance)
      return;
    bool frame = false;
    if (!phase_boundary)
      for (int axis = 0; axis < 2; ++axis)
        for (double value : {0., 1.})
          frame =
              frame || std::all_of(path.begin(), path.end(), [&](Index v) {
                return std::abs(vertices.points[v][axis] - value) <= tolerance;
              });
    // Repeated records of the same complete arc are one graph edge. This
    // does not dissolve fields or compare unrelated, crossing chords.
    for (Index i = 0; i < edges.size(); i += 2) {
      Index candidate =
          edges[i].from == path.front() && edges[i].to == path.back() ? i
                                                                      : i ^ 1;
      if (path.front() == path.back() &&
          area(path, vertices.points) *
                  area(edges[candidate].path, vertices.points) <
              0.)
        candidate ^= 1;
      auto &old = edges[candidate];
      if (old.from != path.front() || old.to != path.back() ||
          old.solution_replacement != replacement)
        continue;
      // Two records of the same reaction, stable endpoints and directed
      // neighbours are one branch even when their coarse chords differ.
      const bool same_reaction =
          source && old.source &&
          canonical_assemblage(source->assemblage) ==
              canonical_assemblage(old.source->assemblage);
      const bool univariant =
          result.section.type == DiagramType::PT && same_reaction &&
          detail::boundary_variance(*source, result) == 1 &&
          detail::boundary_variance(*old.source, result) == 1;
      bool same_branch =
          same_reaction &&
          (source->zero_phase >= 0 &&
           source->zero_phase == old.source->zero_phase &&
           old.labels == std::set<Phases>{left} &&
           edges[candidate ^ 1].labels == std::set<Phases>{right});
      auto follows = [&](const Ring &a, const Ring &b, double allowance) {
        for (Index vertex : a) {
          double distance = std::numeric_limits<double>::infinity();
          for (Index j = 1; j < b.size(); ++j)
            distance =
                std::min(distance, segment_distance(vertices.points[vertex],
                                                    vertices.points[b[j - 1]],
                                                    vertices.points[b[j]]));
          if (distance > allowance)
            return false;
        }
        return true;
      };
      if (same_branch ||
          (univariant &&
           follows(path, old.path, result.settings.node_tolerance) &&
           follows(old.path, path, result.settings.node_tolerance)) ||
          (follows(path, old.path, tolerance) &&
           follows(old.path, path, tolerance))) {
        if (!left.empty())
          old.labels.insert(left);
        if (!right.empty())
          edges[candidate ^ 1].labels.insert(right);
        old.phase_boundary = edges[candidate ^ 1].phase_boundary =
            old.phase_boundary || phase_boundary;
        old.frame_boundary = edges[candidate ^ 1].frame_boundary =
            old.frame_boundary || frame;
        if (source) {
          old.observations[source] = true;
          edges[candidate ^ 1].observations[source] = false;
        }
        return;
      }
    }
    Index index = edges.size();
    Edge forward{path.front(), path.back(), no_index, path};
    std::reverse(path.begin(), path.end());
    Edge backward{path.front(), path.back(), no_index, std::move(path)};
    if (!left.empty())
      forward.labels.insert(left);
    if (!right.empty())
      backward.labels.insert(right);
    forward.phase_boundary = backward.phase_boundary = phase_boundary;
    forward.solution_replacement = backward.solution_replacement = replacement;
    forward.latent_invariant = backward.latent_invariant = latent_invariant;
    forward.source = backward.source = source;
    if (source) {
      forward.observations[source] = true;
      backward.observations[source] = false;
    }
    forward.frame_boundary = backward.frame_boundary = frame;
    endpoints.insert(forward.from);
    endpoints.insert(forward.to);
    edges.push_back(std::move(forward));
    edges.push_back(std::move(backward));
    for (Index j = 1; j < edges[index].path.size(); ++j) {
      Index a = edges[index].path[j - 1], b = edges[index].path[j];
      edge_index.emplace(std::make_pair(a, b), index);
      edge_index.emplace(std::make_pair(b, a), index ^ 1);
    }
  };
  // Sampling detail chooses duplicate geometry only; every record still
  // contributes perimeter evidence. Canonical geometry/reaction order also
  // fixes endpoint snapping, tied fans and face numbering without using IDs.
  auto line_key = [&](const Boundary *line) {
    std::vector<std::array<double, 2>> path;
    for (const auto &point : line->points) {
      const auto p = coordinate(point);
      path.push_back({p.x(), p.y()});
    }
    for (int end : {0, 1}) {
      const auto it = nodes.find(end ? line->end_node : line->start_node);
      if (it != nodes.end() && !path.empty()) {
        const auto &p = vertices.points[it->second];
        path[end ? path.size() - 1 : 0] = {p.x(), p.y()};
      }
    }
    auto reverse = path;
    std::reverse(reverse.begin(), reverse.end());
    auto left = canonical(line->side_b), right = canonical(line->side_a);
    const bool reversed = reverse < path;
    if (reversed) {
      path = std::move(reverse);
      std::swap(left, right);
    }
    return std::tuple{std::move(path),
                      canonical_assemblage(line->assemblage),
                      line->zero_phase,
                      std::move(left),
                      std::move(right),
                      line->is_solution_replacement,
                      detail::boundary_variance(*line, result),
                      reversed};
  };
  std::vector<const Boundary *> lines;
  for (const auto &line : result.boundaries)
    lines.push_back(&line);
  std::sort(lines.begin(), lines.end(), [&](auto a, auto b) {
    return a->points.size() != b->points.size()
               ? a->points.size() > b->points.size()
               : line_key(a) < line_key(b);
  });
  for (const auto *record : lines) {
    const auto &line = *record;
    if (line.points.size() < 2)
      continue;
    std::vector<Point> points;
    for (const auto &point : line.points) {
      auto value = coordinate(point);
      if ((value.array() < -tolerance * 10).any() ||
          (value.array() > 1. + tolerance * 10).any())
        throw std::invalid_argument(
            "Phase-line coordinates lie outside the calculation domain.");
      points.push_back(value.cwiseMax(Point::Zero()).cwiseMin(Point::Ones()));
    }
    auto endpoint = [&](int id, const Point &p) {
      auto it = nodes.find(id);
      Index vertex = it == nodes.end() ? vertices.insert(p) : it->second;
      if (it != nodes.end())
        stable_endpoints.insert(vertex);
      return vertex;
    };
    Ring path{endpoint(line.start_node, points.front())};
    for (Index i = 1; i + 1 < points.size(); ++i)
      path.push_back(vertices.append(points[i]));
    path.push_back(endpoint(line.end_node, points.back()));
    auto invariant_node = [&](int id) -> const Node * {
      for (const auto &node : result.nodes)
        if (node.id == id && node.kind == "junction" &&
            node.gibbs_variance == 0 && node.pt_nullity == 0)
          return &node;
      return nullptr;
    };
    const auto *start = invariant_node(line.start_node),
               *end = invariant_node(line.end_node);
    const bool latent =
        (result.section.type == DiagramType::PS ||
         result.section.type == DiagramType::TV) &&
        start && end &&
        canonical_assemblage(start->assemblage) ==
            canonical_assemblage(end->assemblage) &&
        std::abs(start->pressure - end->pressure) <= .2 &&
        std::abs(start->temperature - end->temperature) <= 1.e-6;
    // In native axis order, side_b is left and side_a is right.
    // Some saved traces pass through another stable vertex without ending
    // there (notably invariant lines in T-X). Split only at saved vertices,
    // never at intersections discovered between arbitrary phase chords.
    Ring arc{path.front()};
    for (Index i = 1; i < path.size(); ++i) {
      const Point a = vertices.points[path[i - 1]],
                  b = vertices.points[path[i]];
      const Point delta = b - a;
      std::vector<std::pair<double, Index>> cuts;
      if (delta.squaredNorm() > 0.)
        for (const auto &node : result.nodes) {
          if (!active.count(node.id))
            continue;
          Index vertex = nodes.at(node.id);
          if (vertex == path.front() || vertex == path.back())
            continue;
          const auto event = canonical_assemblage(node.assemblage);
          const auto assemblage = canonical_assemblage(line.assemblage);
          // Solution-copy IDs can be exchanged between accepted states on a
          // univariant reaction. Match their physical phase at a saved node.
          const bool zero_at_node = std::any_of(
              node.zero_phases.begin(), node.zero_phases.end(), [&](int zero) {
                return zero == line.zero_phase ||
                       (node.gibbs_variance == 1 && node.pt_nullity == 0 &&
                        event == assemblage &&
                        zero / result.settings.max_phase_instances ==
                            line.zero_phase /
                                result.settings.max_phase_instances);
              });
          if (!std::includes(event.begin(), event.end(), assemblage.begin(),
                             assemblage.end()) ||
              (line.zero_phase >= 0 && !zero_at_node))
            continue;
          const Point p = vertices.points[vertex];
          double along = (p - a).dot(delta) / delta.squaredNorm();
          if (along > 0. && along <= 1. &&
              segment_distance(p, a, b) <= tolerance)
            cuts.emplace_back(along, vertex);
        }
      std::sort(cuts.begin(), cuts.end());
      for (const auto &[along, vertex] : cuts) {
        arc.push_back(vertex);
        stable_endpoints.insert(vertex);
        add_edge(arc, canonical(line.side_b), canonical(line.side_a), true,
                 line.is_solution_replacement, latent, &line);
        arc = {vertex};
      }
      // Repeated endpoint coordinates still need the saved vertex ID.
      if (arc.back() != path[i])
        arc.push_back(path[i]);
    }
    add_edge(std::move(arc), canonical(line.side_b), canonical(line.side_a),
             true, line.is_solution_replacement, latent, &line);
  }
  std::vector<std::vector<Point>> excluded;
  auto on_frame = [&](const Point &a, const Point &b) {
    for (int k = 0; k < 2; ++k)
      for (double value : {0., 1.})
        if (std::abs(a[k] - value) <= tolerance &&
            std::abs(b[k] - value) <= tolerance)
          return true;
    return false;
  };
  for (const auto &region : result.excluded_regions) {
    std::vector<Point> points;
    Ring path;
    for (int i = 0; i < region.rows(); ++i) {
      points.push_back(normalise(region(i, 0), region(i, 1)));
      path.push_back(vertices.insert(points.back()));
    }
    for (Index i = 1; i < path.size(); ++i) {
      endpoints.insert(path[i - 1]);
      endpoints.insert(path[i]);
      if (!close_domain || !on_frame(points[i - 1], points[i]))
        add_edge({path[i - 1], path[i]}, {}, {}, false);
    }
    excluded.push_back(std::move(points));
  }
  if (close_domain) {
    // Frame arcs close edge fields at their existing endpoints. They never
    // cut a phase line or join an unfinished interior trace to the frame.
    for (const Point &corner :
         {Point(0., 0.), Point(1., 0.), Point(1., 1.), Point(0., 1.)})
      endpoints.insert(vertices.insert(corner));
    for (int side = 0; side < 4; ++side) {
      const int fixed = side % 2 ? 0 : 1;
      const double value = side == 1 || side == 2 ? 1. : 0.;
      std::vector<Index> ordered;
      for (Index vertex : endpoints)
        if (std::abs(vertices.points[vertex][fixed] - value) <= tolerance)
          ordered.push_back(vertex);
      std::sort(ordered.begin(), ordered.end(), [&](Index a, Index b) {
        return side < 2 ? vertices.points[a][1 - fixed] <
                              vertices.points[b][1 - fixed]
                        : vertices.points[a][1 - fixed] >
                              vertices.points[b][1 - fixed];
      });
      for (Index i = 1; i < ordered.size(); ++i)
        add_edge({ordered[i - 1], ordered[i]}, {}, {}, false);
    }
  }
  std::vector<std::vector<Index>> outgoing(vertices.points.size());
  for (std::size_t i = 0; i < edges.size(); ++i)
    outgoing[edges[i].from].push_back(i);
  for (auto &list : outgoing) {
    std::sort(list.begin(), list.end(), [&](Index a, Index b) {
      auto direction = [&](Index edge) -> Point {
        const auto &path = edges[edge].path;
        for (Index i = 1; i < path.size(); ++i) {
          Point delta = vertices.points[path[i]] - vertices.points[path[0]];
          if (delta.norm() > tolerance) {
            // The exterior frame connects to itself. A phase arc tangent to
            // it belongs to the interior fan, even when coarse chords overlap
            // the frame. Only the ordering changes; saved points stay intact.
            if (!edges[edge].frame_boundary)
              for (int axis = 0; axis < 2; ++axis) {
                if (std::abs(vertices.points[path[0]][axis]) <= tolerance)
                  delta[axis] =
                      std::max(std::abs(delta[axis]), delta.norm() * 1.e-12);
                if (std::abs(vertices.points[path[0]][axis] - 1.) <= tolerance)
                  delta[axis] =
                      -std::max(std::abs(delta[axis]), delta.norm() * 1.e-12);
              }
            return delta;
          }
        }
        return Point::Zero();
      };
      Point da = direction(a), db = direction(b);
      return std::atan2(da.y(), da.x()) < std::atan2(db.y(), db.x());
    });
    if (!list.empty()) {
      const auto p = vertices.points[edges[list.front()].from];
      bool frame = (p.array().abs() <= tolerance).any() ||
                   ((p.array() - 1.).abs() <= tolerance).any();
      link_junction(list, edges, frame, result.settings.max_phase_instances);
    }
  }
  std::vector<Walk> walks;
  std::vector<bool> visited(edges.size(), false);
  for (std::size_t start = 0; start < edges.size(); ++start)
    if (!visited[start]) {
      Walk walk;
      Ring path;
      Index current = start;
      do {
        visited[current] = true;
        for (Index vertex : {edges[current].from, edges[current].to}) {
          const auto p = vertices.points[vertex];
          const bool frame = (p.array().abs() <= tolerance).any() ||
                             ((p.array() - 1.).abs() <= tolerance).any();
          walk.dangling =
              walk.dangling || (outgoing[vertex].size() == 1 && !frame &&
                                !complete_critical.count(vertex));
        }
        const auto &arc = edges[current].path;
        path.insert(path.end(), arc.begin(), arc.end() - 1);
        current = edges[current].next;
      } while (current != start);
      path.push_back(edges[start].from);
      walk.rings = rings(path);
      walks.push_back(std::move(walk));
    }
  std::vector<Face> faces;
  std::vector<Hole> holes;
  const double area_tolerance = tolerance * tolerance * 4.;
  for (auto &walk : walks)
    for (auto &ring : walk.rings) {
      bool latent = true;
      for (Index i = 1; i < ring.size(); ++i)
        latent = latent &&
                 edges[edge_index.at({ring[i - 1], ring[i]})].latent_invariant;
      // A zero-variance P-S/T-V invariant has no two-dimensional interior.
      // Classify its loop without aligning or moving any saved curve.
      if (latent)
        continue;
      double value = area(ring, vertices.points);
      // Ignore zero-area excursions using roundoff scaled by ring length.
      double perimeter = 0.;
      for (std::size_t i = 1; i < ring.size(); ++i)
        perimeter +=
            (vertices.points[ring[i]] - vertices.points[ring[i - 1]]).norm();
      // Intersections inherit absolute coordinate error on the normalised
      // 0..1 axes, even near the origin. Use the same roundoff band as frame
      // and verified-interior classification, rather than scaling it down
      // again by the coordinates of a tiny ring.
      const double allowance =
          std::max(area_tolerance,
                   64. * std::numeric_limits<double>::epsilon() * perimeter);
      if (value > allowance)
        faces.push_back({ring, {}, value, walk.dangling});
      else if (value < -allowance)
        holes.push_back({ring, walk.dangling});
    }
  // Clockwise cycles belong to the smallest containing exterior, excluding
  // their own equal-area counterpart. This handles islands and nested holes.
  // Testing only the first vertex can assign a hole to a neighbouring face
  // touching that vertex. Check the whole split ring and require an interior
  // point; a shared corner alone is not containment.
  auto encloses = [&](const Ring &exterior, const Ring &ring) {
    bool interior = false;
    for (std::size_t i = 1; i < ring.size(); ++i) {
      for (Point point : {vertices.points[ring[i - 1]],
                          Point(.5 * (vertices.points[ring[i - 1]] +
                                      vertices.points[ring[i]]))}) {
        int location = contains(exterior, vertices.points, point, tolerance);
        if (location < 0)
          return false;
        interior = interior || location == 1;
      }
    }
    return interior;
  };
  for (auto &[ring, dangling] : holes) {
    double hole_area = -area(ring, vertices.points);
    Index parent = no_index;
    for (std::size_t i = 0; i < faces.size(); ++i)
      if (faces[i].area > hole_area + area_tolerance &&
          encloses(faces[i].exterior, ring) &&
          (parent == no_index || faces[i].area < faces[parent].area))
        parent = i;
    if (parent != no_index) {
      faces[parent].holes.push_back(ring);
      faces[parent].dangling = faces[parent].dangling || dangling;
    }
  }
  auto ring_edges = [&](const Ring &ring, auto action) {
    for (std::size_t j = 1; j < ring.size(); ++j)
      action(edge_index.at({ring[j - 1], ring[j]}));
  };
  std::map<Phases, int> field_ids;
  for (auto &field : result.fields)
    field_ids.emplace(canonical(field.phases), field.id);
  FieldPolygons output;
  for (auto &face : faces) {
    FieldPolygon polygon;
    polygon.field_number =
        burnman::utils::checked_int(output.polygons.size() + 1);
    polygon.source_regions.push_back(polygon.field_number - 1);
    polygon.has_open_boundary = face.dangling;
    polygon.area = face.area;
    polygon.vertices =
        coordinates(face.exterior, vertices.points, origin, scale);
    for (auto &ring : face.holes) {
      polygon.holes.push_back(
          coordinates(ring, vertices.points, origin, scale));
      polygon.area += area(ring, vertices.points);
    }
    // An interior point identifies regions excluded by the equation of state.
    if (!excluded.empty()) {
      auto pole = label_point(face, vertices.points);
      for (auto &region : excluded) {
        Ring ring;
        for (std::size_t i = 0; i < region.size(); ++i)
          ring.push_back(i);
        if (contains(ring, region, pole.first, 0.) == 1)
          polygon.outside_model_domain = true;
      }
      if (polygon.outside_model_domain) {
        polygon.label_position = origin + pole.first.cwiseProduct(scale);
        polygon.label_clearance = pole.second;
        output.polygons.push_back(std::move(polygon));
        continue;
      }
    }
    Point low = vertices.points[face.exterior[0]], high = low;
    for (Index vertex : face.exterior) {
      low = low.cwiseMin(vertices.points[vertex]);
      high = high.cwiseMax(vertices.points[vertex]);
    }
    Phases sample_phases;
    double clearance = -1.;
    for (std::size_t index = 0; index < result.samples.size(); ++index) {
      const auto &state = result.samples[index];
      if (!state.success)
        continue;
      Point location = coordinate(state);
      if ((location.array() < low.array()).any() ||
          (location.array() > high.array()).any())
        continue;
      // The snapping tolerance is inappropriate as an exclusion band around
      // a tiny face. Its explicitly verified interior solve can be classified
      // using floating-point precision, while general samples keep the band.
      double sample_tolerance =
          state.is_field_verification
              ? std::min(tolerance * 2.,
                         64. * std::numeric_limits<double>::epsilon())
              : tolerance * 2.;
      if (!contains(face, vertices.points, location, sample_tolerance)) {
        continue;
      }
      const auto &phases = sample_labels[index];
      if (phases.empty())
        continue;
      double margin = std::numeric_limits<double>::infinity();
      auto measure = [&](const Ring &ring) {
        for (std::size_t i = 1; i < ring.size(); ++i)
          margin = std::min(
              margin, segment_distance(location, vertices.points[ring[i - 1]],
                                       vertices.points[ring[i]]));
      };
      measure(face.exterior);
      for (auto &hole : face.holes)
        measure(hole);
      if (margin > clearance) {
        clearance = margin;
        sample_phases = canonical(phases);
        polygon.sample_index = burnman::utils::checked_int(index);
      }
    }
    // The exterior loop is the primary evidence. Internal holes and spurs
    // can expose unfinished topology without suppressing its surrounding label.
    Assemblages exterior_labels, perimeter;
    std::array<Assemblages, 3> candidates;
    std::array<bool, 3> constrained{};
    std::set<const Boundary *> sources, observations;
    std::set<Index> perimeter_edges;
    auto collect = [&](Index edge) { perimeter_edges.insert(edge); };
    ring_edges(face.exterior, collect);
    for (const auto &hole : face.holes)
      ring_edges(hole, collect);
    bool physical_perimeter = false, unknown_side = false;
    auto evidence = [&](const Ring &ring, bool exterior) {
      ring_edges(ring, [&](Index edge) {
        // Traversing both directions is a spur, not a separating perimeter.
        if (perimeter_edges.count(edge ^ 1))
          return;
        physical_perimeter = physical_perimeter || edges[edge].phase_boundary;
        unknown_side = unknown_side || (edges[edge].phase_boundary &&
                                        edges[edge].labels.empty());
        if (edges[edge].source)
          sources.insert(edges[edge].source);
        for (const auto &[record, follows] : edges[edge].observations) {
          if (!observations.insert(record).second)
            continue;
          const auto &source = *record;
          // A P-S/T-V invariant can change limiting assemblage along its
          // latent interval. Its global phase-out pair cannot constrain
          // every adjacent face; retain side evidence and use the other
          // perimeter reactions to identify each finite field.
          if (edges[edge].latent_invariant)
            continue;
          intersect_assemblages(candidates[0],
                                boundary_assemblages(source, result),
                                constrained[0]);
          if (result.section.type == DiagramType::PT) {
            const auto limiting = boundary_assemblages(source, result, true);
            intersect_assemblages(candidates[1], limiting, constrained[1]);
            if (detail::boundary_variance(source, result) != 1)
              intersect_assemblages(candidates[2], limiting, constrained[2]);
          }
        }
        for (const auto &phases : edges[edge].labels) {
          if (phases.empty())
            continue;
          perimeter.insert(phases);
          if (exterior)
            exterior_labels.insert(phases);
        }
      });
    };
    evidence(face.exterior, true);
    for (const auto &hole : face.holes)
      evidence(hole, false);
    for (const auto *source : observations)
      if (source->id >= 0)
        polygon.boundary_ids.push_back(source->id);
    std::sort(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
    // Finite probes may skip thin fields, so first use the phase-out choices.
    // A compound reaction can instead replace a different limiting zero phase;
    // require agreement around the entire loop before accepting that
    // alternative.
    Phases common;
    for (const auto &choices : candidates)
      if (choices.size() == 1) {
        common = *choices.begin();
        break;
      }
    bool loop_identified = false;
    // Two records of one univariant curve bound a field of zero width. Its
    // perimeter labels conflict, so label it with every phase present along
    // the curve. Genuine thin fields have consistent labels and skip this.
    Phases univariant;
    if (exterior_labels.size() != 1 && sources.size() == 2) {
      const auto &p = **sources.begin(), &q = **sources.rbegin();
      const auto p_phases = canonical_assemblage(p.assemblage),
                 q_phases = canonical_assemblage(q.assemblage);
      if (p.start_node >= 0 && p.end_node >= 0 && p.start_node != p.end_node &&
          p_phases != q_phases &&
          ((p.start_node == q.start_node && p.end_node == q.end_node) ||
           (p.start_node == q.end_node && p.end_node == q.start_node))) {
        // Both records must trace the same curve, not just join the same
        // nodes; otherwise this is a genuine field between distinct curves.
        auto lies_on = [&](const Boundary &one, const Boundary &two) {
          if (one.points.empty() || two.points.size() < 2)
            return false;
          for (const auto &point : one.points) {
            double distance = std::numeric_limits<double>::infinity();
            for (std::size_t k = 1; k < two.points.size(); ++k)
              distance = std::min(
                  distance, segment_distance(coordinate(point),
                                             coordinate(two.points[k - 1]),
                                             coordinate(two.points[k])));
            if (distance > result.settings.node_tolerance * 2.)
              return false;
          }
          return true;
        };
        if (lies_on(p, q) && lies_on(q, p)) {
          univariant = p_phases;
          univariant.insert(univariant.end(), q_phases.begin(), q_phases.end());
          univariant = canonical(std::move(univariant));
        }
      }
    }
    if (!univariant.empty()) {
      polygon.phases = std::move(univariant);
      loop_identified = true;
    } else if (exterior_labels.size() == 1)
      polygon.phases = *exterior_labels.begin();
    else if (exterior_labels.empty() && perimeter.size() == 1)
      polygon.phases = *perimeter.begin();
    else if (!face.dangling && !common.empty()) {
      polygon.phases = std::move(common);
      loop_identified = true;
    } else if (!sample_phases.empty())
      polygon.phases = sample_phases;
    else if (perimeter.empty()) {
      // No usable loop or interior sample: retain a provisional label from
      // the closest successful saved minimization. This performs no solves.
      const auto pole = label_point(face, vertices.points).first;
      double distance = std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < result.samples.size(); ++i) {
        if (sample_labels[i].empty())
          continue;
        const double next =
            (coordinate(result.samples[i]) - pole).squaredNorm();
        if (next < distance) {
          polygon.phases = sample_labels[i];
          distance = next;
        }
      }
    }
    if (sample_phases != polygon.phases)
      polygon.sample_index = -1;
    polygon.n_phases = burnman::utils::checked_int(polygon.phases.size());
    const auto identified = field_ids.find(polygon.phases);
    if (identified != field_ids.end())
      polygon.field_id = identified->second;
    polygon.is_provisional =
        face.dangling ||
        (!loop_identified && (unknown_side || perimeter.size() > 1)) ||
        (perimeter.empty() && (physical_perimeter || !uniform));
    if (polygon.is_provisional || polygon.phases.empty())
      output.diagnostics.push_back(
          "Field " + std::to_string(polygon.field_number) +
          (polygon.phases.empty() && perimeter.size() > 1
               ? " has conflicting perimeter assemblages; no consistent label "
                 "was found."
           : polygon.phases.empty() ? " has no assemblage evidence."
           : face.dangling
               ? " contains an unfinished boundary; its label is provisional."
           : unknown_side
               ? " has an unlabelled perimeter side; its label is provisional."
           : perimeter.empty() ? " has no labelled perimeter; its saved-sample "
                                 "label is provisional."
                               : " has conflicting perimeter assemblages; its "
                                 "label is provisional."));
    Point sample;
    if (polygon.sample_index >= 0) {
      auto &state =
          result.samples[static_cast<std::size_t>(polygon.sample_index)];
      sample = coordinate(state);
    }
    auto label = label_point(face, vertices.points,
                             polygon.sample_index >= 0 ? &sample : nullptr);
    polygon.label_position = origin + label.first.cwiseProduct(scale);
    polygon.label_clearance = label.second;
    output.polygons.push_back(std::move(polygon));
  }
  // Assign actual rings; dangling excursions remain unassigned.
  std::vector<Index> edge_faces(edges.size(), no_index);
  for (std::size_t i = 0; i < faces.size(); ++i) {
    auto assign = [&](Index edge) { edge_faces[edge] = i; };
    ring_edges(faces[i].exterior, assign);
    for (auto &hole : faces[i].holes)
      ring_edges(hole, assign);
  }
  // Closed loops are the fields. Do not dissolve them into a planar union.
  (void)merge_fields; // Retained for source/API compatibility.
  std::vector<Phases> face_phases;
  for (const auto &polygon : output.polygons)
    face_phases.push_back(polygon.is_provisional ? Phases{} : polygon.phases);
  std::vector<bool> removed(edges.size(), false);
  std::vector<int> degree(vertices.points.size(), 0);
  for (Index i = 0; i < edges.size(); i += 2) {
    const auto &path = edges[i].path;
    bool masked = false;
    Point middle = .5 * (vertices.points[path[0]] + vertices.points[path[1]]);
    for (const auto &region : excluded) {
      Ring ring;
      for (Index j = 0; j < region.size(); ++j)
        ring.push_back(j);
      masked = masked || contains(ring, region, middle, tolerance) == 1;
    }
    if (masked) {
      removed[i] = removed[i ^ 1] = true;
      continue;
    }
    for (Index j = 1; j < path.size(); ++j)
      output.boundary_segments.push_back(
          coordinates({path[j - 1], path[j]}, vertices.points, origin, scale));
    ++degree[edges[i].from];
    ++degree[edges[i].to];
  }
  // Reuse graph adjacency rather than probing across an entire thin field.
  // Every arc of a record must have the same trusted neighbour on each side.
  std::map<int, std::array<Assemblages, 2>> adjacent;
  std::set<int> incomplete;
  for (Index i = 0; i < edges.size(); ++i)
    for (const auto &[record, follows] : edges[i].observations) {
      const Index face = edge_faces[i];
      if (removed[i] || face == no_index || face_phases[face].empty())
        incomplete.insert(record->id);
      else
        adjacent[record->id][follows ? 1 : 0].insert(face_phases[face]);
    }
  for (const auto &line : result.boundaries) {
    const auto it = adjacent.find(line.id);
    if (line.id < 0 || line.is_solution_replacement ||
        incomplete.count(line.id) || it == adjacent.end() ||
        it->second[0].size() != 1 || it->second[1].size() != 1)
      continue;
    const auto right = canonical_phase_ids(*it->second[0].begin(),
                                           result.settings.max_phase_instances),
               left = canonical_phase_ids(*it->second[1].begin(),
                                          result.settings.max_phase_instances);
    const auto choices = boundary_assemblages(line, result);
    if (right != left && choices.count(right) && choices.count(left))
      output.boundary_sides.emplace(line.id,
                                    std::array<Phases, 2>{right, left});
  }
  std::vector<bool> endpoint(vertices.points.size(), false);
  for (Index vertex : stable_endpoints)
    endpoint[vertex] = true;
  for (const auto &[id, vertex] : nodes)
    if (degree[vertex] >= 3)
      output.boundary_nodes.push_back(id);
  // Keep each continuous boundary as one stroke, with its original vertices.
  // Verified fields identify the neighbours; otherwise use unambiguous edge
  // labels. Unknown sides and junctions cannot establish a continuation.
  const Phases unknown;
  auto side = [&](Index edge) -> const Phases & {
    Index face = edge_faces[edge];
    if (face != no_index && !face_phases[face].empty())
      return face_phases[face];
    if (edges[edge].labels.size() == 1)
      return *edges[edge].labels.begin();
    return unknown;
  };
  auto same_assemblages = [&](Index first, Index second) {
    const auto &a = side(first), &b = side(first ^ 1), &c = side(second),
               &d = side(second ^ 1);
    return !a.empty() && !b.empty() &&
           ((a == c && b == d) || (a == d && b == c));
  };
  auto continuation = [&](Index edge) {
    Index vertex = edges[edge].to;
    if (!endpoint[vertex] && degree[vertex] == 2)
      for (Index next : outgoing[vertex])
        if (next != (edge ^ 1) && !removed[next] &&
            same_assemblages(edge, next))
          return next;
    return no_index;
  };
  std::vector<bool> seen(edges.size(), false);
  auto join = [&](Index edge) {
    Ring path{edges[edge].from};
    while (edge != no_index && !seen[edge]) {
      seen[edge] = seen[edge ^ 1] = true;
      const auto &arc = edges[edge].path;
      path.insert(path.end(), arc.begin() + 1, arc.end());
      edge = continuation(edge);
    }
    output.boundary_lines.push_back(
        coordinates(path, vertices.points, origin, scale));
  };
  // Start at the ends first, then walk any remaining closed loops.
  for (Index edge = 0; edge < edges.size(); ++edge)
    if (!removed[edge] && !seen[edge] && continuation(edge ^ 1) == no_index)
      join(edge);
  for (Index edge = 0; edge < edges.size(); edge += 2)
    if (!removed[edge] && !seen[edge])
      join(edge);
  return output;
}
} // namespace burnman::pseudosections
