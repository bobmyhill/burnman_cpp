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
#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
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
// Copy numbers identify instances locally. Compare multiplicity, retaining
// two coexisting solutions even when the vanishing copy is numbered first.
Phases instance_ids(const Phases &phases, int copies) {
  std::map<int, int> counts;
  for (int id : phases)
    ++counts[id / copies];
  Phases result;
  for (auto [base, count] : counts)
    for (int copy = 0; copy < count; ++copy)
      result.push_back(base * copies + copy);
  return result;
}

Assemblages boundary_assemblages(const Boundary &line, const Result &result) {
  Assemblages choices;
  if (line.assemblage.empty() || line.zero_phase < 0)
    return choices;
  auto finite_area = [&](const Phases &ids) {
    if (result.section.type == DiagramType::PT)
      for (const auto &node : result.nodes)
        if ((node.id == line.start_node || node.id == line.end_node) &&
            !node.kind.empty() && !node.assemblage.empty() &&
            node.gibbs_variance >= 0 &&
            static_cast<int>(ids.size()) >
                node.gibbs_variance + static_cast<int>(node.assemblage.size()) -
                    2)
          return false; // A univariant/invariant assemblage has no P-T field
                        // area.
    return !ids.empty();
  };
  auto insert = [&](const Phases &ids) {
    if (!finite_area(ids))
      return;
    choices.insert(instance_ids(ids, result.settings.max_phase_instances));
  };
  // A composition edge or compound reaction can replace phases absent from
  // the source assemblage. Its verified neighbours supply the alternatives.
  const bool replacement =
      has_composition_axis(result.section.type) ||
      !finite_area(line.assemblage) ||
      std::any_of(line.side_a.begin(), line.side_a.end(),
                  [&](int id) {
                    return std::find(line.assemblage.begin(),
                                     line.assemblage.end(),
                                     id) == line.assemblage.end();
                  }) ||
      std::any_of(line.side_b.begin(), line.side_b.end(), [&](int id) {
        return std::find(line.assemblage.begin(), line.assemblage.end(), id) ==
               line.assemblage.end();
      });
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
  const Boundary *source = nullptr;
};

// Infer one cyclic fan from stable side assemblages. Coarse endpoint chords
// can reverse the apparent angular order of neighbouring branches.
void link_junction(const std::vector<Index> &list, std::vector<Edge> &edges,
                   bool frame) {
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
  std::set<Index> neighbours;
  for (Index edge : list)
    neighbours.insert(edges[edge].to);
  // Old files may contain stale side labels. They cannot reorder stable arcs
  // when they name phases outside the boundary's own equilibrium assemblage.
  const bool reliable = std::all_of(list.begin(), list.end(), [&](Index index) {
    const auto *line = edges[index].source;
    if (!line)
      return true;
    for (Index direction : {index, index ^ 1})
      for (const auto &phases : edges[direction].labels)
        for (int phase : phases)
          if (std::find(line->assemblage.begin(), line->assemblage.end(),
                        phase) == line->assemblage.end())
            return false;
    return true;
  });
  // Preserve the frame exterior and the geometric order of parallel arcs
  // bounding thin fields. Bound the search at unusually crowded junctions.
  if (!frame && reliable && order.size() <= 8 &&
      neighbours.size() == list.size())
    while (std::next_permutation(order.begin() + 1, order.end())) {
      auto candidate = score(order);
      if (candidate < best_score) {
        best = order;
        best_score = candidate;
      }
    }
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

// Find an interior label point far from the edges by repeatedly subdividing
// promising squares. A centroid can lie outside a concave region or in a hole.
std::pair<Point, double> label_point(const Face &face,
                                     const std::vector<Point> &points,
                                     const Point *sample = nullptr) {
  auto signed_distance = [&](const Point &p) {
    double d = std::numeric_limits<double>::infinity();
    auto measure = [&](const Ring &ring) {
      for (std::size_t i = 1; i < ring.size(); ++i)
        d = std::min(d,
                     segment_distance(p, points[ring[i - 1]], points[ring[i]]));
    };
    measure(face.exterior);
    for (auto &ring : face.holes)
      measure(ring);
    return contains(face, points, p, 0.) ? d : -d;
  };
  Point low = points[face.exterior[0]], high = low;
  for (Index i : face.exterior) {
    low = low.cwiseMin(points[i]);
    high = high.cwiseMax(points[i]);
  }
  struct Cell {
    Point centre;
    double half, d, maximum;
    bool operator<(const Cell &other) const { return maximum < other.maximum; }
  };
  auto cell = [&](Point p, double half) {
    double d = signed_distance(p);
    return Cell{p, half, d, d + half * std::sqrt(2.)};
  };
  auto best = cell(points[face.exterior[0]], 0.);
  Point centroid = Point::Zero();
  double weight = 0.;
  for (std::size_t i = 1; i < face.exterior.size(); ++i) {
    auto &a = points[face.exterior[i - 1]];
    auto &b = points[face.exterior[i]];
    double w = cross(a, b);
    centroid += (a + b) * w;
    weight += w;
  }
  if (std::abs(weight) > 1.e-20) {
    auto candidate = cell(centroid / (3. * weight), 0.);
    if (candidate.d > best.d)
      best = candidate;
  }
  if (sample) {
    auto candidate = cell(*sample, 0.);
    if (candidate.d > best.d)
      best = candidate;
  }
  // Thin chord slivers can lose centroid accuracy through cancellation of
  // signed area. Convex-corner triangle centres provide stable interior seeds.
  for (std::size_t i = 1; i + 1 < face.exterior.size(); ++i) {
    auto candidate =
        cell((points[face.exterior[i - 1]] + points[face.exterior[i]] +
              points[face.exterior[i + 1]]) /
                 3.,
             0.);
    if (candidate.d > best.d)
      best = candidate;
  }
  auto root = cell(.5 * (low + high), .5 * (high - low).maxCoeff());
  if (root.d > best.d)
    best = root;
  std::priority_queue<Cell> queue;
  queue.push(root);
  double precision = std::max(1.e-7, (high - low).minCoeff() * .002);
  for (int iteration = 0; !queue.empty() && iteration < 20000; ++iteration) {
    auto current = queue.top();
    queue.pop();
    if (current.d > best.d)
      best = current;
    if (current.maximum - best.d <= precision)
      continue;
    double h = current.half * .5;
    for (double x : {-h, h})
      for (double y : {-h, h})
        queue.push(cell(current.centre + Point(x, y), h));
  }
  return {best.centre, std::max(0., best.d)};
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
      has_domain_sample = has_domain_sample ||
                          ((q.array() >= 0.).all() && (q.array() <= 1.).all());
    }
  }
  const bool uniform = has_domain_sample && sampled_assemblages.size() == 1;
  // Only saved junctions and trace endpoints connect boundaries. Interior
  // chords may cross without creating a thermodynamic vertex or another field.
  Vertices vertices(tolerance);
  std::map<int, Index> nodes;
  for (const auto &node : result.nodes)
    nodes.emplace(node.id, vertices.insert(coordinate(node)));
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
      auto &old = edges[candidate];
      if (old.from != path.front() || old.to != path.back() ||
          old.solution_replacement != replacement)
        continue;
      // Two records of the same reaction, stable endpoints and directed
      // neighbours are one branch even when their coarse chords differ.
      bool same_branch =
          source && old.source && source->zero_phase >= 0 &&
          source->zero_phase == old.source->zero_phase &&
          canonical(source->assemblage) == canonical(old.source->assemblage) &&
          old.labels == std::set<Phases>{left} &&
          edges[candidate ^ 1].labels == std::set<Phases>{right};
      auto follows = [&](const Ring &a, const Ring &b) {
        for (Index vertex : a) {
          double distance = std::numeric_limits<double>::infinity();
          for (Index j = 1; j < b.size(); ++j)
            distance =
                std::min(distance, segment_distance(vertices.points[vertex],
                                                    vertices.points[b[j - 1]],
                                                    vertices.points[b[j]]));
          if (distance > tolerance)
            return false;
        }
        return true;
      };
      if (same_branch || (follows(path, old.path) && follows(old.path, path))) {
        if (!left.empty())
          old.labels.insert(left);
        if (!right.empty())
          edges[candidate ^ 1].labels.insert(right);
        old.phase_boundary = edges[candidate ^ 1].phase_boundary =
            old.phase_boundary || phase_boundary;
        old.frame_boundary = edges[candidate ^ 1].frame_boundary =
            old.frame_boundary || frame;
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
  for (const auto &line : result.boundaries) {
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
        canonical(start->assemblage) == canonical(end->assemblage) &&
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
          Index vertex = nodes.at(node.id);
          if (vertex == path.front() || vertex == path.back())
            continue;
          const auto event = canonical(node.assemblage);
          const auto assemblage = canonical(line.assemblage);
          if (!std::includes(event.begin(), event.end(), assemblage.begin(),
                             assemblage.end()) ||
              (line.zero_phase >= 0 &&
               std::find(node.zero_phases.begin(), node.zero_phases.end(),
                         line.zero_phase) == node.zero_phases.end()))
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
      link_junction(list, edges, frame);
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
              walk.dangling || (outgoing[vertex].size() == 1 && !frame);
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
    for (std::size_t j = 1; j < ring.size(); ++j) {
      action(edge_index.at({ring[j - 1], ring[j]}), ring[j - 1], ring[j]);
    }
  };
  std::map<Phases, int> field_ids;
  for (auto &field : result.fields)
    field_ids.emplace(canonical(field.phases), field.id);
  FieldPolygons output;
  for (auto &face : faces) {
    FieldPolygon polygon;
    polygon.source_regions.push_back(
        burnman::utils::checked_int(output.polygons.size()));
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
    Assemblages exterior_labels, perimeter, common;
    bool constrained = false;
    std::set<const Boundary *> sources;
    bool physical_perimeter = false, unknown_side = false;
    auto evidence = [&](const Ring &ring, bool exterior) {
      ring_edges(ring, [&](Index edge, Index from, Index to) {
        physical_perimeter = physical_perimeter || edges[edge].phase_boundary;
        unknown_side = unknown_side || (edges[edge].phase_boundary &&
                                        edges[edge].labels.empty());
        (void)from;
        (void)to;
        if (edges[edge].source && sources.insert(edges[edge].source).second)
          intersect_assemblages(
              common, boundary_assemblages(*edges[edge].source, result),
              constrained);
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
    for (const auto *source : sources)
      if (source->id >= 0)
        polygon.boundary_ids.push_back(source->id);
    std::sort(polygon.boundary_ids.begin(), polygon.boundary_ids.end());
    bool loop_identified = false;
    if (exterior_labels.size() == 1)
      polygon.phases = *exterior_labels.begin();
    else if (exterior_labels.empty() && perimeter.size() == 1)
      polygon.phases = *perimeter.begin();
    else if (!face.dangling && constrained && common.size() == 1) {
      polygon.phases = *common.begin();
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
          "Region " + std::to_string(output.polygons.size()) +
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
    auto assign = [&](Index edge, Index, Index) { edge_faces[edge] = i; };
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
