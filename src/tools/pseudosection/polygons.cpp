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
using Ring = std::vector<Index>;
using Phases = std::vector<int>;
using detail::cross;
using detail::segment_distance;

struct Segment {
  Point a, b;
  Phases left, right;
  std::vector<double> cuts{0., 1.};
};

struct Edge {
  Index from, to, next = no_index;
  std::set<Phases> labels{};
};

struct Walk {
  std::vector<Ring> rings;
  std::set<Phases> labels;
  bool dangling = false;
};

struct Face {
  Ring exterior;
  std::vector<Ring> holes;
  std::set<Phases> labels;
  double area;
  bool dangling;
};

struct Hole {
  Ring ring;
  std::set<Phases> labels;
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
  Index insert(const Point &p) {
    Index existing = find(p);
    if (existing != no_index)
      return existing;
    long long x = static_cast<long long>(std::floor(p.x() / tolerance)),
              y = static_cast<long long>(std::floor(p.y() / tolerance));
    Index index = points.size();
    points.push_back(p);
    bins[{x, y}].push_back(index);
    return index;
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

void intersect(Segment &first, Segment &second, double tolerance) {
  if ((first.a.cwiseMin(first.b).array() >
       second.a.cwiseMax(second.b).array() + tolerance)
          .any() ||
      (second.a.cwiseMin(second.b).array() >
       first.a.cwiseMax(first.b).array() + tolerance)
          .any())
    return;
  Point a = first.b - first.a, b = second.b - second.a,
        offset = second.a - first.a;
  double denominator = cross(a, b);
  auto split_overlap = [&]() {
    // Collinear duplicate edges and domain-frame overlaps must be split
    // before merging. Otherwise the half-edge graph contains false faces.
    for (auto &p : {second.a, second.b}) {
      double t = (p - first.a).dot(a) / a.squaredNorm();
      if (t >= 0. && t <= 1.)
        first.cuts.push_back(t);
    }
    for (auto &p : {first.a, first.b}) {
      double u = (p - second.a).dot(b) / b.squaredNorm();
      if (u >= 0. && u <= 1.)
        second.cuts.push_back(u);
    }
  };
  // Repeated chords differ by roundoff. Dividing their tiny determinants can
  // invent crossings; use coordinate precision here to retain real thin fields.
  double roundoff =
      std::min(tolerance, 256. * std::numeric_limits<double>::epsilon());
  if (std::abs(cross(a, offset)) <= roundoff * a.norm() &&
      std::abs(cross(a, second.b - first.a)) <= roundoff * a.norm() &&
      std::abs(cross(b, -offset)) <= roundoff * b.norm() &&
      std::abs(cross(b, first.b - second.a)) <= roundoff * b.norm()) {
    split_overlap();
  } else if (std::abs(denominator) > 1.e-14 * a.norm() * b.norm()) {
    double t = cross(offset, b) / denominator,
           u = cross(offset, a) / denominator;
    double t_tol = tolerance / a.norm(), u_tol = tolerance / b.norm();
    if (t >= -t_tol && t <= 1. + t_tol && u >= -u_tol && u <= 1. + u_tol) {
      first.cuts.push_back(std::clamp(t, 0., 1.));
      second.cuts.push_back(std::clamp(u, 0., 1.));
    }
  } else if (std::abs(cross(a, offset)) <= tolerance * a.norm())
    split_overlap();
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

// Branch and bound on signed distance, in the normalised domain. The label
// remains inside concave faces and outside holes; a centroid need not do so.
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

std::vector<std::pair<std::size_t, std::size_t>>
detail::field_conflicts(const Result &result, const FieldPolygons &geometry) {
  const auto ranges = result.coordinate_ranges();
  Point scale(ranges[0][1] - ranges[0][0], ranges[1][1] - ranges[1][0]);
  std::set<Phases> represented;
  for (const auto &polygon : geometry.polygons)
    if (!polygon.has_open_boundary && !polygon.outside_model_domain)
      represented.insert(polygon.phases);
  std::vector<std::pair<std::size_t, std::size_t>> conflicts;
  for (std::size_t i = 0; i < geometry.polygons.size(); ++i) {
    const auto &polygon = geometry.polygons[i];
    if (polygon.sample_index < 0 || polygon.has_open_boundary ||
        polygon.outside_model_domain || polygon.phases.empty())
      continue;
    const Point low = polygon.vertices.colwise().minCoeff().transpose();
    const Point high = polygon.vertices.colwise().maxCoeff().transpose();
    std::set<Phases> checked;
    for (std::size_t j = 0; j < result.samples.size(); ++j) {
      const auto &sample = result.samples[j];
      if (!sample.success || sample.outside_model_domain)
        continue;
      const auto location = diagram_coordinates(sample, result.section.type);
      if (location.allFinite() && ((location.array() < low.array()).any() ||
                                   (location.array() > high.array()).any()))
        continue;
      Phases phases;
      double total = 0.;
      for (const auto &phase : sample.phases)
        total += phase.amount;
      for (const auto &phase : sample.phases)
        if (phase.amount > result.settings.amount_tolerance * total)
          phases.push_back(phase.id);
      std::sort(phases.begin(), phases.end());
      if (phases == polygon.phases || checked.count(phases))
        continue;
      // Close-to-edge probes can lie across a curved line's straight chord.
      // A field absent from the subdivision must still be accounted for,
      // however narrow it is; its boundary will be recovered thermodynamically.
      double band =
          represented.count(phases)
              ? std::max(1.e-8, result.settings.node_tolerance * 2.)
              : std::max(1.e-8, result.settings.amount_tolerance * 2.);
      if (polygon.contains_rectangle(location, scale * band)) {
        conflicts.emplace_back(j, i);
        checked.insert(phases);
      }
    }
  }
  return conflicts;
}

bool FieldPolygon::contains_rectangle(const Point &centre,
                                      const Point &half_size) const {
  if (!centre.allFinite() || !half_size.allFinite() ||
      (half_size.array() <= 0.).any())
    throw std::invalid_argument("Label rectangle must have finite coordinates "
                                "and positive half-sizes.");
  // Work in rectangle units: pressure and temperature can differ by many
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
  std::map<int, Point> nodes;
  for (auto &node : result.nodes)
    nodes.emplace(node.id, coordinate(node));
  std::vector<Segment> segments;
  for (auto &line : result.boundaries) {
    std::vector<Point> points;
    for (auto &point : line.points)
      points.push_back(coordinate(point));
    if (points.size() < 2)
      continue;
    if (nodes.count(line.start_node))
      points.front() = nodes.at(line.start_node);
    if (nodes.count(line.end_node))
      points.back() = nodes.at(line.end_node);
    for (auto &point : points) {
      if ((point.array() < -tolerance * 10).any() ||
          (point.array() > 1. + tolerance * 10).any())
        throw std::invalid_argument(
            "Phase-line coordinates lie outside the calculation domain.");
      point = point.cwiseMax(Point::Zero()).cwiseMin(Point::Ones());
    }
    // Native tracing records side_b to the left in (P,T) coordinates and
    // side_a to the right. Plot axes (T,P) reverse this winding.
    for (std::size_t i = 1; i < points.size(); ++i)
      if ((points[i] - points[i - 1]).norm() > tolerance)
        segments.push_back({points[i - 1], points[i], canonical(line.side_b),
                            canonical(line.side_a)});
  }
  if (close_domain) {
    std::array<Point, 5> corners{Point(0., 0.), Point(1., 0.), Point(1., 1.),
                                 Point(0., 1.), Point(0., 0.)};
    for (Index i = 1; i < corners.size(); ++i)
      segments.push_back({corners[i - 1], corners[i], {}, {}});
  }
  std::vector<std::vector<Point>> excluded;
  for (auto &region : result.excluded_regions) {
    std::vector<Point> ring;
    for (int i = 0; i < region.rows(); ++i)
      ring.push_back(normalise(region(i, 0), region(i, 1)));
    for (std::size_t i = 1; i < ring.size(); ++i)
      if ((ring[i] - ring[i - 1]).norm() > tolerance)
        segments.push_back({ring[i - 1], ring[i], {}, {}});
    excluded.push_back(std::move(ring));
  }
  for (std::size_t i = 0; i < segments.size(); ++i)
    for (std::size_t j = i + 1; j < segments.size(); ++j)
      intersect(segments[i], segments[j], tolerance);

  Vertices vertices(tolerance);
  std::vector<Edge> edges;
  std::map<std::pair<Index, Index>, Index> edge_index;
  for (auto &segment : segments) {
    std::sort(segment.cuts.begin(), segment.cuts.end());
    segment.cuts.erase(
        std::unique(segment.cuts.begin(), segment.cuts.end(),
                    [&](double a, double b) {
                      return std::abs(a - b) * (segment.b - segment.a).norm() <=
                             tolerance;
                    }),
        segment.cuts.end());
    for (std::size_t i = 1; i < segment.cuts.size(); ++i) {
      Index a = vertices.insert(segment.a +
                                segment.cuts[i - 1] * (segment.b - segment.a));
      Index b = vertices.insert(segment.a +
                                segment.cuts[i] * (segment.b - segment.a));
      if (a == b)
        continue;
      auto key = std::minmax(a, b);
      auto it = edge_index.find(key);
      Index index;
      if (it == edge_index.end()) {
        index = edges.size();
        edges.push_back({a, b});
        edges.push_back({b, a});
        edge_index.emplace(key, index);
      } else {
        index = it->second;
        if (edges[index].from != a)
          index ^= 1;
      }
      if (!segment.left.empty())
        edges[index].labels.insert(segment.left);
      if (!segment.right.empty())
        edges[index ^ 1].labels.insert(segment.right);
    }
  }
  std::vector<std::vector<Index>> outgoing(vertices.points.size());
  for (std::size_t i = 0; i < edges.size(); ++i)
    outgoing[edges[i].from].push_back(i);
  std::vector<Index> positions(edges.size());
  for (auto &list : outgoing) {
    std::sort(list.begin(), list.end(), [&](Index a, Index b) {
      Point da = vertices.points[edges[a].to] - vertices.points[edges[a].from];
      Point db = vertices.points[edges[b].to] - vertices.points[edges[b].from];
      return std::atan2(da.y(), da.x()) < std::atan2(db.y(), db.x());
    });
    for (std::size_t i = 0; i < list.size(); ++i)
      positions[list[i]] = i;
  }
  for (std::size_t i = 0; i < edges.size(); ++i) {
    auto &list = outgoing[edges[i].to];
    edges[i].next = list[(positions[i ^ 1] + list.size() - 1) % list.size()];
  }
  auto in_cycle = [&](Index edge) {
    std::vector<bool> seen(outgoing.size(), false);
    std::vector<Index> pending{edges[edge].from};
    seen[pending.front()] = true;
    for (Index i = 0; i < pending.size(); ++i)
      for (Index candidate : outgoing[pending[i]]) {
        if (candidate / 2 == edge / 2)
          continue;
        Index next = edges[candidate].to;
        if (next == edges[edge].to)
          return true;
        if (!seen[next]) {
          seen[next] = true;
          pending.push_back(next);
        }
      }
    return false;
  };
  std::vector<Walk> walks;
  std::vector<bool> visited(edges.size(), false);
  bool finer = false;
  for (std::size_t start = 0; start < edges.size(); ++start)
    if (!visited[start]) {
      Walk walk;
      Ring path;
      std::set<Index> traversed;
      Index current = start;
      do {
        visited[current] = true;
        if (traversed.count(current ^ 1)) {
          walk.dangling = true;
          finer = finer || in_cycle(current);
        }
        traversed.insert(current);
        path.push_back(edges[current].from);
        walk.labels.insert(edges[current].labels.begin(),
                           edges[current].labels.end());
        current = edges[current].next;
      } while (current != start);
      path.push_back(edges[start].from);
      walk.rings = rings(path);
      walks.push_back(std::move(walk));
    }
  // A cycle edge cannot border the same face on both sides in a planar graph.
  // Snapping has crossed thin fields; retry more precisely. True dangling
  // branches have no alternate path and retain their unresolved diagnostics.
  if (finer && tolerance > 1.e-12)
    return field_polygons(result, std::max(1.e-12, tolerance * .1),
                          close_domain, merge_fields);
  std::vector<Face> faces;
  std::vector<Hole> holes;
  const double area_tolerance = tolerance * tolerance * 4.;
  for (auto &walk : walks)
    for (auto &ring : walk.rings) {
      double value = area(ring, vertices.points);
      if (value > area_tolerance)
        faces.push_back({ring, {}, walk.labels, value, walk.dangling});
      else if (value < -area_tolerance)
        holes.push_back({ring, walk.labels, walk.dangling});
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
  for (auto &[ring, labels, dangling] : holes) {
    double hole_area = -area(ring, vertices.points);
    Index parent = no_index;
    for (std::size_t i = 0; i < faces.size(); ++i)
      if (faces[i].area > hole_area + area_tolerance &&
          encloses(faces[i].exterior, ring) &&
          (parent == no_index || faces[i].area < faces[parent].area))
        parent = i;
    if (parent != no_index) {
      faces[parent].holes.push_back(ring);
      faces[parent].labels.insert(labels.begin(), labels.end());
      faces[parent].dangling = faces[parent].dangling || dangling;
    }
  }
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
      Phases phases;
      double total = 0.;
      for (auto &phase : state.phases)
        total += phase.amount;
      for (auto &phase : state.phases)
        if (phase.amount > total * result.settings.amount_tolerance)
          phases.push_back(phase.id);
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
    // Use the verified state furthest from the polygon boundary. Close to a
    // curved trace, straight chords can put an adjacent-field sample inside
    // the polygon. A dangling physical boundary always leaves the face open.
    auto labels =
        sample_phases.empty() ? face.labels : std::set<Phases>{sample_phases};
    std::set<int> counts;
    for (auto &phases : labels)
      counts.insert(burnman::utils::checked_int(phases.size()));
    if (!face.dangling && counts.size() == 1) {
      polygon.n_phases = *counts.begin();
      if (labels.size() == 1) {
        polygon.phases = *labels.begin();
        auto it = field_ids.find(polygon.phases);
        if (it != field_ids.end())
          polygon.field_id = it->second;
      } else {
        output.diagnostics.push_back(
            "Closed region " + std::to_string(output.polygons.size()) +
            " has a known phase count but conflicting assemblage labels.");
      }
    } else {
      output.diagnostics.push_back(
          "Closed region " + std::to_string(output.polygons.size()) +
          (face.dangling    ? " contains an unfinished boundary."
           : counts.empty() ? " has no assemblage label."
                            : " has conflicting boundary phase counts."));
    }
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
  // Assign each directed edge to its face on the left. Hole rings have the
  // reverse winding and belong to the surrounding face. Zero-area dangling
  // excursions intentionally remain unassigned and cannot be dissolved.
  std::vector<Index> edge_faces(edges.size(), no_index);
  auto ring_edges = [&](const Ring &ring, auto action) {
    for (std::size_t j = 1; j < ring.size(); ++j) {
      Index edge = edge_index.at(std::minmax(ring[j - 1], ring[j]));
      if (edges[edge].from != ring[j - 1])
        edge ^= 1;
      action(edge);
    }
  };
  for (std::size_t i = 0; i < faces.size(); ++i) {
    auto assign = [&](Index edge) { edge_faces[edge] = i; };
    ring_edges(faces[i].exterior, assign);
    for (auto &hole : faces[i].holes)
      ring_edges(hole, assign);
  }
  std::vector<Index> parent(faces.size());
  std::iota(parent.begin(), parent.end(), Index{0});
  auto root = [&](Index i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  for (auto [sample, polygon] : detail::field_conflicts(result, output))
    output.diagnostics.push_back(
        "Closed region " + std::to_string(polygon) +
        " contains a conflicting equilibrium assemblage at sample " +
        std::to_string(sample) + "; a phase boundary may be missing.");
  if (merge_fields)
    for (std::size_t i = 0; i < edges.size(); i += 2) {
      Index a = edge_faces[i], b = edge_faces[i ^ 1];
      if (a == no_index || b == no_index || a == b)
        continue;
      auto &first = output.polygons[a];
      auto &second = output.polygons[b];
      // Matching phase counts or text abbreviations alone are insufficient.
      // Use identified, closed assemblages, including solution-instance IDs.
      if (first.has_open_boundary || second.has_open_boundary ||
          first.phases.empty() || first.phases != second.phases)
        continue;
      a = root(a);
      b = root(b);
      parent[std::max(a, b)] = std::min(a, b);
    }
  std::map<Index, std::vector<Index>> groups;
  for (std::size_t i = 0; i < faces.size(); ++i)
    groups[root(i)].push_back(i);
  std::vector<bool> removed(edges.size(), false);
  std::vector<FieldPolygon> polygons;
  for (auto &[group, members] : groups) {
    if (members.size() == 1) {
      polygons.push_back(std::move(output.polygons[members[0]]));
      continue;
    }
    std::vector<bool> boundary(edges.size(), false), seen(edges.size(), false);
    for (std::size_t i = 0; i < edges.size(); ++i)
      boundary[i] =
          edge_faces[i] != no_index && root(edge_faces[i]) == group &&
          (edge_faces[i ^ 1] == no_index || root(edge_faces[i ^ 1]) != group);
    std::vector<Ring> exteriors, interiors;
    bool valid = true;
    for (std::size_t start = 0; start < edges.size() && valid; ++start)
      if (boundary[start] && !seen[start]) {
        Ring path;
        Index current = start;
        do {
          if (seen[current]) {
            valid = false;
            break;
          }
          seen[current] = true;
          path.push_back(edges[current].from);
          // Rotate clockwise from the twin, skipping internal edges. This
          // follows the exterior of the union even at multi-line crossings.
          auto &list = outgoing[edges[current].to];
          Index next = no_index;
          for (std::size_t turn = 1; turn <= list.size(); ++turn) {
            Index candidate =
                list[(positions[current ^ 1] + list.size() - turn) %
                     list.size()];
            if (boundary[candidate]) {
              next = candidate;
              break;
            }
          }
          if (next == no_index) {
            valid = false;
            break;
          }
          current = next;
        } while (current != start);
        if (!valid)
          break;
        path.push_back(edges[start].from);
        for (auto &ring : rings(path)) {
          double value = area(ring, vertices.points);
          if (value > area_tolerance)
            exteriors.push_back(std::move(ring));
          else if (value < -area_tolerance)
            interiors.push_back(std::move(ring));
        }
      }
    // Sharing an edge makes the union's interior connected. Refuse to erase
    // boundaries if numerical topology or area checks contradict that.
    valid = valid && exteriors.size() == 1;
    double expected = 0.;
    for (Index i : members)
      expected += output.polygons[i].area;
    Face combined;
    double combined_area = 0.;
    if (valid) {
      combined = {exteriors[0],
                  interiors,
                  {},
                  area(exteriors[0], vertices.points),
                  false};
      combined_area = combined.area;
      for (auto &hole : interiors) {
        combined_area += area(hole, vertices.points);
        if (!encloses(combined.exterior, hole))
          valid = false;
      }
      if (std::abs(combined_area - expected) > 1.e-12 * std::max(1., expected))
        valid = false;
    }
    if (!valid) {
      output.diagnostics.push_back(
          "Could not merge regions adjoining source region " +
          std::to_string(members[0]) +
          ": inconsistent union topology or area (" +
          std::to_string(exteriors.size()) +
          " exterior rings; area difference " +
          std::to_string(combined_area - expected) + ").");
      for (Index i : members)
        polygons.push_back(std::move(output.polygons[i]));
      continue;
    }
    FieldPolygon polygon = output.polygons[members[0]];
    polygon.source_regions.clear();
    for (Index member : members)
      polygon.source_regions.push_back(utils::checked_int(member));
    polygon.area = combined_area;
    polygon.vertices =
        coordinates(combined.exterior, vertices.points, origin, scale);
    polygon.holes.clear();
    for (auto &hole : combined.holes)
      polygon.holes.push_back(
          coordinates(hole, vertices.points, origin, scale));
    Index seed = members[0];
    for (Index i : members)
      if (output.polygons[i].label_clearance >
          output.polygons[seed].label_clearance)
        seed = i;
    Point location =
        (output.polygons[seed].label_position - origin).cwiseQuotient(scale);
    auto label = label_point(combined, vertices.points, &location);
    polygon.label_position = origin + label.first.cwiseProduct(scale);
    polygon.label_clearance = label.second;
    polygon.sample_index = output.polygons[seed].sample_index;
    polygons.push_back(std::move(polygon));
    for (std::size_t i = 0; i < edges.size(); i += 2)
      if (edge_faces[i] != no_index && edge_faces[i ^ 1] != no_index &&
          root(edge_faces[i]) == group && root(edge_faces[i ^ 1]) == group)
        removed[i] = removed[i ^ 1] = true;
  }
  output.polygons = std::move(polygons);
  std::vector<int> degree(vertices.points.size(), 0);
  for (std::size_t i = 0; i < edges.size(); i += 2)
    if (!removed[i]) {
      bool masked = false;
      Point middle =
          .5 * (vertices.points[edges[i].from] + vertices.points[edges[i].to]);
      for (auto &region : excluded) {
        Ring ring;
        for (std::size_t j = 0; j < region.size(); ++j)
          ring.push_back(j);
        if (contains(ring, region, middle, tolerance) == 1)
          masked = true;
      }
      if (masked)
        continue;
      output.boundary_segments.push_back(coordinates(
          {edges[i].from, edges[i].to}, vertices.points, origin, scale));
      ++degree[edges[i].from];
      ++degree[edges[i].to];
    }
  for (auto &[id, point] : nodes) {
    Index vertex = vertices.find(point);
    if (vertex != no_index && degree[vertex] >= 3)
      output.boundary_nodes.push_back(id);
  }
  return output;
}
} // namespace burnman::pseudosections
