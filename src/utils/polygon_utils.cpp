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

#include "burnman/utils/polygon_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace burnman {
namespace utils {

using Point = Eigen::Vector2d;

double segment_distance(const Point &p, const Point &a, const Point &b) {
  const Point ab = b - a;
  const double length_squared = ab.squaredNorm();
  if (length_squared == 0.)
    return (p - a).norm();
  const double t = std::clamp((p - a).dot(ab) / length_squared, 0., 1.);
  return (p - (a + t * ab)).norm();
}

double signed_distance_to_polygon(const Point &p,
                                  const std::vector<Eigen::MatrixXd> &rings) {
  bool inside = false;
  double nearest = std::numeric_limits<double>::infinity();
  for (const auto &ring : rings) {
    const Eigen::Index n = ring.rows();
    if (n < 2)
      continue;
    Point b = ring.row(n - 1).transpose();
    for (Eigen::Index i = 0; i < n; ++i) {
      const Point a = ring.row(i).transpose();
      if ((a.y() > p.y()) != (b.y() > p.y()) &&
          p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x())
        inside = !inside;
      nearest = std::min(nearest, segment_distance(p, a, b));
      b = a;
    }
  }
  return inside ? nearest : -nearest;
}

VisualCenter visual_center_of_polygon(const std::vector<Eigen::MatrixXd> &rings,
                                      double precision,
                                      const std::vector<Point> &seeds) {
  if (rings.empty() || rings[0].cols() != 2 || rings[0].rows() < 3)
    throw std::invalid_argument(
        "visual_center_of_polygon needs an exterior ring of at least three "
        "points in an N x 2 matrix.");
  for (const auto &ring : rings)
    if (ring.cols() != 2 || !ring.allFinite())
      throw std::invalid_argument(
          "Polygon rings must be finite N x 2 matrices.");
  const Point low = rings[0].colwise().minCoeff().transpose();
  const Point high = rings[0].colwise().maxCoeff().transpose();
  const Point size = high - low;
  if (precision <= 0.)
    precision = std::max(1.e-7, size.minCoeff() * .002);

  struct Cell {
    Point centre;
    double half, d, maximum;
    bool operator<(const Cell &other) const { return maximum < other.maximum; }
  };
  auto cell = [&](const Point &p, double half) {
    const double d = signed_distance_to_polygon(p, rings);
    return Cell{p, half, d, d + half * std::sqrt(2.)};
  };
  auto best = cell(rings[0].row(0).transpose(), 0.);
  auto consider = [&](const Point &p) {
    auto candidate = cell(p, 0.);
    if (candidate.d > best.d)
      best = candidate;
  };

  // Area centroid of the exterior: a good start for convex shapes.
  Point centroid = Point::Zero();
  double weight = 0.;
  const Eigen::Index n = rings[0].rows();
  for (Eigen::Index i = 0; i < n; ++i) {
    const Point a = rings[0].row(i).transpose();
    const Point b = rings[0].row((i + 1) % n).transpose();
    const double w = a.x() * b.y() - b.x() * a.y();
    centroid += (a + b) * w;
    weight += w;
  }
  if (std::abs(weight) > 1.e-20)
    consider(centroid / (3. * weight));
  // Thin slivers lose centroid accuracy to cancellation; triangle centres of
  // consecutive vertices give stable interior candidates.
  for (Eigen::Index i = 0; i < n; ++i)
    consider((rings[0].row(i) + rings[0].row((i + 1) % n) +
              rings[0].row((i + 2) % n))
                 .transpose() /
             3.);
  for (const auto &seed : seeds)
    consider(seed);

  const auto root = cell(.5 * (low + high), .5 * size.maxCoeff());
  if (root.d > best.d)
    best = root;
  std::priority_queue<Cell> queue;
  queue.push(root);
  for (int iteration = 0; !queue.empty() && iteration < 20000; ++iteration) {
    const auto current = queue.top();
    queue.pop();
    if (current.d > best.d)
      best = current;
    if (current.maximum - best.d <= precision)
      continue;
    const double h = current.half * .5;
    for (double x : {-h, h})
      for (double y : {-h, h})
        queue.push(cell(current.centre + Point(x, y), h));
  }
  return {best.centre, std::max(0., best.d)};
}

} // namespace utils
} // namespace burnman
