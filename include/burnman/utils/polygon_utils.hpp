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

#ifndef BURNMAN_UTILS_POLYGON_UTILS_INCLUDED
#define BURNMAN_UTILS_POLYGON_UTILS_INCLUDED

#include <Eigen/Dense>
#include <vector>

namespace burnman {
namespace utils {

/**
 * @brief Result of `visual_center_of_polygon`.
 */
struct VisualCenter {
  Eigen::Vector2d point = Eigen::Vector2d::Zero();
  /// Distance from `point` to the nearest ring; zero if no interior was found.
  double distance = 0.;
};

/**
 * @brief Distance from a point to the closed segment ab.
 */
double segment_distance(const Eigen::Vector2d &p, const Eigen::Vector2d &a,
                        const Eigen::Vector2d &b);

/**
 * @brief Signed distance from a point to the nearest ring edge.
 *
 * Positive inside the polygon (even-odd rule over all rings, so holes are
 * excluded), negative outside.
 *
 * @param rings Rings as N x 2 matrices. The first is the exterior and the
 *        rest are holes. Rings may be open or repeat their first point.
 */
double signed_distance_to_polygon(const Eigen::Vector2d &p,
                                  const std::vector<Eigen::MatrixXd> &rings);

/**
 * @brief Pole of inaccessibility: the interior point farthest from every edge.
 *
 * Unlike a centroid, this point always lies inside concave polygons and
 * outside holes, so it suits label placement and choosing a representative
 * sample point. It subdivides square cells best-first until no cell can
 * improve the best distance by more than `precision`.
 *
 * Mirrors `burnman.utils.plotting.visual_center_of_polygon`.
 *
 * @param rings Exterior ring followed by any holes, as N x 2 matrices.
 * @param precision Absolute distance tolerance. If not positive, a default of
 *        0.2% of the shorter bounding-box side (at least 1e-7) is used.
 * @param seeds Optional extra candidate points, for example an existing sample.
 * @return The centre and its distance to the nearest edge.
 */
VisualCenter
visual_center_of_polygon(const std::vector<Eigen::MatrixXd> &rings,
                         double precision = 0.,
                         const std::vector<Eigen::Vector2d> &seeds = {});

} // namespace utils
} // namespace burnman

#endif // BURNMAN_UTILS_POLYGON_UTILS_INCLUDED
