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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman;
using Catch::Approx;

namespace {
Eigen::MatrixXd rectangle(double x0, double y0, double x1, double y1) {
  Eigen::MatrixXd ring(4, 2);
  ring << x0, y0, x1, y0, x1, y1, x0, y1;
  return ring;
}
} // namespace

TEST_CASE("visual_center_of_polygon (square)", "[utils][polygon_utils]") {
  const auto centre = utils::visual_center_of_polygon({rectangle(0, 0, 2, 2)});
  CHECK(centre.point.x() == Approx(1.).margin(1.e-2));
  CHECK(centre.point.y() == Approx(1.).margin(1.e-2));
  CHECK(centre.distance == Approx(1.).margin(1.e-2));
}

TEST_CASE("visual_center_of_polygon (L shape stays inside)",
          "[utils][polygon_utils]") {
  Eigen::MatrixXd ring(6, 2);
  ring << 0, 0, 10, 0, 10, 1, 1, 1, 1, 10, 0, 10;
  const auto centre = utils::visual_center_of_polygon({ring}, 1.e-6);
  CHECK(utils::signed_distance_to_polygon(centre.point, {ring}) > 0.);
  // The elbow fits a larger circle than the arms: 2 - sqrt2.
  CHECK(centre.distance == Approx(2. - std::sqrt(2.)).margin(1.e-4));
}

TEST_CASE("visual_center_of_polygon (hole is avoided)",
          "[utils][polygon_utils]") {
  const std::vector<Eigen::MatrixXd> rings{rectangle(0, 0, 10, 10),
                                           rectangle(4, 4, 6, 6)};
  const auto centre = utils::visual_center_of_polygon(rings, 1.e-6);
  CHECK(utils::signed_distance_to_polygon(centre.point, rings) > 0.);
  // The centroid lies in the hole; the best point sits in a corner region,
  // equidistant from the outer corner edges and the hole corner: 4
  // sqrt2/(1+sqrt2).
  CHECK(centre.distance ==
        Approx(4. * std::sqrt(2.) / (1. + std::sqrt(2.))).margin(1.e-3));
}

TEST_CASE("visual_center_of_polygon (seed and input checks)",
          "[utils][polygon_utils]") {
  const auto ring = rectangle(0, 0, 1, 1);
  const auto centre =
      utils::visual_center_of_polygon({ring}, 0., {Eigen::Vector2d(.5, .5)});
  CHECK(centre.distance >= .49);
  CHECK_THROWS_AS(utils::visual_center_of_polygon({}), std::invalid_argument);
  CHECK(utils::signed_distance_to_polygon(Eigen::Vector2d(2, .5), {ring}) ==
        Approx(-1.));
}
