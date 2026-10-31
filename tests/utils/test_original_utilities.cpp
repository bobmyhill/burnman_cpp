// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of upstream tests; reference: tests/reference/burnman.json.

#include "burnman/tools/averaging/averaging_utils.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace burnman;
using Catch::Approx;

TEST_CASE("Original VRH utility tests", "[original][averaging]") {
  SECTION("VRH_average.test_one_object") {
    CHECK(averaging::utils::voigt_reuss_hill_fn(
              Eigen::ArrayXd::Constant(1, 2.),
              Eigen::ArrayXd::Constant(1, .123)) ==
          Approx(.123).epsilon(1.e-5));
  }
  SECTION("VRH_average.test_two_same") {
    CHECK(averaging::utils::voigt_reuss_hill_fn(Eigen::Array2d(2., 2.),
                                                Eigen::Array2d(.456, .456)) ==
          Approx(.456).epsilon(1.e-5));
  }
  SECTION("VRH_average.test_one_no_volume") {
    CHECK(averaging::utils::voigt_reuss_hill_fn(Eigen::Array2d(0., 2.),
                                                Eigen::Array2d(.123, .456)) ==
          Approx(.456).epsilon(1.e-5));
  }
  SECTION("VRH_average.test_mix") {
    CHECK(averaging::utils::voigt_reuss_hill_fn(Eigen::Array2d(1., 2.),
                                                Eigen::Array2d(.1, .2)) ==
          Approx(.15833333333333).epsilon(1.e-5));
  }
}
