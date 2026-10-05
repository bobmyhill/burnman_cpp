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

#include "burnman/tools/equilibration/equilibrate_lambda_bounds.hpp"
#include "solution_fixtures.hpp"
#include "tolerances.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace burnman;

TEST_CASE("equilibrate_lambda_bounds", "[tools][equilibration]") {
  Eigen::VectorXd x(6), dx = Eigen::VectorXd::Zero(6);
  x << 1.0e9, 2000.0, 0.4, 0.2, 0.6, 0.8;
  double expected = 1.0;
  SECTION("Zero step") {}
  SECTION("Pressure step") {
    dx(0) = 40.0e9;
    expected = 0.5;
  }
  SECTION("Temperature step") {
    dx(1) = -1000.0;
    expected = 0.5;
  }
  SECTION("Depletion of a phase") {
    dx(2) = -0.8;
    expected = 0.999 * 0.4 / 0.8;
  }
  SECTION("Endmember fraction step") {
    dx(3) = -0.4;
    expected = 0.99 * 0.2 / 0.4;
  }
  SECTION("Trace endmember minimum step") {
    x(3) = 0.0;
    dx(3) = 0.1;
    expected = 0.1;
  }
  SECTION("Growth of a phase") { dx(2) = 10.0; }
  SECTION("Smallest simultaneous bound") {
    dx(0) = 40.0e9;
    dx(1) = 2000.0;
    expected = 0.25;
  }
  auto bounds = equilibration::lambda_bounds_func(dx, x, {2, 2});
  CHECK(bounds.first == 1.0e-8);
  CHECK_THAT(bounds.second, WithinRel(expected, tol_rel));
}
