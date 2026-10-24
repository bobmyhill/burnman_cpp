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

#include "burnman/optim/composition_fitting.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using burnman::optim::fit_phase_proportions_to_bulk_composition;
using Catch::Approx;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

TEST_CASE(
    "Phase amounts recover the bulk inventory without a unit-sum constraint",
    "[composition_fitting]") {
  Matrix phases(3, 2);
  phases << 1., 0., 0., 1., 1., 1.;
  Vector bulk(3);
  bulk << 1., 3., 4.;
  const auto [amounts, covariance, residual] =
      fit_phase_proportions_to_bulk_composition(phases, bulk);
  CHECK(amounts[0] == Approx(1.));
  CHECK(amounts[1] == Approx(3.));
  CHECK((phases * amounts - bulk).norm() < 1.e-13);
  CHECK(covariance(0, 0) == Approx(2. / 3.));
  CHECK(covariance(0, 1) == Approx(-1. / 3.));
  CHECK(covariance(1, 1) == Approx(2. / 3.));
  CHECK(residual == Approx(0.).margin(1.e-25));
}

TEST_CASE("Phase fitting respects nonnegativity when no exact mixture exists",
          "[composition_fitting]") {
  Matrix phases(3, 2);
  phases << 1., 0., 0., 1., 1., 1.;
  Vector bulk(3);
  bulk << -1., 2., 0.;
  const auto [amounts, covariance, residual] =
      fit_phase_proportions_to_bulk_composition(phases, bulk);
  CHECK(amounts[0] == Approx(0.).margin(1.e-10));
  CHECK(amounts[1] == Approx(1.));
  CHECK(residual == Approx(3.));
  CHECK(covariance(0, 0) == Approx(2. / 3.));
  CHECK_THROWS_AS(
      fit_phase_proportions_to_bulk_composition(phases, Vector::Ones(2)),
      std::invalid_argument);
  phases.col(1) = phases.col(0);
  CHECK_THROWS_AS(fit_phase_proportions_to_bulk_composition(phases, bulk),
                  std::invalid_argument);
}
