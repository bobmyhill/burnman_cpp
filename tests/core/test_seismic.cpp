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

#include "burnman/core/seismic.hpp"
#include "burnman/tools/seismic.hpp"
#include "burnman/utils/constants.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

using Catch::Approx;
using namespace burnman::seismic;
using Array = Eigen::ArrayXd;

TEST_CASE("Seismic reference models match BurnMan at 2000 km", "[seismic]") {
  const PREM prem;
  const Slow slow;
  const Fast fast;
  const STW105 stw;
  const AK135 ak;
  const IASP91 iasp;
  const std::vector<const Seismic1DModel *> models = {&prem, &slow, &fast,
                                                      &stw,  &ak,   &iasp};
  const double expected[][3] = {
      {12817.6924, 6932.8549, 5120.6291},
      {12795.360742611414, 6904.329188001315, 5120.6291},
      {12795.360742611414, 6941.422556820191, 5120.6291},
      {12817.85012987013, 6932.934206525182, 5120.674925562243},
      {12798.468686868688, 6920.521212121212, 5103.89696969697},
      {12794.4, 6921.0, 0.0}};
  for (std::size_t i = 0; i < models.size(); ++i) {
    const auto &model = *models[i];
    const auto values =
        model.evaluate({"v_p", "v_s", "v_phi"}, Array::Constant(1, 2.0e6));
    CHECK(values.rows() == 3);
    CHECK(values.cols() == 1);
    CHECK(values(0, 0) == Approx(expected[i][0]).epsilon(1.0e-12));
    CHECK(values(1, 0) == Approx(expected[i][1]).epsilon(1.0e-12));
    if (i + 1 < models.size())
      CHECK(model.density(2.0e6) == Approx(expected[i][2]).epsilon(1.0e-12));
    CHECK(model.v_phi(2.0e6) ==
          Approx(std::sqrt(expected[i][0] * expected[i][0] -
                           4.0 / 3.0 * expected[i][1] * expected[i][1])));
  }
  CHECK(prem.internal_depth_list().size() == 94);
  CHECK(stw.internal_depth_list().size() == 750);
  CHECK(ak.internal_depth_list().size() == 145);
  CHECK(iasp.internal_depth_list().size() == 152);
  CHECK_THROWS_AS(iasp.density(1.0e6), std::invalid_argument);
}

TEST_CASE("Seismic interpolation resolves discontinuities and inverse pressure",
          "[seismic]") {
  SeismicTable table;
  table.table_depth = (Array(4) << 0.0, 10.0, 10.0, 20.0).finished();
  table.table_vp = (Array(4) << 1.0, 2.0, 4.0, 6.0).finished();
  table.table_pressure = (Array(4) << 0.0, 1.0, 1.1, 2.0).finished();
  CHECK(table.v_p(-5.0) == 1.0);
  CHECK(table.v_p(5.0) == 1.5);
  CHECK(table.v_p(10.0) == 4.0);
  CHECK(table.v_p(15.0) == 5.0);
  CHECK(table.v_p(25.0) == 6.0);
  CHECK(table.depth(1.05) == 10.0);
  CHECK(table.radius(1.05) == table.earth_radius - 10.0);
  CHECK_THROWS_AS(table.depth(3.0), std::invalid_argument);
  CHECK_THROWS_AS(table.evaluate({"unknown"}), std::invalid_argument);
  const auto shifted = table.internal_depth_list();
  CHECK(shifted[1] == 9.0);
  CHECK(shifted[2] == 11.0);
  CHECK(table.internal_depth_list(11.0, 19.0).size() == 0);
  table.table_vs = Array::Ones(3);
  CHECK_THROWS_AS(table.v_s(0.0), std::invalid_argument);
}

TEST_CASE("Gravity and pressure obey the spherical and hydrostatic integrals",
          "[seismic]") {
  SeismicTable sphere;
  sphere.earth_radius = 1.0e6;
  sphere.table_depth = Array::LinSpaced(2001, 0.0, sphere.earth_radius);
  sphere.table_radius = sphere.earth_radius - sphere.table_depth;
  sphere.table_density = Array::Constant(2001, 4000.0);
  sphere._compute_gravity();
  sphere._compute_pressure();
  const double coefficient = 4.0 / 3.0 * std::acos(-1.0) *
                             burnman::constants::physics::gravitation * 4000.0;
  CHECK(sphere.gravity(0.0) ==
        Approx(coefficient * sphere.earth_radius).epsilon(2.0e-7));
  CHECK(sphere.gravity(sphere.earth_radius) == 0.0);
  CHECK(sphere.pressure(0.0) == 0.0);
  CHECK(sphere.pressure(sphere.earth_radius) ==
        Approx(0.5 * 4000.0 * coefficient * sphere.earth_radius *
               sphere.earth_radius)
            .epsilon(3.0e-6));
  CHECK(sphere.depth(sphere.pressure(500000.0)) == Approx(500000.0));
}

TEST_CASE(
    "Bullen diagnostic uses second-order gradients and handles zero gravity",
    "[seismic]") {
  SeismicTable table;
  table.table_depth = (Array(5) << 0.0, 1.0, 3.0, 6.0, 10.0).finished();
  table.table_pressure = 10.0 * table.table_depth;
  table.table_density = Array::Constant(5, 2.0);
  table.table_vp = (20.0 + 3.0 * table.table_depth.square()).sqrt();
  table.table_vs = Array::Zero(5);
  table.table_gravity = Array::Constant(5, 10.0);
  const auto result = table.bullen(table.table_depth);
  for (Eigen::Index i = 0; i < result.size(); ++i)
    CHECK(result[i] == Approx(0.6 * table.table_depth[i]).margin(1.0e-12));
  table.table_gravity[4] = 0.0;
  CHECK(std::isnan(table.bullen(table.table_depth)[4]));
  CHECK_THROWS_AS(table.bullen(Array::Zero(3)), std::invalid_argument);
}

TEST_CASE("Attenuation correction follows Matas and handles a fluid",
          "[seismic]") {
  using burnman::tools::seismic::attenuation_correction;
  const auto corrected =
      attenuation_correction(12000.0, 7000.0, 9000.0, 200.0, 1000.0);
  const double cot = 1.0 / std::tan(0.3 * std::acos(-1.0) / 2.0);
  const double qp = 0.75 * std::pow(12000.0 / 7000.0, 2) * 200.0;
  CHECK(corrected[0] == Approx(12000.0 * (1.0 - 0.5 * cot / qp)));
  CHECK(corrected[1] == Approx(7000.0 * (1.0 - 0.5 * cot / 200.0)));
  CHECK(corrected[2] == Approx(9000.0 * (1.0 - 0.5 * cot / 1000.0)));
  const auto fluid =
      attenuation_correction(10000.0, 0.0, 10000.0, 100.0, 1000.0);
  CHECK(fluid[0] == 10000.0);
  CHECK(fluid[1] == 0.0);
}
