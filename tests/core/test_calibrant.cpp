// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "burnman/calibrants/catalogue.hpp"
#include "burnman/calibrants/tools.hpp"
#include "burnman/core/calibrant.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using burnman::Calibrant;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
Calibrant linear(bool pressure_function = true) {
  Calibrant::Params p{{"V_0", 1.e-5}, {"k", 2.e16}, {"alpha", 2.e6}};
  if (pressure_function)
    return Calibrant(
        [](double v, double t, const Calibrant::Params &a) {
          return a.at("k") * (a.at("V_0") - v) + a.at("alpha") * (t - 300.);
        },
        "pressure", p);
  return Calibrant(
      [](double pressure, double t, const Calibrant::Params &a) {
        return a.at("V_0") -
               (pressure - a.at("alpha") * (t - 300.)) / a.at("k");
      },
      "volume", p);
}
} // namespace

TEST_CASE("Calibrants invert pressure and volume functions", "[calibrant]") {
  for (bool pressure_function : {false, true}) {
    auto cal = linear(pressure_function);
    for (double p : {-1.e9, 0., 10.e9, 100.e9, 180.e9}) {
      const double v = cal.volume(p, 1000.);
      CHECK_THAT(v, WithinRel(1.e-5 - (p - 1.4e9) / 2.e16, 1.e-12));
      CHECK_THAT(cal.pressure(v, 1000.), WithinAbs(p, .001));
    }
  }
  CHECK_THROWS_AS(Calibrant({}, "invalid", {}), std::invalid_argument);
  auto cal = linear();
  CHECK_THROWS_AS(cal.pressure(0., 1000.), std::invalid_argument);
  CHECK_THROWS_AS(cal.pressure(1.e-5, -1.), std::invalid_argument);
  CHECK_THROWS_AS(cal.volume(std::nan(""), 1000.), std::invalid_argument);
  cal.params.erase("V_0");
  CHECK_THROWS_AS(cal.volume(1.e9, 1000.), std::invalid_argument);
  Calibrant impossible(
      [](double, double, const Calibrant::Params &) { return 1.; }, "volume",
      {});
  CHECK_THROWS_AS(impossible.pressure(2., 1000.), std::invalid_argument);
}

TEST_CASE("Calibrant covariance orders and correlations", "[calibrant]") {
  auto cal = linear();
  Eigen::Matrix2d VT;
  VT << 1.e-18, 2.e-9, 2.e-9, 100.;
  Eigen::Matrix<double, 3, 2> JP;
  JP << -2.e16, 2.e6, 1., 0., 0., 1.;
  const auto [p, PVT] = cal.pressure(9.e-6, 1000., VT);
  CHECK_THAT(p, WithinRel(21.4e9, 1.e-12));
  const Eigen::Matrix3d expected_p = JP * VT * JP.transpose();
  for (Eigen::Index i = 0; i < 3; ++i)
    for (Eigen::Index j = 0; j < 3; ++j)
      CHECK_THAT(PVT(i, j), WithinRel(expected_p(i, j), 1.e-7));
  Eigen::Matrix2d PT;
  PT << 1.e16, 2.e8, 2.e8, 100.;
  Eigen::Matrix<double, 3, 2> JV;
  JV << -1. / 2.e16, 1.e-10, 1., 0., 0., 1.;
  const auto [v, VPT] = cal.volume(21.4e9, 1000., PT);
  CHECK_THAT(v, WithinRel(9.e-6, 1.e-12));
  const Eigen::Matrix3d expected_v = JV * PT * JV.transpose();
  for (Eigen::Index i = 0; i < 3; ++i)
    for (Eigen::Index j = 0; j < 3; ++j)
      CHECK_THAT(VPT(i, j), WithinAbs(expected_v(i, j),
                                      1.e-5 * std::sqrt(expected_v(i, i) *
                                                        expected_v(j, j))));
  CHECK(cal.pressure(9.e-6, 1000., Eigen::Matrix2d::Zero()).second.isZero());
  PT(1, 1) = -1.;
  CHECK_THROWS_AS(cal.volume(21.4e9, 1000., PT), std::invalid_argument);
  PT << 1., 2., 2., 1.;
  CHECK_THROWS_AS(cal.volume(21.4e9, 1000., PT), std::invalid_argument);
  PT << 1., 0., 1., 1.;
  CHECK_THROWS_AS(cal.volume(21.4e9, 1000., PT), std::invalid_argument);
}

TEST_CASE("Every published pressure calibrant is native and invertible",
          "[calibrant]") {
  const auto &catalogue = burnman::calibrants::catalogue();
  REQUIRE(catalogue.size() == 52);
  for (const auto &[id, factory] : catalogue) {
    INFO(id);
    auto cal = factory();
    const double v = cal.volume(10.e9, 1000.);
    CHECK(v > 0.);
    CHECK_THAT(cal.pressure(v, 1000.), WithinAbs(10.e9, 20.));
    Eigen::Matrix2d VT;
    VT << std::pow(v * 1.e-4, 2), v * 2.e-4, v * 2.e-4, 100.;
    const auto [p, covariance] = cal.pressure(v, 1000., VT);
    CHECK(std::isfinite(p));
    CHECK(covariance.allFinite());
    CHECK(covariance.diagonal().minCoeff() >= 0.);
  }
}

TEST_CASE("Pressure scale conversion preserves volume and covariance",
          "[calibrant]") {
  using burnman::calibrants::pressure_to_pressure;
  auto old_cal = burnman::calibrants::Anderson_1989::Au();
  auto new_cal = burnman::calibrants::Fei_2007::Au();
  const double p2 = pressure_to_pressure(old_cal, new_cal, 20.e9, 1000.);
  CHECK_THAT(pressure_to_pressure(new_cal, old_cal, p2, 1000.),
             WithinAbs(20.e9, .1));
  Eigen::Matrix2d PT;
  PT << 1.e18, 1.e8, 1.e8, 10.;
  const auto [p, C] = pressure_to_pressure(old_cal, new_cal, 20.e9, 1000., PT);
  const auto [back, recovered] =
      pressure_to_pressure(new_cal, old_cal, p, 1000., C);
  CHECK_THAT(back, WithinAbs(20.e9, .1));
  for (Eigen::Index i = 0; i < 2; ++i)
    for (Eigen::Index j = 0; j < 2; ++j)
      CHECK_THAT(recovered(i, j), WithinRel(PT(i, j), 2.e-5));
}
