// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/macaw.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/murnaghan.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
types::MineralParams macaw_params() {
  types::MineralParams p;
  p.V_0 = 11.24e-6;
  p.K_0 = 161.e9;
  p.Kprime_0 = 4.21796;
  p.Kprime_inf = 3.;
  return p;
}
} // namespace

TEST_CASE("MACAW factory and reference state", "[macaw][eos]") {
  auto model = eos::make_eos(types::EOSType::MACAW);
  REQUIRE(dynamic_cast<eos::MACAW *>(model.get()));
  auto p = macaw_params();
  model->validate_parameters(p);
  CHECK(*p.F_0 == 0.);
  CHECK(*p.P_0 == 1.e5);
  CHECK_FALSE(p.E_0);
  p.P_0 = 2.5e9;
  p.F_0 = -125000.;
  const double t = GENERATE(0., 300., 2000.), pressure = *p.P_0, v = *p.V_0;
  CHECK(model->compute_pressure(t, v, p) == pressure);
  CHECK(model->compute_volume(pressure, t, p) == v);
  CHECK_THAT(model->compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(*p.K_0, 1.e-15));
  CHECK_THAT(model->compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(*p.K_0, 1.e-15));
  CHECK(model->compute_helmholtz_free_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_molar_internal_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_gibbs_free_energy(pressure, t, v, p) ==
        *p.F_0 + pressure * v);
  CHECK(model->compute_enthalpy(pressure, t, v, p) == *p.F_0 + pressure * v);
  CHECK(model->compute_entropy(pressure, t, v, p) == 0.);
  CHECK(model->compute_thermal_expansivity(pressure, t, v, p) == 0.);
  CHECK(model->compute_grueneisen_parameter(pressure, t, v, p) == 0.);
  CHECK(model->compute_shear_modulus(pressure, t, v, p) == 1.e99);
  CHECK(model->compute_molar_heat_capacity_p(pressure, t, v, p) == 1.e-99);
  CHECK(model->compute_molar_heat_capacity_v(pressure, t, v, p) == 1.e-99);
}

TEST_CASE("MACAW independent reference values", "[macaw][eos]") {
  // 60-digit evaluation of the parameter cubic and MACAW expressions,
  // independently of Python's Cardano expression and volume inversion.
  struct Reference {
    double ratio, pressure, bulk, energy;
  };
  const auto row = GENERATE(
      Reference{.4, 1142011807654.1572, 3903229829507.1904, 1646885.2546638586},
      Reference{.8, 58089267503.034828, 387494292131.85278, 53508.360730554698},
      Reference{.99, 1652807997.7735183, 167946788810.27551,
                92.078276054214797},
      Reference{1., 0., 161000000000., 0.},
      Reference{1.2, -20180941186.792255, 70505174438.965683,
                26436.074242230563});
  eos::MACAW model;
  auto p = macaw_params();
  p.P_0 = 0.;
  model.validate_parameters(p);
  const double v = row.ratio * *p.V_0;
  CAPTURE(row.ratio);
  CHECK_THAT(model.compute_pressure(300., v, p),
             WithinRel(row.pressure, 1.e-12) || WithinAbs(0., 1.e-4));
  CHECK_THAT(model.compute_volume(row.pressure, 300., p), WithinRel(v, 1.e-13));
  CHECK_THAT(
      model.compute_isothermal_bulk_modulus_reuss(row.pressure, 300., v, p),
      WithinRel(row.bulk, 1.e-12));
  CHECK_THAT(model.compute_helmholtz_free_energy(row.pressure, 300., v, p),
             WithinRel(row.energy, 1.e-12) || WithinAbs(0., 1.e-8));
}

TEST_CASE("MACAW thermodynamic derivatives and reference Kprime",
          "[macaw][eos]") {
  eos::MACAW model;
  auto p = macaw_params();
  struct Slopes {
    double reference, infinite;
  };
  const auto slopes = GENERATE(Slopes{4.21796, 3.}, Slopes{6., 3.},
                               Slopes{2., 4.}, Slopes{2., 2.});
  p.Kprime_0 = slopes.reference;
  p.Kprime_inf = slopes.infinite;
  p.P_0 = GENERATE(0., 2.5e9);
  p.F_0 = -125000.;
  model.validate_parameters(p);
  const double ratio = GENERATE(.65, .95, 1.2), v = ratio * *p.V_0,
               dv = v * 1.e-5;
  const double pressure = model.compute_pressure(300., v, p);
  CAPTURE(slopes.reference, slopes.infinite, *p.P_0, ratio);
  const auto f = [&](double volume) {
    return model.compute_helmholtz_free_energy(pressure, 300., volume, p);
  };
  CHECK_THAT(model.compute_volume(pressure, 300., p), WithinRel(v, 1.e-13));
  CHECK_THAT(-(f(v + dv) - f(v - dv)) / (2. * dv), WithinRel(pressure, 2.e-8));
  const double dpdv = (model.compute_pressure(300., v + dv, p) -
                       model.compute_pressure(300., v - dv, p)) /
                      (2. * dv);
  CHECK_THAT(-v * dpdv, WithinRel(model.compute_isothermal_bulk_modulus_reuss(
                                      pressure, 300., v, p),
                                  2.e-8));
  const auto g = [&](double at_pressure) {
    return model.compute_gibbs_free_energy(
        at_pressure, 300., model.compute_volume(at_pressure, 300., p), p);
  };
  CHECK_THAT((g(pressure + 1.e5) - g(pressure - 1.e5)) / 2.e5,
             WithinRel(v, 2.e-8));
  const auto k = [&](double at_pressure) {
    return model.compute_isothermal_bulk_modulus_reuss(
        at_pressure, 300., model.compute_volume(at_pressure, 300., p), p);
  };
  CHECK_THAT((k(*p.P_0 + 1.e5) - k(*p.P_0 - 1.e5)) / 2.e5,
             WithinRel(*p.Kprime_0, 2.e-8));
}

TEST_CASE("MACAW constant-Kprime limit and small strain energy",
          "[macaw][eos]") {
  eos::MACAW model;
  eos::Murnaghan reference;
  auto p = macaw_params();
  p.Kprime_inf = p.Kprime_0 = 3.;
  p.F_0 = -125000.;
  p.P_0 = 2.5e9;
  model.validate_parameters(p);
  auto rp = p;
  reference.validate_parameters(rp);
  for (double ratio : {.2, .8, 1.2, 2.}) {
    const double v = ratio * *p.V_0,
                 pressure = reference.compute_pressure(300., v, rp);
    CHECK_THAT(model.compute_pressure(300., v, p), WithinRel(pressure, 1.e-13));
    CHECK_THAT(model.compute_volume(pressure, 300., p), WithinRel(v, 1.e-13));
    CHECK_THAT(
        model.compute_isothermal_bulk_modulus_reuss(pressure, 300., v, p),
        WithinRel(reference.compute_isothermal_bulk_modulus_reuss(pressure,
                                                                  300., v, rp),
                  1.e-13));
    CHECK_THAT(model.compute_helmholtz_free_energy(pressure, 300., v, p),
               WithinRel(reference.compute_helmholtz_free_energy(pressure, 300.,
                                                                 v, rp),
                         1.e-12));
  }
  p.F_0 = p.P_0 = 0.;
  p.Kprime_0 = GENERATE(3., 4.21796);
  model.validate_parameters(p);
  const double v = *p.V_0 * GENERATE(1. - 1.e-8, 1. + 1.e-8), dv = v - *p.V_0;
  CHECK_THAT(model.compute_helmholtz_free_energy(0., 300., v, p),
             WithinRel(.5 * *p.K_0 / *p.V_0 * dv * dv, 1.e-7));
}

TEST_CASE("MACAW inversion range, tension bound and asymptote",
          "[macaw][eos]") {
  eos::MACAW model;
  auto p = macaw_params();
  model.validate_parameters(p);
  const double minimum = model.compute_pressure(300., 1.e6 * *p.V_0, p);
  CHECK_THROWS_AS(model.compute_volume(minimum - 1., 300., p),
                  std::invalid_argument);
  for (double ratio : {.05, 2.}) {
    const double v = ratio * *p.V_0;
    CHECK_THAT(
        model.compute_volume(model.compute_pressure(300., v, p), 300., p),
        WithinRel(v, 1.e-13));
  }
  const double v = model.compute_volume(1.e30, 300., p);
  CHECK_THAT(model.compute_pressure(300., v, p), WithinRel(1.e30, 1.e-12));
  CHECK_THAT(model.compute_isothermal_bulk_modulus_reuss(1.e30, 300., v, p) /
                 1.e30,
             WithinRel(*p.Kprime_inf, 1.e-8));
}

TEST_CASE("MACAW parameter and state validation", "[macaw][eos]") {
  eos::MACAW model;
  auto p = macaw_params();
  SECTION("Required parameters") {
    const auto field = GENERATE(
        &types::MineralParams::V_0, &types::MineralParams::K_0,
        &types::MineralParams::Kprime_0, &types::MineralParams::Kprime_inf);
    (p.*field).reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Helmholtz reference") {
    p.E_0 = 0.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid shape") {
    p.Kprime_inf = GENERATE(0., 1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid states") {
    model.validate_parameters(p);
    const double v =
        GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.compute_pressure(300., v, p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_helmholtz_free_energy(0., 300., v, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        model.compute_volume(std::numeric_limits<double>::quiet_NaN(), 300., p),
        std::invalid_argument);
  }
}
