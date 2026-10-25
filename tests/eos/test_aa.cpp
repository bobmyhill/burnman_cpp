// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/aa.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/minerals/other.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE("AA factory and exact reference state", "[aa][eos]") {
  const auto iron = minerals::other::liquid_iron();
  const auto &p = iron->params;
  auto model = eos::make_eos(types::EOSType::AA);
  REQUIRE(dynamic_cast<eos::AA *>(model.get()));
  const double t = *p.T_0, v = *p.V_0, pressure = *p.P_0;
  CHECK(model->compute_pressure(t, v, p) == pressure);
  CHECK_THAT(model->compute_volume(pressure, t, p), WithinRel(v, 2.e-14));
  CHECK(model->compute_entropy(pressure, t, v, p) == *p.S_0);
  CHECK(model->compute_molar_internal_energy(pressure, t, v, p) == *p.E_0);
  CHECK_THAT(model->compute_grueneisen_parameter(pressure, t, v, p),
             WithinRel(*p.grueneisen_0, 2.e-14));
  CHECK_THAT(model->compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(*p.K_0, 2.e-14));
  CHECK(model->compute_shear_modulus(pressure, t, v, p) == 0.);
  iron->set_state(pressure, t);
  CHECK_THAT(iron->get_density(), WithinRel(7019., 2.e-14));
}

TEST_CASE("AA first law and response derivatives", "[aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  eos::AA model;
  const double rho = GENERATE(5000., 7019., 10000., 13000.);
  const double t = GENERATE(1811., 5000., 7000.);
  const double v = *p.molar_mass / rho, dt = .05, dv = v * 1.e-5;
  const double pressure = model.compute_pressure(t, v, p);
  const auto f = [&](double temperature, double volume) {
    return model.compute_helmholtz_free_energy(pressure, temperature, volume,
                                               p);
  };
  const auto s = [&](double temperature, double volume) {
    return model.compute_entropy(pressure, temperature, volume, p);
  };
  const double kt =
      model.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p);
  const double cv = model.compute_molar_heat_capacity_v(pressure, t, v, p);
  const double cp = model.compute_molar_heat_capacity_p(pressure, t, v, p);
  const double alpha = model.compute_thermal_expansivity(pressure, t, v, p);
  const double gamma = model.compute_grueneisen_parameter(pressure, t, v, p);
  CAPTURE(rho, t);
  CHECK_THAT(-(f(t, v + dv) - f(t, v - dv)) / (2. * dv),
             WithinRel(pressure, 5.e-8) || WithinAbs(pressure, 300.));
  CHECK_THAT(-(f(t + dt, v) - f(t - dt, v)) / (2. * dt),
             WithinRel(s(t, v), 2.e-8));
  CHECK_THAT(t * (s(t + dt, v) - s(t - dt, v)) / (2. * dt),
             WithinRel(cv, 2.e-8));
  CHECK_THAT(-v *
                 (model.compute_pressure(t, v + dv, p) -
                  model.compute_pressure(t, v - dv, p)) /
                 (2. * dv),
             WithinRel(kt, 2.e-8));
  CHECK_THAT((s(t, v + dv) - s(t, v - dv)) / (2. * dv),
             WithinRel(alpha * kt, 2.e-8));
  CHECK_THAT((model.compute_pressure(t + dt, v, p) -
              model.compute_pressure(t - dt, v, p)) /
                 (2. * dt),
             WithinRel(alpha * kt, 2.e-8));
  CHECK_THAT(gamma, WithinRel(v * alpha * kt / cv, 2.e-14));
  CHECK_THAT(model.compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(kt * cp / cv, 2.e-14));
  CHECK_THAT(model.compute_molar_internal_energy(pressure, t, v, p),
             WithinRel(f(t, v) + t * s(t, v), 2.e-13) || WithinAbs(0., 1.e-8));
  CHECK_THAT(model.compute_volume(pressure, t, p), WithinRel(v, 2.e-13));
}

TEST_CASE("AA preserves caloric strategy and gauge independence", "[aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  eos::AA model;
  const double t = *p.T_0, v = *p.V_0, dt = .1;
  CHECK((model.compute_grueneisen_parameter(0., t + dt, v, p) -
         model.compute_grueneisen_parameter(0., t - dt, v, p)) /
            (2. * dt) >
        0.);
  const double pressure = model.compute_pressure(7000., .7 * v, p);
  const double kt =
      model.compute_isothermal_bulk_modulus_reuss(pressure, 7000., .7 * v, p);
  p.E_0 = 1.e12;
  p.S_0 = 1.e9;
  CHECK(model.compute_pressure(7000., .7 * v, p) == pressure);
  CHECK(model.compute_isothermal_bulk_modulus_reuss(pressure, 7000., .7 * v,
                                                    p) == kt);
  CHECK(model.compute_gibbs_free_energy(pressure, 7000., .7 * v, p) ==
        model.compute_helmholtz_free_energy(pressure, 7000., .7 * v, p) +
            pressure * .7 * v);
}

TEST_CASE("AA formula-unit scaling", "[aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  eos::AA model;
  const double scale = GENERATE(.001, 1., 1000.);
  for (auto field :
       {&types::MineralParams::V_0, &types::MineralParams::molar_mass,
        &types::MineralParams::napfu, &types::MineralParams::S_0,
        &types::MineralParams::E_0, &types::MineralParams::xi_0})
    *(p.*field) *= scale;
  for (auto field : {&types::MineralParams::aa_a, &types::MineralParams::aa_b})
    for (double &x : *(p.*field))
      x *= scale;
  for (double &x : *p.aa_lmda)
    x *= scale;
  *p.grueneisen_prime /= scale;
  const double v = .7 * *p.V_0, t = 5000.;
  const double pressure = model.compute_pressure(t, v, p);
  REQUIRE(std::isfinite(pressure));
  CHECK_THAT(model.compute_volume(pressure, t, p), WithinRel(v, 2.e-13));
}

TEST_CASE("AA required parameters and invalid states", "[aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  eos::AA model;
  SECTION("Required scalar") {
    auto field = GENERATE(
        &types::MineralParams::P_0, &types::MineralParams::T_0,
        &types::MineralParams::V_0, &types::MineralParams::K_0,
        &types::MineralParams::grueneisen_prime, &types::MineralParams::theta);
    (p.*field).reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Required array") {
    p.aa_a.reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid coefficient") {
    (*p.aa_F)[1] = GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid state") {
    const double t =
        GENERATE(0., -10., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.compute_pressure(t, *p.V_0, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(model.compute_volume(0., t, p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_pressure(1811., 0., p),
                    std::invalid_argument);
  }
  SECTION("Pressure outside the retained volume bracket") {
    CHECK_THROWS_AS(model.compute_volume(1.e30, 1811., p),
                    std::invalid_argument);
  }
}
