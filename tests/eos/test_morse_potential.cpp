// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/morse_potential.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
types::MineralParams morse_params() {
  types::MineralParams p;
  p.V_0 = 11.24e-6;
  p.K_0 = 161.e9;
  p.Kprime_0 = 3.8;
  return p;
}
} // namespace

TEST_CASE("Morse factory and reference state", "[morse][eos]") {
  auto model = eos::make_eos(types::EOSType::Morse);
  REQUIRE(dynamic_cast<eos::Morse *>(model.get()));
  auto p = morse_params();
  model->validate_parameters(p);
  CHECK(*p.F_0 == 0.);
  CHECK(*p.P_0 == 0.);
  CHECK_FALSE(p.E_0.has_value());
  p.F_0 = -125000.;
  p.P_0 = 2.5e9;
  const double pressure = *p.P_0, v = *p.V_0;
  const double t = GENERATE(0., 300., 2000.);
  CHECK(model->compute_pressure(t, v, p) == pressure);
  CHECK(model->compute_volume(pressure, t, p) == v);
  CHECK(model->compute_isothermal_bulk_modulus_reuss(pressure, t, v, p) ==
        *p.K_0);
  CHECK(model->compute_isentropic_bulk_modulus_reuss(pressure, t, v, p) ==
        *p.K_0);
  CHECK(model->compute_helmholtz_free_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_molar_internal_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_gibbs_free_energy(pressure, t, v, p) ==
        *p.F_0 + pressure * v);
  CHECK(model->compute_enthalpy(pressure, t, v, p) == *p.F_0 + pressure * v);
  CHECK(model->compute_entropy(pressure, t, v, p) == 0.);
  CHECK(model->compute_thermal_expansivity(pressure, t, v, p) == 0.);
  CHECK(model->compute_grueneisen_parameter(pressure, t, v, p) == 0.);
  CHECK(model->compute_shear_modulus(pressure, t, v, p) == 0.);
  CHECK(model->compute_molar_heat_capacity_p(pressure, t, v, p) == 1.e-99);
  CHECK(model->compute_molar_heat_capacity_v(pressure, t, v, p) == 1.e-99);
}

TEST_CASE("Morse Python reference values", "[morse][eos]") {
  // burnman.eos.morse_potential at
  // historical_commits.initial_snapshots_and_tables
  // (tests/reference/burnman.json); the local
  // historical_commits.local_morse_formulas (tests/reference/burnman.json)
  // formulas are identical.
  struct Reference {
    double ratio, pressure, bulk, energy;
  };
  const auto row = GENERATE(
      Reference{.4, 723392993278.80225, 1933782242684.3271, 1233026.3441978362},
      Reference{.8, 54380737902.413048, 342468842218.10699, 51315.973318865021},
      Reference{.99, 1649267673.9802709, 167227989173.40518,
                91.947301585943549},
      Reference{1., 0., 161000000000., 0.},
      Reference{1.2, -20625544382.82692, 72463193693.771759,
                26895.832468530567});
  eos::Morse model;
  auto p = morse_params();
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
             WithinRel(row.energy, 1.e-10) || WithinAbs(0., 1.e-8));
}

TEST_CASE("Morse thermodynamic derivatives and Kprime=1 limit",
          "[morse][eos]") {
  eos::Morse model;
  auto p = morse_params();
  p.Kprime_0 = GENERATE(0., 1. - 1.e-12, 1., 1. + 1.e-12, 3.8, 10.);
  p.P_0 = GENERATE(0., 2.5e9);
  p.F_0 = -125000.;
  model.validate_parameters(p);
  const double ratio = GENERATE(.65, .95, 1.2), v = ratio * *p.V_0;
  const double pressure = model.compute_pressure(300., v, p), dv = v * 1.e-5;
  CAPTURE(*p.Kprime_0, *p.P_0, ratio);
  const auto f = [&](double volume) {
    return model.compute_helmholtz_free_energy(pressure, 300., volume, p);
  };
  CHECK_THAT(model.compute_volume(pressure, 300., p), WithinRel(v, 1.e-13));
  CHECK_THAT(-(f(v + dv) - f(v - dv)) / (2. * dv), WithinRel(pressure, 2.e-8));
  CHECK_THAT(-v *
                 (model.compute_pressure(300., v + dv, p) -
                  model.compute_pressure(300., v - dv, p)) /
                 (2. * dv),
             WithinRel(model.compute_isothermal_bulk_modulus_reuss(pressure,
                                                                   300., v, p),
                       2.e-8));
  const auto g = [&](double pressure_at) {
    return model.compute_gibbs_free_energy(
        pressure_at, 300., model.compute_volume(pressure_at, 300., p), p);
  };
  CHECK_THAT((g(pressure + 1.e5) - g(pressure - 1.e5)) / 2.e5,
             WithinRel(v, 2.e-8));
  if (*p.Kprime_0 == 1.) {
    const double s = std::cbrt(ratio);
    CHECK_THAT(pressure - *p.P_0,
               WithinRel(3. * *p.K_0 * (1. - s) / (s * s), 1.e-13));
    CHECK_THAT(
        model.compute_isothermal_bulk_modulus_reuss(pressure, 300., v, p),
        WithinRel(*p.K_0 * (2. - s) / (s * s), 1.e-13));
  }
}

TEST_CASE("Morse energy near the reference volume", "[morse][eos]") {
  eos::Morse model;
  auto p = morse_params();
  p.Kprime_0 = GENERATE(1., 1. + 1.e-12, 3.8);
  model.validate_parameters(p);
  const double v = *p.V_0 * GENERATE(1. - 1.e-8, 1. + 1.e-8);
  const double dv = v - *p.V_0;
  CHECK_THAT(model.compute_helmholtz_free_energy(0., 300., v, p),
             WithinRel(.5 * *p.K_0 / *p.V_0 * dv * dv, 1.e-7));
}

TEST_CASE("Morse inversion stays on the stable branch", "[morse][eos]") {
  eos::Morse model;
  auto p = morse_params();
  p.Kprime_0 = 1.;
  model.validate_parameters(p);
  // At Kprime=1 the spinodal is V=8*V_0, P=-3*K_0/4. At a less negative
  // pressure there are two volume roots; choose the one connected to V_0.
  const double pressure = -.7 * *p.K_0;
  const double s = 2. / (1. + std::sqrt(1. + 4. * pressure / (3. * *p.K_0)));
  CHECK_THAT(model.compute_volume(pressure, 300., p),
             WithinRel(*p.V_0 * s * s * s, 1.e-13));
  CHECK_THROWS_AS(model.compute_volume(-.8 * *p.K_0, 300., p),
                  std::invalid_argument);
  const double v = .05 * *p.V_0;
  CHECK_THAT(model.compute_volume(model.compute_pressure(300., v, p), 300., p),
             WithinRel(v, 1.e-13));
}

TEST_CASE("Morse parameter and state validation", "[morse][eos]") {
  eos::Morse model;
  auto p = morse_params();
  SECTION("Required parameters") {
    const auto field =
        GENERATE(&types::MineralParams::V_0, &types::MineralParams::K_0,
                 &types::MineralParams::Kprime_0);
    (p.*field).reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Helmholtz reference") {
    p.E_0 = 0.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid reference") {
    p.K_0 = GENERATE(0., -1., std::numeric_limits<double>::infinity());
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid states") {
    model.validate_parameters(p);
    const double v =
        GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.compute_pressure(300., v, p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_morse_bulk_modulus(v, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(model.compute_helmholtz_free_energy(0., 300., v, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        model.compute_volume(std::numeric_limits<double>::quiet_NaN(), 300., p),
        std::invalid_argument);
  }
}
