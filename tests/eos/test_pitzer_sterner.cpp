// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/pitzer_sterner.hpp"
#include "burnman/minerals/water.hpp"
#include "burnman/utils/constants.hpp"
#include "ps1994_reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
types::MineralParams water(bool ideal_reference = false) {
  auto p = minerals::Pitzer_Sterner_1994::H2O_Pitzer_Sterner()->params;
  if (ideal_reference) {
    p.Cp = types::CpParams{40.1, .008656, 487500., -251.2};
    p.H_0 = -241810.;
    p.S_0 = 188.8;
  }
  eos::PitzerSterner().validate_parameters(p);
  return p;
}
} // namespace

TEST_CASE("Original PS1994 golden states require their reference gas constant",
          "[eos][pitzer_sterner][original]") {
  // Upstream patches Python's R. A Python attribute patch cannot configure a
  // compiled EOS: set its existing alternate-R constructor explicitly instead.
  const bool ideal_reference = GENERATE(false, true);
  auto phase = ideal_reference
                   ? minerals::water_fluid()
                   : minerals::Pitzer_Sterner_1994::H2O_Pitzer_Sterner();
  if (!ideal_reference)
    phase->set_method(std::make_shared<eos::PitzerSterner>(8.31451));
  auto reference = phase->copy();
  for (const auto &s : ps1994_reference::states) {
    INFO("HP reference=" << ideal_reference << " P=" << s.pressure
                         << " T=" << s.temperature);
    phase->set_state(s.pressure, s.temperature);
    reference->set_state(1.e5, s.temperature);
    CHECK_THAT(phase->get_molar_volume(), WithinRel(s.volume, 2.e-12));
    CHECK_THAT(phase->get_isothermal_bulk_modulus_reuss(),
               WithinRel(s.bulk_modulus, 2.e-8));
    CHECK_THAT(phase->get_thermal_expansivity(),
               WithinRel(s.expansivity, 2.e-6));
    CHECK_THAT(
        phase->get_molar_gibbs() - reference->get_molar_gibbs(),
        WithinAbs(s.delta_gibbs, 2.e-8 + 2.e-11 * std::abs(s.delta_gibbs)));
    CHECK_THAT(
        phase->get_molar_entropy() - reference->get_molar_entropy(),
        WithinAbs(s.delta_entropy, 2.e-7 + 2.e-7 * std::abs(s.delta_entropy)));
    CHECK_THAT(phase->get_molar_heat_capacity_p() -
                   reference->get_molar_heat_capacity_p(),
               WithinAbs(s.delta_cp, 2.e-3 + 2.e-5 * std::abs(s.delta_cp)));
    CHECK(phase->get_shear_modulus() == 0.);
  }
}

TEST_CASE("PitzerSterner derivatives follow its Helmholtz potential",
          "[eos][pitzer_sterner]") {
  const auto p = water(GENERATE(false, true));
  const double t = GENERATE(300., 700., 1600.), v = 2.5e-5, dt = .002,
               dv = v * 1.e-5;
  const eos::PitzerSterner e;
  const auto f = [&](double T, double V) {
    return e.compute_helmholtz_free_energy(0., T, V, p);
  };
  const double pressure = e.compute_pressure(t, v, p),
               entropy = e.compute_entropy(pressure, t, v, p),
               cv = e.compute_molar_heat_capacity_v(pressure, t, v, p),
               bulk =
                   e.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
               alpha = e.compute_thermal_expansivity(pressure, t, v, p),
               cp = e.compute_molar_heat_capacity_p(pressure, t, v, p);
  CHECK_THAT(pressure,
             WithinRel(-(f(t, v + dv) - f(t, v - dv)) / (2. * dv), 2.e-7));
  CHECK_THAT(entropy,
             WithinRel(-(f(t + dt, v) - f(t - dt, v)) / (2. * dt), 2.e-7));
  CHECK_THAT(bulk, WithinRel(-v *
                                 (e.compute_pressure(t, v + dv, p) -
                                  e.compute_pressure(t, v - dv, p)) /
                                 (2. * dv),
                             2.e-7));
  CHECK_THAT(cv, WithinRel(t *
                               (e.compute_entropy(pressure, t + dt, v, p) -
                                e.compute_entropy(pressure, t - dt, v, p)) /
                               (2. * dt),
                           2.e-7));
  CHECK_THAT(alpha * bulk, WithinRel((e.compute_pressure(t + dt, v, p) -
                                      e.compute_pressure(t - dt, v, p)) /
                                         (2. * dt),
                                     2.e-7));
  CHECK_THAT(cp, WithinRel(cv + t * v * alpha * alpha * bulk, 1.e-13));
  CHECK_THAT(e.compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(bulk * cp / cv, 1.e-13));
  CHECK_THAT(e.compute_grueneisen_parameter(pressure, t, v, p),
             WithinRel(alpha * bulk * v / cv, 1.e-13));
  CHECK_THAT(e.compute_gibbs_free_energy(pressure, t, v, p),
             WithinRel(f(t, v) + pressure * v, 1.e-13));
  CHECK_THAT(e.compute_enthalpy(pressure, t, v, p),
             WithinRel(f(t, v) + t * entropy + pressure * v, 1.e-13));
}

TEST_CASE("PitzerSterner selects stable liquid and vapour volumes over its "
          "parameter domain",
          "[eos][pitzer_sterner]") {
  auto p = water();
  auto e = eos::make_eos(types::EOSType::PitzerSterner);
  for (const auto &pt : {std::pair{1.e3, 300.},
                         {1.e5, 300.},
                         {1.e8, 800.},
                         {1.e10, 2000.},
                         {8.e10, 5000.}}) {
    const double pressure = pt.first, t = pt.second,
                 v = e->compute_volume(pressure, t, p);
    INFO("P=" << pressure << " T=" << t);
    CHECK_THAT(e->compute_pressure(t, v, p), WithinRel(pressure, 2.e-8));
    CHECK(e->compute_isothermal_bulk_modulus_reuss(pressure, t, v, p) > 0.);
    if (t == 300.)
      CHECK((pressure == 1.e3 ? v > 1.e-3 : v < 3.e-5));
  }
}

TEST_CASE(
    "PitzerSterner recovers an ideal gas and zero exponential coefficients",
    "[eos][pitzer_sterner]") {
  auto p = water(true);
  p.c_coeffs = types::PitzerSternerCoefficients{};
  (*p.c_coeffs)[1][3] = 1.;
  const double r = constants::physics::gas_constant;
  p.Cp = types::CpParams{3.5 * r, 0., 0., 0.};
  const eos::PitzerSterner e;
  const double t = GENERATE(300., 1500.), pressure = GENERATE(1.e3, 1.e7),
               v = e.compute_volume(pressure, t, p);
  CHECK_THAT(v, WithinRel(r * t / pressure, 1.e-12));
  CHECK_THAT(e.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(pressure, 1.e-12));
  CHECK_THAT(e.compute_molar_heat_capacity_v(pressure, t, v, p),
             WithinRel(2.5 * r, 1.e-13));
  CHECK_THAT(e.compute_molar_heat_capacity_p(pressure, t, v, p),
             WithinRel(3.5 * r, 1.e-13));
  CHECK_THAT(e.compute_thermal_expansivity(pressure, t, v, p),
             WithinRel(1. / t, 1.e-12));
}

TEST_CASE("PitzerSterner validates parameters and optional state limits",
          "[eos][pitzer_sterner]") {
  eos::PitzerSterner e;
  auto p = water();
  SECTION("Coefficient table") {
    p.c_coeffs.reset();
    CHECK_THROWS_AS(e.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Nonfinite coefficients") {
    (*p.c_coeffs)[0][0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(e.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Thermal reference") {
    p.Debye_n.reset();
    CHECK_THROWS_AS(e.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Domain") {
    p.T_min = 500.;
    p.T_max = 1700.;
    p.P_max = 5.e9;
    e.validate_parameters(p);
    CHECK_THROWS_AS(e.compute_volume(1.e5, 499., p), std::invalid_argument);
    CHECK_THROWS_AS(e.compute_volume(1.e5, 1701., p), std::invalid_argument);
    CHECK_THROWS_AS(e.compute_volume(5.1e9, 1000., p), std::invalid_argument);
    CHECK_THROWS_AS(e.compute_volume(0., 1000., p), std::invalid_argument);
    CHECK_THROWS_AS(e.compute_pressure(1000., 0., p), std::invalid_argument);
  }
}
