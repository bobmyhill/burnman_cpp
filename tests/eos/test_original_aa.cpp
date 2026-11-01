// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of tests/test_aa.py; reference: tests/reference/burnman.json.

#include "../../src/eos/reference_states.hpp"
#include "burnman/eos/aa.hpp"
#include "burnman/minerals/other.hpp"
#include "burnman/utils/constants.hpp"
#include "original_test_helpers.hpp"
#include <array>

using namespace burnman;
using original::check_close;
using original::quad;

TEST_CASE("Original AA reference pressure work and temperature",
          "[original][aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  p.P_0 = 2.5e9;
  auto zero = p;
  zero.P_0 = 0.;
  const eos::AA model;
  for (double ratio : {.55, .8, 1., 1.3}) {
    CAPTURE(ratio);
    const double v = ratio * *p.V_0;
    const auto s = eos::detail::aa_reference(v, p);
    check_close(s.energy - eos::detail::aa_reference(v, zero).energy,
                *p.P_0 * (*p.V_0 - v), 1.e-13, 1.e-9);
    check_close(model.compute_pressure(s.temperature, v, p), s.pressure, 1.e-13,
                1.e-4);
    check_close(
        model.compute_isentropic_bulk_modulus_reuss(0., s.temperature, v, p),
        s.bulk_modulus, 1.e-13);
    CHECK(model.compute_entropy(0., s.temperature, v, p) == *p.S_0);
    const double dl = 1.e-5;
    const double slope =
        (std::log(eos::detail::aa_reference(v * std::exp(-dl), p).temperature) -
         std::log(eos::detail::aa_reference(v * std::exp(dl), p).temperature)) /
        (2. * dl);
    const double gamma =
        *p.grueneisen_0 +
        *p.grueneisen_prime * std::pow(ratio, -*p.grueneisen_n) * s.energy;
    check_close(slope, gamma, 1.e-9);
  }
}

TEST_CASE("Original AA reference temperature singular exponents",
          "[original][aa][eos]") {
  for (double exponent : {0., -2. / 3., -4. / 3., -2., -8. / 3., 1.})
    for (double ratio : {.6, 1., 1.3}) {
      CAPTURE(exponent, ratio);
      auto p = minerals::other::liquid_iron()->params;
      p.grueneisen_n = exponent;
      p.P_0 = 2.5e9;
      const double integral = quad(
          [&](double z) {
            return *p.grueneisen_0 +
                   *p.grueneisen_prime * std::exp(exponent * z) *
                       eos::detail::aa_reference(*p.V_0 * std::exp(-z), p)
                           .energy;
          },
          0., -std::log(ratio), 1.e-12);
      check_close(eos::detail::aa_reference(ratio * *p.V_0, p).temperature,
                  *p.T_0 * std::exp(integral), 1.e-13);
    }
}

TEST_CASE("Original AA small strain energy", "[original][aa][eos]") {
  const auto p = minerals::other::liquid_iron()->params;
  const double ell = 1.e-10, v = *p.V_0 * std::exp(-ell);
  const double expected = *p.V_0 * (*p.P_0 * ell + .5 * *p.K_0 * ell * ell);
  check_close(eos::detail::aa_reference(v, p).energy, expected, 2.e-6, 0.);
}

TEST_CASE("Original AA zero reference pressure free energy",
          "[original][aa][eos]") {
  auto p = minerals::other::liquid_iron()->params;
  p.P_0 = 0.;
  const eos::AA model;
  const std::array<std::array<double, 5>, 5> cases{{
      {5000., 5000., -630205.9208554473, 166.79387460706806,
       37.874064801504474},
      {7019., 1811., -181100.00000000003, 100., 35.9580348274425},
      {10000., 7000., -678896.7891424899, 126.93412115230728,
       37.84536603348719},
      {13000., 7000., -411867.4803151762, 110.90353154347144,
       33.90557395400822},
      {15000., 15000., -1118472.526226906, 129.97493939581932,
       36.582954003664966},
  }};
  for (const auto &[rho, t, f, s, cv] : cases) {
    CAPTURE(rho, t);
    const double v = *p.molar_mass / rho;
    const double dc =
        1.5 * *p.napfu * (constants::physics::gas_constant - 8.31446261815324);
    const double ts = eos::detail::aa_reference(v, p).temperature;
    const double log_ratio = std::log(t / ts);
    check_close(model.compute_helmholtz_free_energy(0., t, v, p),
                f + dc * (t - ts - t * log_ratio), 5.e-13, 1.e-7);
    check_close(model.compute_entropy(0., t, v, p), s + dc * log_ratio, 5.e-13);
    check_close(model.compute_molar_heat_capacity_v(0., t, v, p), cv + dc,
                2.e-15);
  }
}

TEST_CASE("Original AA caloric integrals", "[original][aa][eos]") {
  const auto p = minerals::other::liquid_iron()->params;
  const eos::AA model;
  for (const auto &[rho, t] :
       {std::pair{5000., 7000.}, {10000., 5000.}, {15000., 4000.}}) {
    CAPTURE(rho, t);
    const double v = *p.molar_mass / rho;
    const auto s = eos::detail::aa_reference(v, p);
    const double energy =
        *p.E_0 + s.energy +
        quad(
            [&](double temperature) {
              return model.compute_molar_heat_capacity_v(0., temperature, v, p);
            },
            s.temperature, t);
    const double entropy =
        *p.S_0 + quad(
                     [&](double temperature) {
                       return model.compute_molar_heat_capacity_v(
                                  0., temperature, v, p) /
                              temperature;
                     },
                     s.temperature, t);
    check_close(model.compute_molar_internal_energy(0., t, v, p), energy,
                2.e-13, 1.e-8);
    check_close(model.compute_entropy(0., t, v, p), entropy, 2.e-13);
  }
}

TEST_CASE("Original AA first law and Maxwell derivatives",
          "[original][aa][eos]") {
  const auto p = minerals::other::liquid_iron()->params;
  const eos::AA model;
  for (const auto &[rho, t] : {std::pair{7019., 1811.},
                               {5000., 7000.},
                               {10000., 5000.},
                               {13000., 7000.}}) {
    CAPTURE(rho, t);
    const double v = *p.molar_mass / rho, dv = 1.e-5 * v, dt = .05;
    const double pressure = model.compute_pressure(t, v, p);
    const double kt = model.compute_isothermal_bulk_modulus_reuss(0., t, v, p);
    const double akt = model.compute_thermal_expansivity(0., t, v, p) * kt;
    const auto f = [&](double volume, double temperature) {
      return model.compute_helmholtz_free_energy(0., temperature, volume, p);
    };
    const auto s = [&](double volume, double temperature) {
      return model.compute_entropy(0., temperature, volume, p);
    };
    const auto energy = [&](double volume, double temperature) {
      return model.compute_molar_internal_energy(0., temperature, volume, p);
    };
    check_close(-(f(v + dv, t) - f(v - dv, t)) / (2. * dv), pressure, 5.e-8,
                100.);
    check_close(-(f(v, t + dt) - f(v, t - dt)) / (2. * dt), s(v, t), 1.e-8);
    check_close((energy(v, t + dt) - energy(v, t - dt)) / (2. * dt),
                model.compute_molar_heat_capacity_v(0., t, v, p), 1.e-8);
    check_close((model.compute_pressure(t + dt, v, p) -
                 model.compute_pressure(t - dt, v, p)) /
                    (2. * dt),
                akt, 1.e-8);
    check_close((s(v + dv, t) - s(v - dv, t)) / (2. * dv), akt, 1.e-8);
    check_close(-v *
                    (model.compute_pressure(t, v + dv, p) -
                     model.compute_pressure(t, v - dv, p)) /
                    (2. * dv),
                kt, 1.e-8);
    check_close((energy(v + dv, t) - energy(v - dv, t)) / (2. * dv),
                t * akt - pressure, 1.e-8, 100.);
  }
}

TEST_CASE("Original AA energy and entropy gauge independence",
          "[original][aa][eos]") {
  const auto original = minerals::other::liquid_iron()->params;
  auto p = original;
  p.E_0 = 1.e12;
  p.S_0 = 1.e9;
  const double t = 7000., v = .7 * *p.V_0;
  const eos::AA model;
  const auto derivatives = [&](const auto &params) {
    const double kt =
        model.compute_isothermal_bulk_modulus_reuss(0., t, v, params);
    return std::array<double, 3>{
        model.compute_pressure(t, v, params), kt,
        kt * model.compute_thermal_expansivity(0., t, v, params)};
  };
  CHECK(derivatives(p) == derivatives(original));
  check_close(model.compute_helmholtz_free_energy(0., t, v, p) -
                  model.compute_helmholtz_free_energy(0., t, v, original),
              (*p.E_0 - *original.E_0) - t * (*p.S_0 - *original.S_0), 1.e-15);
}
