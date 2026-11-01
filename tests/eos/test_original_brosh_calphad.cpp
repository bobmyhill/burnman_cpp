// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of tests/test_brosh_calphad.py; reference:
// tests/reference/burnman.json.

#include "../../src/eos/reference_states.hpp"
#include "burnman/eos/brosh_calphad.hpp"
#include "burnman/minerals/SE_2015.hpp"
#include "burnman/utils/constants.hpp"
#include "original_test_helpers.hpp"

using namespace burnman;
using original::check_close;
using original::quad;

namespace {
using Factory = std::shared_ptr<Mineral> (*)();
constexpr Factory phases[]{
    minerals::SE_2015::bcc_iron, minerals::SE_2015::fcc_iron,
    minerals::SE_2015::hcp_iron, minerals::SE_2015::liquid_iron};
} // namespace

TEST_CASE("Original Brosh low temperature caloric continuation",
          "[original][brosh][eos]") {
  const eos::BroshCalphad model;
  for (auto phase : phases) {
    const auto p = phase()->params;
    const double pressure = 300.e9, tr = *p.T_0;
    const auto thermal = eos::detail::brosh_thermal(pressure, p);
    const auto cp = [&](double t) {
      const double x = thermal.theta / t, y = std::exp(-x);
      return 3. * *p.napfu * constants::physics::gas_constant * x * x * y /
                 ((1. - y) * (1. - y)) +
             thermal.interpolation * t / tr * *p.delta_Cpr;
    };
    const double gr = model.compute_gibbs_free_energy(pressure, tr, 0., p);
    const double sr = model.compute_entropy(pressure, tr, 0., p);
    for (double t : {1., 100.}) {
      CAPTURE(*p.name, t);
      const double expected_s =
          sr + quad([&](double u) { return cp(u) / u; }, tr, t);
      const double expected_g =
          gr - sr * (t - tr) -
          quad([&](double u) { return (t - u) * cp(u) / u; }, tr, t);
      check_close(model.compute_entropy(pressure, t, 0., p), expected_s);
      check_close(model.compute_gibbs_free_energy(pressure, t, 0., p),
                  expected_g, 1.e-12);
      check_close(model.compute_molar_heat_capacity_p(pressure, t, 0., p),
                  cp(t), 1.e-13);
      CHECK(model.compute_molar_heat_capacity_p(pressure, t, 0., p) > 0.);
    }
  }
}

TEST_CASE("Original Brosh final SGTE segment extrapolation",
          "[original][brosh][eos]") {
  for (auto phase : phases) {
    const auto p = phase()->params;
    auto last = p;
    last.gibbs_coefficients =
        types::GibbsCoefficients{p.gibbs_coefficients->back()};
    for (double t : {12000., 12000.01, 15000.}) {
      CAPTURE(*p.name, t);
      const auto all = eos::detail::brosh_sgte(t, p);
      const auto final = eos::detail::brosh_sgte(t, last);
      CHECK(all.gibbs == final.gibbs);
      CHECK(all.entropy == final.entropy);
      CHECK(all.heat_capacity == final.heat_capacity);
    }
  }
}

TEST_CASE("Original Brosh reference calorics", "[original][brosh][eos]") {
  const eos::BroshCalphad model;
  for (auto phase : phases) {
    const auto p = phase()->params;
    for (double t : {*p.T_0, 1000., 3000., 8000.}) {
      CAPTURE(*p.name, t);
      const auto ref = eos::detail::brosh_sgte(t, p);
      check_close(model.compute_gibbs_free_energy(0., t, 0., p), ref.gibbs,
                  1.e-13, 1.e-9);
      check_close(model.compute_entropy(0., t, 0., p), ref.entropy);
      check_close(model.compute_molar_heat_capacity_p(0., t, 0., p),
                  ref.heat_capacity);
    }
  }
}

TEST_CASE("Original Brosh thermodynamic derivatives",
          "[original][brosh][eos]") {
  const eos::BroshCalphad model;
  for (auto phase : phases) {
    const auto p = phase()->params;
    for (double t : {100., 300., 2000., 8000.}) {
      CAPTURE(*p.name, t);
      const double pressure = 1.e10, dp = 1.e7, dt = .01;
      const double v = model.compute_volume(pressure, t, p);
      const auto g = [&](double P, double T) {
        return model.compute_gibbs_free_energy(P, T, 0., p);
      };
      const auto s = [&](double P, double T) {
        return model.compute_entropy(P, T, 0., p);
      };
      const auto volume = [&](double P, double T) {
        return model.compute_volume(P, T, p);
      };
      const double alpha = model.compute_thermal_expansivity(pressure, t, v, p);
      check_close(-(g(pressure, t + dt) - g(pressure, t - dt)) / (2. * dt),
                  s(pressure, t), 1.e-7);
      check_close((g(pressure + dp, t) - g(pressure - dp, t)) / (2. * dp), v,
                  1.e-6);
      check_close(t * (s(pressure, t + dt) - s(pressure, t - dt)) / (2. * dt),
                  model.compute_molar_heat_capacity_p(pressure, t, v, p),
                  1.e-7);
      check_close((volume(pressure, t + dt) - volume(pressure, t - dt)) /
                      (2. * dt) / v,
                  alpha, 1.e-7);
      check_close(-(s(pressure + dp, t) - s(pressure - dp, t)) / (2. * dp),
                  alpha * v, 1.e-6);
      // Native EOS exposes this derivative through K_T = -V/(dV/dP).
      check_close(
          (volume(pressure + dp, t) - volume(pressure - dp, t)) / (2. * dp),
          -v / model.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
          1.e-6, 1.e-24);
    }
  }
}
