// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/make_eos.hpp"
#include "original_test_helpers.hpp"
#include <array>

using namespace burnman;
namespace {
constexpr std::array models{types::EOSType::BM3Shear2, types::EOSType::BM3,
                            types::EOSType::BM4,       types::EOSType::Vinet,
                            types::EOSType::Morse,     types::EOSType::SPOCK};
types::MineralParams parameters(types::EOSType type) {
  types::MineralParams p;
  p.equation_of_state = type;
  p.F_0 = 1000.;
  p.T_0 = 100.;
  p.P_0 = 0.;
  p.V_0 = 7.95626e-6;
  p.K_0 = 109.7e9;
  p.Kprime_0 = 4.66;
  p.Kprime_inf = 3.;
  p.Kdprime_0 = type == types::EOSType::BM4 ? -.043e-9 : -4.66 / 100.e9;
  p.molar_mass = .055845;
  return p;
}
} // namespace

TEST_CASE("Cold EOS reference pressure work and energy calibration",
          "[eos][reference_pressure]") {
  for (auto type : models) {
    auto model = eos::make_eos(type);
    auto zero = parameters(type);
    model->validate_parameters(zero);
    const auto helmholtz = [&](double v, const types::MineralParams &p) {
      const double pressure = model->compute_pressure(*p.T_0, v, p);
      return model->compute_gibbs_free_energy(pressure, *p.T_0, v, p) -
             pressure * v;
    };
    for (double pressure : {0., 1.e5, 2.5e9, 1.e10}) {
      auto p = zero;
      p.P_0 = pressure;
      CAPTURE(type, pressure);
      original::check_close(helmholtz(*p.V_0, p), *p.F_0, 1.e-12, 1.e-8);
      for (double ratio : {.8, 1., 1.1}) {
        CAPTURE(ratio);
        const double v = ratio * *p.V_0;
        original::check_close(helmholtz(v, p) - helmholtz(v, zero),
                              pressure * (*p.V_0 - v), 1.e-10, 1.e-8);
      }
    }
  }
}

TEST_CASE("Cold EOS resolves the consistency checker's reference pressure step",
          "[eos][reference_pressure][roots]") {
  for (auto type : models) {
    auto model = eos::make_eos(type);
    auto p = parameters(type);
    model->validate_parameters(p);
    for (double pressure : {1.e5, 2.5e9, 1.e10}) {
      p.P_0 = pressure;
      CAPTURE(type, pressure);
      const double dp = 1000.;
      const double v0 = model->compute_volume(pressure, *p.T_0, p);
      const double v1 = model->compute_volume(pressure + dp, *p.T_0, p);
      REQUIRE(v1 < v0);
      original::check_close(-.5 * (v1 + v0) * dp / (v1 - v0),
                            model->compute_isothermal_bulk_modulus_reuss(
                                pressure + .5 * dp, *p.T_0, .5 * (v1 + v0), p),
                            1.e-4);
    }
  }
}
