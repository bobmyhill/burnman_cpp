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

#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/murnaghan.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman;
using namespace Catch::Matchers;

namespace {
types::MineralParams murnaghan_params() {
  types::MineralParams params;
  params.equation_of_state = types::EOSType::Murnaghan;
  params.V_0 = 11.24e-6;
  params.K_0 = 161.0e9;
  params.Kprime_0 = 3.8;
  return params;
}
} // namespace

TEST_CASE("Murnaghan parameter validation", "[murnaghan][eos]") {
  eos::Murnaghan murnaghan;
  auto params = murnaghan_params();
  SECTION("Required parameters") {
    auto missing =
        GENERATE(&types::MineralParams::V_0, &types::MineralParams::K_0,
                 &types::MineralParams::Kprime_0);
    (params.*missing).reset();
    REQUIRE_THROWS_AS(murnaghan.validate_parameters(params),
                      std::invalid_argument);
  }
  SECTION("Defaults follow the existing athermal EOS") {
    murnaghan.validate_parameters(params);
    REQUIRE(*params.E_0 == 0.0);
    REQUIRE(*params.P_0 == 0.0);
    REQUIRE(std::isnan(*params.G_0));
    REQUIRE(std::isnan(*params.Gprime_0));
  }
  SECTION("Explicit reference values are preserved") {
    params.E_0 = -125000.0;
    params.P_0 = 2.5e9;
    params.G_0 = 131.0e9;
    params.Gprime_0 = 2.1;
    murnaghan.validate_parameters(params);
    REQUIRE(*params.E_0 == -125000.0);
    REQUIRE(*params.P_0 == 2.5e9);
    REQUIRE(*params.G_0 == 131.0e9);
    REQUIRE(*params.Gprime_0 == 2.1);
  }
  SECTION("Helmholtz reference energy follows the BM4 convention") {
    params.F_0 = -125000.0;
    murnaghan.validate_parameters(params);
    REQUIRE(*params.E_0 == *params.F_0);
  }
}

TEST_CASE("Murnaghan factory and reference state", "[murnaghan][eos]") {
  auto murnaghan = eos::make_eos(types::EOSType::Murnaghan);
  REQUIRE(dynamic_cast<eos::Murnaghan *>(murnaghan.get()) != nullptr);
  auto params = murnaghan_params();
  params.E_0 = -125000.0;
  params.P_0 = GENERATE(0.0, 2.5e9);
  murnaghan->validate_parameters(params);
  const double p = *params.P_0;
  const double v = *params.V_0;
  const double t = GENERATE(0.0, 300.0, 2000.0);
  CHECK(murnaghan->compute_pressure(t, v, params) == p);
  CHECK(murnaghan->compute_volume(p, t, params) == v);
  CHECK(murnaghan->compute_isothermal_bulk_modulus_reuss(p, t, v, params) ==
        *params.K_0);
  CHECK(murnaghan->compute_isentropic_bulk_modulus_reuss(p, t, v, params) ==
        *params.K_0);
  CHECK(murnaghan->compute_molar_internal_energy(p, t, v, params) ==
        *params.E_0);
  CHECK(murnaghan->compute_helmholtz_free_energy(p, t, v, params) ==
        *params.E_0);
  CHECK(murnaghan->compute_gibbs_free_energy(p, t, v, params) ==
        *params.E_0 + p * v);
  CHECK(murnaghan->compute_enthalpy(p, t, v, params) == *params.E_0 + p * v);
  CHECK(murnaghan->compute_entropy(p, t, v, params) == 0.0);
  CHECK(murnaghan->compute_thermal_expansivity(p, t, v, params) == 0.0);
  CHECK(murnaghan->compute_grueneisen_parameter(p, t, v, params) == 0.0);
  CHECK(murnaghan->compute_shear_modulus(p, t, v, params) == 0.0);
  CHECK(murnaghan->compute_molar_heat_capacity_p(p, t, v, params) == 1.0e99);
  CHECK(murnaghan->compute_molar_heat_capacity_v(p, t, v, params) == 1.0e99);
}

TEST_CASE("Murnaghan Python reference values", "[murnaghan][eos]") {
  // burnman.eos.murnaghan pressure(), bulk_modulus() and energy() at
  // Python BurnMan historical_commits.initial_snapshots_and_tables
  // (tests/reference/burnman.json).
  struct TestData {
    double ratio, pressure, bulk_modulus, energy;
  };
  const auto data = GENERATE(
      TestData{0.4, 1335520830506.8005, 5235979155925.8418, 1756684.8764137772},
      TestData{0.8, 56555273737.019379, 375910040200.67358, 52363.221944027995},
      TestData{0.99, 1649399940.741955, 167267719774.81943, 91.95099307144784},
      TestData{1.0, 0.0, 161000000000.0, 0.0},
      TestData{1.2, -21177318928.78894, 80526188070.602036,
               27245.829388748101});
  const double t = GENERATE(300.0, 2000.0);
  eos::Murnaghan murnaghan;
  auto params = murnaghan_params();
  murnaghan.validate_parameters(params);
  const double v = *params.V_0 * data.ratio;
  CAPTURE(data.ratio, t);
  CHECK_THAT(murnaghan.compute_pressure(t, v, params),
             WithinRel(data.pressure, 1.0e-12) || WithinAbs(0.0, 1.0e-4));
  CHECK_THAT(murnaghan.compute_volume(data.pressure, t, params),
             WithinRel(v, 1.0e-12));
  CHECK_THAT(murnaghan.compute_isothermal_bulk_modulus_reuss(data.pressure, t,
                                                             v, params),
             WithinRel(data.bulk_modulus, 1.0e-12));
  CHECK_THAT(
      murnaghan.compute_molar_internal_energy(data.pressure, t, v, params),
      WithinRel(data.energy, 1.0e-11) || WithinAbs(0.0, 1.0e-8));
}

TEST_CASE("Murnaghan thermodynamic derivatives and limiting Kprime",
          "[murnaghan][eos]") {
  eos::Murnaghan murnaghan;
  auto params = murnaghan_params();
  params.Kprime_0 =
      GENERATE(-1.0, 0.0, 1.0e-12, 1.0 - 1.0e-12, 1.0, 1.0 + 1.0e-12, 3.8);
  params.P_0 = GENERATE(0.0, 2.5e9);
  params.E_0 = -125000.0;
  murnaghan.validate_parameters(params);
  const double v = *params.V_0 * GENERATE(0.65, 0.95, 1.2);
  const double t = 300.0;
  const double p = murnaghan.compute_pressure(t, v, params);
  const double dv = v * 1.0e-5;
  CAPTURE(*params.Kprime_0, *params.P_0, v);
  CHECK_THAT(murnaghan.compute_volume(p, t, params), WithinRel(v, 1.0e-12));
  const double dEdV =
      (murnaghan.compute_molar_internal_energy(p, t, v + dv, params) -
       murnaghan.compute_molar_internal_energy(p, t, v - dv, params)) /
      (2.0 * dv);
  CHECK_THAT(-dEdV, WithinRel(p, 2.0e-8));
  const double dPdV = (murnaghan.compute_pressure(t, v + dv, params) -
                       murnaghan.compute_pressure(t, v - dv, params)) /
                      (2.0 * dv);
  CHECK_THAT(murnaghan.compute_isothermal_bulk_modulus_reuss(p, t, v, params),
             WithinRel(-v * dPdV, 2.0e-8));
  const auto gibbs = [&](double pressure) {
    return murnaghan.compute_gibbs_free_energy(
        pressure, t, murnaghan.compute_volume(pressure, t, params), params);
  };
  const double dp = 1.0e5;
  CHECK_THAT((gibbs(p + dp) - gibbs(p - dp)) / (2.0 * dp),
             WithinRel(v, 2.0e-8));
}

TEST_CASE("Murnaghan energy near the reference volume", "[murnaghan][eos]") {
  eos::Murnaghan murnaghan;
  auto params = murnaghan_params();
  params.Kprime_0 = GENERATE(0.0, 1.0, 3.8);
  murnaghan.validate_parameters(params);
  const double v = *params.V_0 * GENERATE(1.0 - 1.0e-8, 1.0 + 1.0e-8);
  const double dv = v - *params.V_0;
  const double expected = 0.5 * *params.K_0 / *params.V_0 * dv * dv;
  CHECK_THAT(murnaghan.compute_molar_internal_energy(0.0, 300.0, v, params),
             WithinRel(expected, 1.0e-7));
}

TEST_CASE("Murnaghan rejects states outside its domain", "[murnaghan][eos]") {
  eos::Murnaghan murnaghan;
  auto params = murnaghan_params();
  params.K_0 = 160.0e9;
  params.Kprime_0 = 4.0;
  murnaghan.validate_parameters(params);
  const double p = GENERATE(-40.0e9, -50.0e9);
  CHECK_THROWS_AS(murnaghan.compute_volume(p, 300.0, params),
                  std::invalid_argument);
  const double v = GENERATE(0.0, -1.0e-5);
  CHECK_THROWS_AS(murnaghan.compute_pressure(300.0, v, params),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      murnaghan.compute_molar_internal_energy(0.0, 300.0, v, params),
      std::invalid_argument);
}
