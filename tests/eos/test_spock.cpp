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

#include "burnman/core/mineral.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/spock.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman;
using namespace Catch::Matchers;

namespace {
types::MineralParams spock_params() {
  types::MineralParams params;
  params.equation_of_state = types::EOSType::SPOCK;
  params.V_0 = 11.24e-6;
  params.K_0 = 161.0e9;
  params.Kprime_0 = 3.8;
  params.Kdprime_0 = -0.02e-9;
  params.Kprime_inf = 5.0 / 3.0;
  params.molar_mass = 0.0403;
  return params;
}
} // namespace

TEST_CASE("SPOCK parameter validation", "[spock][eos]") {
  eos::SPOCK spock;
  auto params = spock_params();
  SECTION("Required parameters") {
    auto missing = GENERATE(
        &types::MineralParams::V_0, &types::MineralParams::K_0,
        &types::MineralParams::Kprime_0, &types::MineralParams::Kdprime_0,
        &types::MineralParams::Kprime_inf);
    (params.*missing).reset();
    REQUIRE_THROWS_AS(spock.validate_parameters(params), std::invalid_argument);
  }
  SECTION("Python defaults") {
    spock.validate_parameters(params);
    REQUIRE(*params.F_0 == 0.0);
    REQUIRE(*params.P_0 == 1.0e5);
    REQUIRE_FALSE(params.E_0.has_value());
  }
  SECTION("Explicit reference values are preserved") {
    params.F_0 = -125000.0;
    params.P_0 = 2.5e9;
    spock.validate_parameters(params);
    REQUIRE(*params.F_0 == -125000.0);
    REQUIRE(*params.P_0 == 2.5e9);
  }
  SECTION("Internal energy reference is rejected") {
    params.E_0 = 0.0;
    REQUIRE_THROWS_AS(spock.validate_parameters(params), std::invalid_argument);
  }
}

TEST_CASE("SPOCK reference state and isothermal conventions", "[spock][eos]") {
  auto spock = eos::make_eos(types::EOSType::SPOCK);
  REQUIRE(dynamic_cast<eos::SPOCK *>(spock.get()) != nullptr);
  auto params = spock_params();
  params.F_0 = -125000.0;
  spock->validate_parameters(params);
  const double p = *params.P_0;
  const double v = *params.V_0;
  const double t = GENERATE(0.0, 300.0, 2000.0);
  CHECK(spock->compute_pressure(t, v, params) == p);
  CHECK_THAT(spock->compute_volume(p, t, params), WithinRel(v, 2.0e-13));
  CHECK(spock->compute_isothermal_bulk_modulus_reuss(p, t, v, params) ==
        *params.K_0);
  CHECK(spock->compute_isentropic_bulk_modulus_reuss(p, t, v, params) ==
        *params.K_0);
  CHECK(spock->compute_helmholtz_free_energy(p, t, v, params) == *params.F_0);
  CHECK(spock->compute_molar_internal_energy(p, t, v, params) == *params.F_0);
  CHECK(spock->compute_gibbs_free_energy(p, t, v, params) ==
        *params.F_0 + p * v);
  CHECK(spock->compute_enthalpy(p, t, v, params) == *params.F_0 + p * v);
  CHECK(spock->compute_grueneisen_parameter(p, t, v, params) == 0.0);
  CHECK(spock->compute_entropy(p, t, v, params) == 0.0);
  CHECK(spock->compute_thermal_expansivity(p, t, v, params) == 0.0);
  CHECK(spock->compute_molar_heat_capacity_p(p, t, v, params) == 1.0e-99);
  CHECK(spock->compute_molar_heat_capacity_v(p, t, v, params) == 1.0e-99);
  CHECK(spock->compute_shear_modulus(p, t, v, params) == 1.0e99);
}

TEST_CASE("SPOCK Python reference values", "[spock][eos]") {
  // Python BurnMan historical_commits.initial_snapshots_and_tables
  // (tests/reference/burnman.json), burnman.eos.spock.SPOCK: pressure,
  // isothermal_bulk_modulus_reuss, _molar_helmholtz_energy and gibbs_energy at
  // T=300 K. Both parameter sets use F_0=-125000 J/mol and P_0=2.5e9 Pa.
  struct TestData {
    double ratio, pressure, bulk_modulus, helmholtz, gibbs;
  };
  auto params = spock_params();
  params.F_0 = -125000.0;
  params.P_0 = 2.5e9;
  const double t = GENERATE(300.0, 2000.0);
  TestData data{};
  SECTION("MgO-like parameters") {
    data = GENERATE(TestData{1.2, -18346401827.64814, 75928925874.401947,
                             -92349.651144379881, -339805.91899569798},
                    TestData{0.99, 4149310021.1486011, 167240657097.53671,
                             -125189.05151355328, -79017.189322220103},
                    TestData{0.95, 11604756396.181242, 194842239481.59894,
                             -123949.69681873947, -34.108020316169132},
                    TestData{0.8, 57407194374.313217, 349839259642.87732,
                             -79034.781994412391, 437170.70981941209},
                    TestData{0.4, 778567749973.11621, 2137602925512.9077,
                             1150226.8421472919, 4650667.446026423},
                    TestData{0.1, 15817254126411.432, 29398353432917.707,
                             14443426.613608031, 32222020.251694486});
  }
  SECTION("Iron-like parameters") {
    params.V_0 = 6.75e-6;
    params.K_0 = 163.4e9;
    params.Kprime_0 = 5.38;
    params.Kdprime_0 = -0.05e-9;
    params.Kprime_inf = 2.0;
    data = GENERATE(TestData{1.2, -15779751555.180702, 52308920413.103416,
                             -106654.64547194974, -234470.63306891342},
                    TestData{0.99, 4187200461.726141, 172407814969.5896,
                             -125112.41159609857, -97131.444510613626},
                    TestData{0.95, 12116174904.476889, 213117191333.97147,
                             -124306.57776323399, -46611.606188275953},
                    TestData{0.8, 67872197725.107269, 457343877599.01453,
                             -93208.130277971766, 273301.73743760749},
                    TestData{0.4, 1200101990896.8479, 3549128960512.7041,
                             973002.73349194648, 4213278.1089134356},
                    TestData{0.1, 31560499446886.777, 65798711616799.391,
                             15387957.107870169, 36691294.234518744});
  }
  eos::SPOCK spock;
  spock.validate_parameters(params);
  const double v = data.ratio * *params.V_0;
  CAPTURE(data.ratio, t);
  CHECK_THAT(spock.compute_pressure(t, v, params),
             WithinRel(data.pressure, 2.0e-12));
  CHECK_THAT(eos::SPOCK::compute_spock(data.ratio, params),
             WithinRel(data.pressure, 2.0e-12));
  CHECK_THAT(
      spock.compute_isothermal_bulk_modulus_reuss(data.pressure, t, v, params),
      WithinRel(data.bulk_modulus, 2.0e-13));
  CHECK_THAT(spock.compute_helmholtz_free_energy(data.pressure, t, v, params),
             WithinRel(data.helmholtz, 2.0e-12));
  // The small Gibbs value in one row subtracts energies of order 1e5 J/mol.
  CHECK_THAT(spock.compute_gibbs_free_energy(data.pressure, t, v, params),
             WithinRel(data.gibbs, 2.0e-12) || WithinAbs(data.gibbs, 2.0e-8));
  CHECK_THAT(
      spock.compute_volume(spock.compute_pressure(t, v, params), t, params),
      WithinRel(v, 2.0e-13));
}

TEST_CASE("SPOCK pressure and energy derivatives", "[spock][eos]") {
  eos::SPOCK spock;
  auto params = spock_params();
  params.P_0 = 0.0;
  params.F_0 = 0.0;
  spock.validate_parameters(params);
  const double ratio = GENERATE(0.4, 0.8, 0.99, 1.0, 1.2);
  const double v = ratio * *params.V_0;
  const double h = 1.0e-6 * *params.V_0;
  const double p = spock.compute_pressure(300.0, v, params);
  const double p_plus = spock.compute_pressure(300.0, v + h, params);
  const double p_minus = spock.compute_pressure(300.0, v - h, params);
  const double k = spock.compute_spock_bulk_modulus(v, params);
  CHECK_THAT(-v * (p_plus - p_minus) / (2.0 * h), WithinRel(k, 2.0e-8));
  // A wider five-point stencil limits roundoff in the energy differences.
  const double energy_h = 1.0e-4 * *params.V_0;
  const auto energy = [&](double offset) {
    return spock.compute_helmholtz_free_energy(p, 300.0, v + offset, params);
  };
  const double energy_derivative =
      (energy(-2.0 * energy_h) - 8.0 * energy(-energy_h) +
       8.0 * energy(energy_h) - energy(2.0 * energy_h)) /
      (12.0 * energy_h);
  CHECK_THAT(-energy_derivative, WithinRel(p, 2.0e-8) || WithinAbs(p, 1.0));
  const double kp = (spock.compute_spock_bulk_modulus(v + h, params) -
                     spock.compute_spock_bulk_modulus(v - h, params)) /
                    (p_plus - p_minus);
  const double a = -*params.Kdprime_0 * *params.K_0 /
                   (*params.Kprime_0 - *params.Kprime_inf);
  const double expected_kp =
      *params.Kprime_inf +
      (*params.Kprime_0 - *params.Kprime_inf) * std::pow(ratio, a);
  CHECK_THAT(kp, WithinRel(expected_kp, 2.0e-8));
}

TEST_CASE("SPOCK retains small strains near the reference volume",
          "[spock][eos]") {
  eos::SPOCK spock;
  auto params = spock_params();
  params.P_0 = 0.0;
  params.F_0 = 0.0;
  spock.validate_parameters(params);
  const double ratio =
      GENERATE(1.0 - 1.0e-8, 1.0 + 1.0e-8, 1.0 - 1.0e-12, 1.0 + 1.0e-12);
  const double v = ratio * *params.V_0;
  const double strain = v / *params.V_0 - 1.0;
  const double p = spock.compute_pressure(300.0, v, params);
  CHECK_THAT(p, WithinRel(-*params.K_0 * strain, 1.0e-7));
  const double f = spock.compute_helmholtz_free_energy(p, 300.0, v, params);
  CHECK_THAT(
      f, WithinRel(0.5 * *params.K_0 * *params.V_0 * strain * strain, 1.0e-7));
  CHECK_THAT(spock.compute_volume(p, 300.0, params), WithinRel(v, 2.0e-13));
}

TEST_CASE("SPOCK handles negative integer incomplete-gamma orders",
          "[spock][eos]") {
  eos::SPOCK spock;
  auto params = spock_params();
  params.V_0 = 1.0e-5;
  params.K_0 = 100.0e9;
  params.Kprime_0 = 4.0;
  params.Kdprime_0 = -0.04e-9;
  params.Kprime_inf = 2.0;
  params.P_0 = 0.0;
  params.F_0 = 0.0;
  spock.validate_parameters(params);
  // a=2, b=1, c=2 gives gamma orders -1 and -1/2. Reference values
  // independently integrate K/K_0 = exp(1-exp(2u)-2u), u=ln(V/V_0),
  // using scipy.integrate.quad with epsabs=epsrel=1e-13.
  struct TestData {
    double ratio, pressure, helmholtz;
  };
  const auto data =
      GENERATE(TestData{0.99, 1025439493.9178277, 50.84428961864826},
               TestData{0.8, 34759665301.44075, 28891.546396986967},
               TestData{0.4, 512153718562.55853, 746152.459706438},
               TestData{1.2, -12690470558.954823, 14744.958140654742});
  const double v = data.ratio * *params.V_0;
  CAPTURE(data.ratio);
  CHECK_THAT(spock.compute_pressure(300.0, v, params),
             WithinRel(data.pressure, 2.0e-12));
  CHECK_THAT(
      spock.compute_helmholtz_free_energy(data.pressure, 300.0, v, params),
      WithinRel(data.helmholtz, 2.0e-12));
}

TEST_CASE("SPOCK integrates into Mineral", "[spock][eos][mineral]") {
  Mineral mineral;
  mineral.params = spock_params();
  mineral.params.F_0 = -125000.0;
  mineral.set_method(types::EOSType::Auto);
  mineral.set_state(25.0e9, 1000.0);
  const double v = mineral.get_molar_volume();
  CHECK(v < *mineral.params.V_0);
  CHECK_THAT(eos::SPOCK::compute_spock(v / *mineral.params.V_0, mineral.params),
             WithinRel(25.0e9, 2.0e-13));
  CHECK_THAT(mineral.get_density(), WithinRel(*mineral.params.molar_mass / v));
  CHECK_THAT(mineral.get_molar_gibbs(),
             WithinRel(mineral.get_molar_helmholtz() + 25.0e9 * v));
  CHECK(mineral.get_molar_heat_capacity_v() == 1.0e-99);
  CHECK(mineral.get_molar_entropy() == 0.0);
  CHECK(mineral.get_grueneisen_parameter() == 0.0);
  CHECK_THAT(mineral.get_isentropic_bulk_modulus_reuss(),
             WithinRel(mineral.get_isothermal_bulk_modulus_reuss()));
}
