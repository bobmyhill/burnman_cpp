/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
// Numerical snapshots: Python BurnMan 69e700647ae7efeed91dfbce147c036bce619516.
// Reproduce with tools/check_reference_data.py; see docs/reference_data.md.
#include "burnman/eos/components/einstein.hpp"
#include "tolerances.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace burnman;
using namespace eos::einstein;

TEST_CASE("Check zero returns in einstein model functions", "[eos][einstein]") {
  double theta_0 = 773.0;
  int napfu = 2;
  // Explicit return of 0
  CHECK(compute_thermal_energy(0, theta_0, napfu) == 0);
  CHECK(compute_molar_heat_capacity_v(0, theta_0, napfu) == 0);
  CHECK(compute_helmholtz_free_energy(0, theta_0, napfu) == 0);
  CHECK(compute_entropy(0, theta_0, napfu) == 0);
  CHECK(compute_dmolar_heat_capacity_v_dT(0, theta_0, napfu) == 0);
  double small_val = 1.e-16;
  CHECK(compute_thermal_energy(small_val, theta_0, napfu) == 0);
  CHECK(compute_molar_heat_capacity_v(small_val, theta_0, napfu) == 0);
  CHECK(compute_helmholtz_free_energy(small_val, theta_0, napfu) == 0);
  CHECK(compute_entropy(small_val, theta_0, napfu) == 0);
  CHECK(compute_dmolar_heat_capacity_v_dT(small_val, theta_0, napfu) == 0);
}

TEST_CASE("Fractional atom counts", "[eos][einstein]") {
  const std::array functions = {
      &compute_thermal_energy, &compute_molar_heat_capacity_v,
      &compute_helmholtz_free_energy, &compute_entropy,
      &compute_dmolar_heat_capacity_v_dT};
  for (auto function : functions) {
    const double per_atom = function(1000.0, 773.0, 1.0);
    for (double napfu : {0.5, 81.5}) {
      const double expected = napfu * per_atom;
      CHECK_THAT(function(1000.0, 773.0, napfu),
                 WithinRel(expected, tol_rel) || WithinAbs(expected, tol_abs));
    }
  }
}

TEST_CASE("Einstein napfu constant", "[eos][einstein]") {
  double T = 800.0;
  double theta_0 = 500.0;
  int napfu = 2;
  int natpfu_half = 1;
  double E = compute_thermal_energy(T, theta_0, napfu);
  double half_E = compute_thermal_energy(T, theta_0, natpfu_half);
  double C = compute_molar_heat_capacity_v(T, theta_0, napfu);
  double half_C = compute_molar_heat_capacity_v(T, theta_0, natpfu_half);
  double F = compute_helmholtz_free_energy(T, theta_0, napfu);
  double half_F = compute_helmholtz_free_energy(T, theta_0, natpfu_half);
  double S = compute_entropy(T, theta_0, napfu);
  double half_S = compute_entropy(T, theta_0, natpfu_half);
  double dCdT = compute_dmolar_heat_capacity_v_dT(T, theta_0, napfu);
  double half_dCdT = compute_dmolar_heat_capacity_v_dT(T, theta_0, natpfu_half);
  CHECK_THAT(E / 2.0, WithinRel(half_E, tol_rel) || WithinAbs(half_E, tol_abs));
  CHECK_THAT(C / 2.0, WithinRel(half_C, tol_rel) || WithinAbs(half_C, tol_abs));
  CHECK_THAT(F / 2.0, WithinRel(half_F, tol_rel) || WithinAbs(half_F, tol_abs));
  CHECK_THAT(S / 2.0, WithinRel(half_S, tol_rel) || WithinAbs(half_S, tol_abs));
  CHECK_THAT(dCdT / 2.0,
             WithinRel(half_dCdT, tol_rel) || WithinAbs(half_dCdT, tol_abs));
}

TEST_CASE("Einstein functions python reference values", "[eos][einstein]") {
  double T_a = 2000.0;
  double theta_a = 543.0;
  int n_a = 2;
  double T_b = 300.0;
  double theta_b = 773.0;
  int n_b = 1;
  SECTION("compute_thermal_energy") {
    double ref_a = 86841.4179357221;
    double ref_b = 1586.512654905218;
    CHECK_THAT(compute_thermal_energy(T_a, theta_a, n_a),
               WithinRel(ref_a, tol_rel) || WithinAbs(ref_a, tol_abs));
    CHECK_THAT(compute_thermal_energy(T_b, theta_b, n_b),
               WithinRel(ref_b, tol_rel) || WithinAbs(ref_b, tol_abs));
  }
  SECTION("compute_molar_heat_capacity_v") {
    double ref_a = 49.581462955163104;
    double ref_b = 14.747596514705776;
    CHECK_THAT(compute_molar_heat_capacity_v(T_a, theta_a, n_a),
               WithinRel(ref_a, tol_rel) || WithinAbs(ref_a, tol_abs));
    CHECK_THAT(compute_molar_heat_capacity_v(T_b, theta_b, n_b),
               WithinRel(ref_b, tol_rel) || WithinAbs(ref_b, tol_abs));
  }
  SECTION("compute_helmholtz_free_energy") {
    double ref_a = -143322.0806002133;
    double ref_b = -591.7003204584444;
    CHECK_THAT(compute_helmholtz_free_energy(T_a, theta_a, n_a),
               WithinRel(ref_a, tol_rel) || WithinAbs(ref_a, tol_abs));
    CHECK_THAT(compute_helmholtz_free_energy(T_b, theta_b, n_b),
               WithinRel(ref_b, tol_rel) || WithinAbs(ref_b, tol_abs));
  }
  SECTION("compute_entropy") {
    double ref_a = 115.0817492679677;
    double ref_b = 7.260709917878874;
    CHECK_THAT(compute_entropy(T_a, theta_a, n_a),
               WithinRel(ref_a, tol_rel) || WithinAbs(ref_a, tol_abs));
    CHECK_THAT(compute_entropy(T_b, theta_b, n_b),
               WithinRel(ref_b, tol_rel) || WithinAbs(ref_b, tol_abs));
  }
  SECTION("compute_dmolar_heat_capacity_v_dT") {
    double ref_a = 0.0003041899206044669;
    double ref_b = 0.049192914623578235;
    CHECK_THAT(compute_dmolar_heat_capacity_v_dT(T_a, theta_a, n_a),
               WithinRel(ref_a, tol_rel) || WithinAbs(ref_a, tol_abs));
    CHECK_THAT(compute_dmolar_heat_capacity_v_dT(T_b, theta_b, n_b),
               WithinRel(ref_b, tol_rel) || WithinAbs(ref_b, tol_abs));
  }
}
