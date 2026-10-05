/* GPL v3 or later. PS1994 H2O regressions against Python BurnMan.
 * Reference policy and thermal-reference differences:
 * tests/reference/ps1994.json and docs/ps1994_reference.md. Regenerate with
 * tools/generate_ps1994_reference.py.
 */
#include "burnman/minerals/water.hpp"
#include "ps1994_reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace Catch::Matchers;
using burnman::minerals::water_fluid;
using burnman::minerals::WaterThermalReference;

TEST_CASE("PS1994 water volumes and compressibility match Python BurnMan",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  for (const auto &state : ps1994_reference::states) {
    DYNAMIC_SECTION("P=" << state.pressure << " Pa, T=" << state.temperature
                         << " K") {
      water->set_state(state.pressure, state.temperature);
      CHECK_THAT(water->get_molar_volume(), WithinRel(state.volume, 2.e-12));
      CHECK_THAT(water->get_density(),
                 WithinRel(.01801528 / state.volume, 2.e-12));
      // The reference differentiates P(V) and the native EOS differentiates
      // P(rho), using different finite-difference steps.
      CHECK_THAT(water->get_isothermal_bulk_modulus_reuss(),
                 WithinRel(state.bulk_modulus, 2.e-8));
      CHECK(water->get_shear_modulus() == 0.);
    }
  }
}

TEST_CASE("PS1994 water pressure-dependent Gibbs energy matches Python BurnMan",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  auto reference = water_fluid(thermal_reference);
  for (const auto &state : ps1994_reference::states) {
    DYNAMIC_SECTION("P=" << state.pressure << " Pa, T=" << state.temperature
                         << " K") {
      water->set_state(state.pressure, state.temperature);
      reference->set_state(ps1994_reference::reference_pressure,
                           state.temperature);
      const double change =
          water->get_molar_gibbs() - reference->get_molar_gibbs();
      CHECK_THAT(change, WithinRel(state.delta_gibbs, 2.e-11) ||
                             WithinAbs(state.delta_gibbs, 2.e-8));
    }
  }
}

TEST_CASE("PS1994 water thermal derivatives match Python BurnMan",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  auto reference = water_fluid(thermal_reference);
  for (const auto &state : ps1994_reference::states) {
    if (state.temperature <= 500. || state.temperature >= 1700.)
      continue;
    DYNAMIC_SECTION("P=" << state.pressure << " Pa, T=" << state.temperature
                         << " K") {
      water->set_state(state.pressure, state.temperature);
      reference->set_state(ps1994_reference::reference_pressure,
                           state.temperature);
      CHECK_THAT(water->get_thermal_expansivity(),
                 WithinRel(state.expansivity, 2.e-6));
      const double entropy_change =
          water->get_molar_entropy() - reference->get_molar_entropy();
      CHECK_THAT(entropy_change, WithinRel(state.delta_entropy, 2.e-7) ||
                                     WithinAbs(state.delta_entropy, 2.e-7));
      const double cp_change = water->get_molar_heat_capacity_p() -
                               reference->get_molar_heat_capacity_p();
      // Native Cp extrapolates Gibbs-energy second differences at 0.1 and
      // 0.05 K. Python uses a 0.01 K entropy difference and thermoelastic
      // identities. Allow their discretization and roundoff errors.
      CHECK_THAT(cp_change, WithinRel(state.delta_cp, 2.e-5) ||
                                WithinAbs(state.delta_cp, 2.e-3));
    }
  }
}

TEST_CASE("PS1994 water selects the stable liquid or vapour density root",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  int compared = 0;
  bool liquid = false, vapour = false;
  for (const auto &state : ps1994_reference::states) {
    if (state.stable_roots < 2)
      continue;
    INFO("P=" << state.pressure << " Pa, T=" << state.temperature << " K");
    water->set_state(state.pressure, state.temperature);
    CHECK_THAT(water->get_molar_volume(), WithinRel(state.volume, 2.e-12));
    // Both branches occur among the independently minimized Python roots.
    liquid = liquid || state.volume < 1.e-4;
    vapour = vapour || state.volume > 1.e-4;
    ++compared;
  }
  CHECK(compared >= 4);
  CHECK(liquid);
  CHECK(vapour);
}

TEST_CASE("PS1994 water rejects states outside its documented domain",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double pressure : {0., -1., std::nextafter(5.e9, inf), inf, nan}) {
    INFO("P=" << pressure);
    water->set_state(pressure, 1000.);
    CHECK_THROWS_AS(water->get_molar_volume(), std::invalid_argument);
    CHECK_THROWS_AS(water->get_molar_gibbs(), std::invalid_argument);
  }
  for (double temperature :
       {std::nextafter(500., 0.), std::nextafter(1700., inf), -1., inf, nan}) {
    INFO("T=" << temperature);
    water->set_state(1.e8, temperature);
    CHECK_THROWS_AS(water->get_molar_volume(), std::invalid_argument);
    CHECK_THROWS_AS(water->get_molar_gibbs(), std::invalid_argument);
  }
}

TEST_CASE("PS1994 water includes temperature endpoints but excludes derivative "
          "endpoints",
          "[eos][water][ps1994]") {
  const auto thermal_reference = GENERATE(
      WaterThermalReference::HollandPowell2011, WaterThermalReference::NIST);
  auto water = water_fluid(thermal_reference);
  for (double temperature : {500., 1700.}) {
    INFO("T=" << temperature);
    water->set_state(5.e9, temperature);
    CHECK(std::isfinite(water->get_molar_volume()));
    CHECK(std::isfinite(water->get_molar_gibbs()));
    CHECK(water->get_isothermal_bulk_modulus_reuss() > 0.);
    CHECK_THROWS_AS(water->get_molar_entropy(), std::invalid_argument);
    CHECK_THROWS_AS(water->get_molar_heat_capacity_p(), std::invalid_argument);
    CHECK_THROWS_AS(water->get_thermal_expansivity(), std::invalid_argument);
  }
}
