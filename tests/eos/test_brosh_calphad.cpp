// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/brosh_calphad.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/minerals/SE_2015.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
using Factory = std::shared_ptr<Mineral> (*)();
constexpr Factory factories[]{
    minerals::SE_2015::bcc_iron, minerals::SE_2015::fcc_iron,
    minerals::SE_2015::hcp_iron, minerals::SE_2015::liquid_iron};
double qsm(double z, bool upper) {
  const double lz = std::log(z);
  return (upper ? .025692 : .02713) *
         std::exp((upper ? .97914 : .97626) * lz -
                  (upper ? .057741 : .057848) * lz * lz) *
         1.e-6;
}
} // namespace

TEST_CASE("Brosh factory and Appendix A constraints", "[brosh][eos]") {
  auto model = eos::make_eos(types::EOSType::BroshCalphad);
  REQUIRE(dynamic_cast<eos::BroshCalphad *>(model.get()));
  for (auto factory : factories) {
    auto mineral = factory();
    const auto &p = mineral->params;
    CAPTURE(*p.name);
    CHECK_THAT(model->compute_volume(0., 0., p), WithinRel(*p.V_0, 1.e-13));
    CHECK_THAT(model->compute_isothermal_bulk_modulus_reuss(0., 0., *p.V_0, p),
               WithinRel(*p.K_0, 1.e-13));
    const auto &c = *p.brosh_c;
    CHECK_THAT(6. * c[1] + 8. * c[2] + 9. * c[3], WithinRel(8., 1.e-13));
    CHECK_THAT(model->compute_volume(300.e12, 0., p),
               WithinRel(qsm(26., false), 1.e-13));
    CHECK_THAT(model->compute_volume(330.e12, 0., p),
               WithinRel(qsm(26., true), 1.e-13));
    CHECK(model->compute_shear_modulus(0., 300., *p.V_0, p) == 0.);
  }
  // The high-pressure target is additive for a compound and fractional formula
  // units.
  auto p = minerals::SE_2015::hcp_iron()->params;
  p.formula = types::FormulaMap{{"Mg", .5}, {"O", .5}};
  p.napfu = 1.;
  p.V_0 = 5.6e-6;
  p.K_0 = 166.e9;
  p.Kprime_0 = 4.15;
  p.brosh_a.reset();
  p.brosh_c.reset();
  model->validate_parameters(p);
  CHECK_THAT(model->compute_volume(300.e12, 0., p),
             WithinRel(.5 * (qsm(12., false) + qsm(8., false)), 1.e-13));
}

TEST_CASE("Brosh Gibbs and response derivatives", "[brosh][eos]") {
  const double t = GENERATE(100., 300., 2000., 8000.);
  const double pressure = GENERATE(1.e5, 100.e9);
  eos::BroshCalphad model;
  for (auto factory : factories) {
    const auto p = factory()->params;
    const double v = model.compute_volume(pressure, t, p), dt = .02, dp = 1.e6;
    CAPTURE(*p.name, t, pressure);
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
    const double kt =
        model.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p);
    const double cp = model.compute_molar_heat_capacity_p(pressure, t, v, p);
    const double cv = model.compute_molar_heat_capacity_v(pressure, t, v, p);
    CHECK_THAT((g(pressure + dp, t) - g(pressure - dp, t)) / (2. * dp),
               WithinRel(v, 2.e-7));
    CHECK_THAT(-(g(pressure, t + dt) - g(pressure, t - dt)) / (2. * dt),
               WithinRel(s(pressure, t), 2.e-7));
    CHECK_THAT(t * (s(pressure, t + dt) - s(pressure, t - dt)) / (2. * dt),
               WithinRel(cp, 2.e-7));
    CHECK_THAT((volume(pressure, t + dt) - volume(pressure, t - dt)) /
                   (2. * dt),
               WithinRel(alpha * v, 2.e-7));
    CHECK_THAT(-(s(pressure + dp, t) - s(pressure - dp, t)) / (2. * dp),
               WithinRel(alpha * v, 2.e-7));
    CHECK_THAT(-v * 2. * dp /
                   (volume(pressure + dp, t) - volume(pressure - dp, t)),
               WithinRel(kt, 2.e-7));
    CHECK_THAT(cv, WithinRel(cp - t * v * alpha * alpha * kt, 1.e-13));
    CHECK_THAT(model.compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
               WithinRel(kt * cp / cv, 1.e-13));
    CHECK_THAT(model.compute_grueneisen_parameter(pressure, t, v, p),
               WithinRel(alpha * kt * v / cv, 1.e-13));
    CHECK_THAT(model.compute_pressure(t, v, p),
               WithinRel(pressure, 1.e-10) || WithinAbs(pressure, .01));
    CHECK_THAT(model.compute_helmholtz_free_energy(pressure, t, v, p),
               WithinRel(g(pressure, t) - pressure * v, 1.e-13));
    CHECK_THAT(model.compute_enthalpy(pressure, t, v, p),
               WithinRel(g(pressure, t) + t * s(pressure, t), 1.e-13));
    CHECK_THAT(
        model.compute_molar_internal_energy(pressure, t, v, p),
        WithinRel(g(pressure, t) + t * s(pressure, t) - pressure * v, 1.e-13));
  }
}

TEST_CASE("Brosh low temperature, zero temperature and tension",
          "[brosh][eos]") {
  eos::BroshCalphad model;
  for (auto factory : factories) {
    auto mineral = factory();
    const auto &p = mineral->params;
    CAPTURE(*p.name);
    mineral->set_state(300.e9, 0.);
    CHECK(std::isfinite(mineral->get_molar_gibbs()));
    CHECK(std::isfinite(mineral->get_molar_entropy()));
    CHECK(mineral->get_molar_heat_capacity_p() == 0.);
    CHECK(mineral->get_thermal_expansivity() == 0.);
    CHECK(mineral->get_isentropic_bulk_modulus_reuss() ==
          mineral->get_isothermal_bulk_modulus_reuss());
    const double zero_gamma = mineral->get_grueneisen_parameter();
    CHECK(std::isfinite(zero_gamma));
    mineral->set_state(300.e9, 1.e-4);
    CHECK_THAT(mineral->get_grueneisen_parameter(),
               WithinRel(zero_gamma, 1.e-8));
    mineral->set_state(300.e9, 100.);
    CHECK(mineral->get_molar_heat_capacity_p() > 0.);
    const double v = model.compute_volume(-1.e8, 1000., p);
    CHECK_THAT(model.compute_pressure(1000., v, p), WithinRel(-1.e8, 1.e-10));
    const double tr = *p.T_0;
    CHECK_THAT(model.compute_molar_heat_capacity_p(300.e9, tr - 1.e-6, v, p),
               WithinRel(model.compute_molar_heat_capacity_p(300.e9, tr, v, p),
                         1.e-7));
  }
}

TEST_CASE("Brosh SGTE segment boundaries and extrapolation", "[brosh][eos]") {
  eos::BroshCalphad model;
  auto p = minerals::SE_2015::liquid_iron()->params;
  std::array<double, 13> first{}, last{};
  first[2] = -20.;
  last[0] = 10.;
  last[1] = -2.;
  last[2] = -30.;
  p.gibbs_coefficients =
      types::GibbsCoefficients{{500., first}, {12000., last}};
  model.validate_parameters(p);
  for (double t : {500., 12000., 15000.}) {
    CHECK_THAT(model.compute_gibbs_free_energy(0., t, 0., p),
               WithinRel(10. - 2. * t - 30. * t * std::log(t), 1.e-13));
    CHECK_THAT(model.compute_molar_heat_capacity_p(0., t, 0., p),
               WithinRel(30., 1.e-13));
  }
}

TEST_CASE("Brosh parameter and state validation", "[brosh][eos]") {
  eos::BroshCalphad model;
  auto p = minerals::SE_2015::hcp_iron()->params;
  SECTION("Required parameter") {
    p.theta_0.reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid thermal law") {
    p.b_0 = 1. / 3.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Empty SGTE") {
    p.gibbs_coefficients->clear();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Unsorted SGTE") {
    (*p.gibbs_coefficients)[1].first = 1.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Nonfinite SGTE") {
    (*p.gibbs_coefficients)[0].second[0] =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Unknown atomic number") {
    p.formula = types::FormulaMap{{"Xx", 1.}};
    p.brosh_c.reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Explicit c needs no formula") {
    p.formula.reset();
    CHECK_NOTHROW(model.validate_parameters(p));
  }
  SECTION("Tension cannot cross a pole in the thermal X function") {
    p.brosh_a = std::array<double, 4>{100., 100., 100., 100.};
    p.b_0 = .5;
    p.delta_0 = 0.;
    p.b_1 = .001;
    p.delta_1 = 0.;
    CHECK_THROWS_AS(model.compute_volume(-2.5 * *p.K_0, 300., p),
                    std::invalid_argument);
  }
  SECTION("Invalid states") {
    CHECK_THROWS_AS(model.compute_volume(1.e9, -1., p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_volume(-1.e15, 300., p),
                    std::invalid_argument);
    CHECK_THROWS_AS(model.compute_pressure(300., 0., p), std::invalid_argument);
    CHECK_THROWS_AS(
        model.compute_volume(std::numeric_limits<double>::quiet_NaN(), 300., p),
        std::invalid_argument);
  }
}
