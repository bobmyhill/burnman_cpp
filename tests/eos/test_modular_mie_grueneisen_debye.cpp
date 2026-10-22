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

#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/mie_grueneisen_debye.hpp"
#include "burnman/eos/modular_mie_grueneisen_debye.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman;
using namespace Catch::Matchers;
namespace dt = burnman::eos::debye_temperature_models;
namespace ap = burnman::eos::anharmonic_prefactor_models;
namespace at = burnman::eos::anharmonic_thermal_models;
namespace {
types::MineralParams parameters() {
  types::MineralParams p;
  p.equation_of_state = types::EOSType::ModularMGD;
  p.reference_eos = eos::make_eos(types::EOSType::BM3);
  p.debye_temperature_model = std::make_shared<dt::SLB>();
  p.V_0 = 4.3603e-5;
  p.K_0 = 1.279555e11;
  p.Kprime_0 = 4.21796;
  p.T_0 = 298.15;
  p.P_0 = 1.e5;
  p.F_0 = -2055403.;
  p.napfu = 7.;
  p.molar_mass = .1406931;
  p.debye_0 = 1000.;
  p.grueneisen_0 = 1.2;
  p.q_0 = 1.1;
  p.G_0 = 0.;
  p.Gprime_0 = 0.;
  p.Kprime_inf = 2.;
  p.Kdprime_0 = -4.e-11;
  p.c_1 = .2;
  p.c_2 = .1;
  p.q_1 = 1.;
  p.q_2 = 2.;
  return p;
}
} // namespace
TEST_CASE("Modular MGD validates its components", "[modular_mgd][eos]") {
  auto p = parameters();
  eos::ModularMGD model;
  SECTION("Reference EOS is required") {
    p.reference_eos.reset();
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Debye model is required") {
    p.debye_temperature_model.reset();
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Reference temperature is required") {
    p.T_0.reset();
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Debye model validates its parameters") {
    p.q_0.reset();
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Electronic parameters must be complete") {
    p.bel_0 = .005;
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Anharmonic model needs a prefactor") {
    p.anharmonic_thermal_model = std::make_shared<at::Pade>();
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Recursive reference is rejected") {
    p.reference_eos = eos::make_eos(types::EOSType::ModularMGD);
    REQUIRE_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Factories") {
    REQUIRE(dynamic_cast<eos::ModularMGD *>(
                eos::make_eos(types::EOSType::ModularMGD).get()) != nullptr);
    REQUIRE(
        dynamic_cast<eos::ModularMGDWithAnharmonicity *>(
            eos::make_eos(types::EOSType::ModularMGDWithAnharmonicity).get()) !=
        nullptr);
  }
}

TEST_CASE("Modular MGD Python snapshots", "[modular_mgd][eos]") {
  // Python BurnMan e2cfac31052273b3fa9d5ce905978e97723005e9.
  // ModularMGD at T=2000 K and V/V_0=0.8, parameters() above.
  auto p = parameters();
  double expected_p, expected_k, expected_f, expected_s, expected_cv,
      expected_alpha;
  SECTION("BM3 with SLB Debye") {
    expected_p = 53328936217.04308;
    expected_k = 303971835297.05597;
    expected_f = -2241479.9491308597;
    expected_s = 313.3943585576211;
    expected_cv = 171.11548080036843;
    expected_alpha = 1.5971133874115268e-5;
  }
  SECTION("Vinet with simple power-law gamma") {
    p.reference_eos = eos::make_eos(types::EOSType::Vinet);
    p.debye_temperature_model = std::make_shared<dt::PowerLawGammaSimple>();
    expected_p = 52254458877.70522;
    expected_k = 290254427718.3197;
    expected_f = -2243889.110791036;
    expected_s = 314.11511723369915;
    expected_cv = 171.1443249257958;
    expected_alpha = 1.5869290966747644e-5;
  }
  SECTION("SPOCK with two power laws") {
    p.reference_eos = eos::make_eos(types::EOSType::SPOCK);
    p.debye_temperature_model = std::make_shared<dt::PowerLawGamma>();
    expected_p = 53792019771.07384;
    expected_k = 298685945384.632;
    expected_f = -2237873.884818332;
    expected_s = 310.4561439890685;
    expected_cv = 170.99537106965138;
    expected_alpha = 1.8447160225189827e-5;
  }
  eos::ModularMGD e;
  e.validate_parameters(p);
  const double t = 2000., v = .8 * *p.V_0;
  const double pressure = e.compute_pressure(t, v, p);
  CHECK_THAT(pressure, WithinRel(expected_p, 2.e-12));
  CHECK_THAT(e.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(expected_k, 2.e-12));
  CHECK_THAT(e.compute_helmholtz_free_energy(pressure, t, v, p),
             WithinRel(expected_f, 2.e-12));
  CHECK_THAT(e.compute_entropy(pressure, t, v, p),
             WithinRel(expected_s, 2.e-12));
  CHECK_THAT(e.compute_molar_heat_capacity_v(pressure, t, v, p),
             WithinRel(expected_cv, 2.e-12));
  CHECK_THAT(e.compute_thermal_expansivity(pressure, t, v, p),
             WithinRel(expected_alpha, 2.e-12));
  CHECK_THAT(e.compute_volume(pressure, t, p), WithinRel(v, 2.e-13));
}

TEST_CASE("Modular MGD matches classic MGD for a power-law Debye model",
          "[modular_mgd][eos]") {
  auto p = parameters();
  p.P_0 = 0.;
  p.debye_temperature_model = std::make_shared<dt::PowerLawGammaSimple>();
  eos::ModularMGD modular;
  eos::MGD3 classic;
  modular.validate_parameters(p);
  auto classic_params = p;
  classic.validate_parameters(classic_params);
  const double t = GENERATE(0., 300., 2000.), ratio = GENERATE(.6, .9, 1.);
  const double v = ratio * *p.V_0, pressure = modular.compute_pressure(t, v, p);
  CHECK_THAT(
      pressure,
      WithinRel(classic.compute_pressure(t, v, classic_params), 2.e-12) ||
          WithinAbs(classic.compute_pressure(t, v, classic_params), 1.e-4));
  CHECK_THAT(modular.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(classic.compute_isothermal_bulk_modulus_reuss(
                           pressure, t, v, classic_params),
                       2.e-12));
  CHECK_THAT(modular.compute_entropy(pressure, t, v, p),
             WithinRel(classic.compute_entropy(pressure, t, v, classic_params),
                       2.e-12) ||
                 WithinAbs(0., 1.e-15));
  CHECK_THAT(modular.compute_helmholtz_free_energy(pressure, t, v, p),
             WithinRel(classic.compute_helmholtz_free_energy(pressure, t, v,
                                                             classic_params),
                       2.e-12));
}

TEST_CASE("Modular MGD obeys Helmholtz derivatives with optional contributions",
          "[modular_mgd][eos]") {
  auto p = parameters();
  p.P_0 = 0.;
  const int contribution = GENERATE(0, 1, 2, 3);
  if (contribution > 0) {
    p.bel_0 = .005;
    p.gel = 1.5;
  }
  if (contribution > 1) {
    p.a_anh = -.2;
    p.m_anh = 3.;
    p.b_anh = .2;
    p.c_anh = 2.;
    p.mu_anh = .2;
    p.sigma_anh = 1.2;
    if (contribution == 2) {
      p.anharmonic_prefactor_model = std::make_shared<ap::PowerLaw>();
      p.anharmonic_thermal_model = std::make_shared<at::Pade>();
    } else {
      p.anharmonic_prefactor_model = std::make_shared<ap::Sigmoid>();
      p.anharmonic_thermal_model = std::make_shared<at::LogNormal>();
    }
  }
  eos::ModularMGD e;
  e.validate_parameters(p);
  const double t = 2000., v = .8 * *p.V_0, dv = 1.e-5 * *p.V_0, dtemp = .02;
  const double pressure = e.compute_pressure(t, v, p);
  const auto f = [&](double temperature, double volume) {
    return e.compute_helmholtz_free_energy(pressure, temperature, volume, p);
  };
  CHECK_THAT(-(f(t, v + dv) - f(t, v - dv)) / (2. * dv),
             WithinRel(pressure, 2.e-8));
  CHECK_THAT((f(t - dtemp, v) - f(t + dtemp, v)) / (2. * dtemp),
             WithinRel(e.compute_entropy(pressure, t, v, p), 2.e-8));
  const double k = e.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p);
  CHECK_THAT(-v *
                 (e.compute_pressure(t, v + dv, p) -
                  e.compute_pressure(t, v - dv, p)) /
                 (2. * dv),
             WithinRel(k, 2.e-8));
  const double alpha = e.compute_thermal_expansivity(pressure, t, v, p);
  CHECK_THAT((e.compute_pressure(t + dtemp, v, p) -
              e.compute_pressure(t - dtemp, v, p)) /
                 (2. * dtemp),
             WithinRel(alpha * k, 2.e-8));
  const double cv = e.compute_molar_heat_capacity_v(pressure, t, v, p);
  CHECK_THAT(t *
                 (e.compute_entropy(pressure, t + dtemp, v, p) -
                  e.compute_entropy(pressure, t - dtemp, v, p)) /
                 (2. * dtemp),
             WithinRel(cv, 2.e-8));
  CHECK_THAT(
      e.compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
      WithinRel(k * e.compute_molar_heat_capacity_p(pressure, t, v, p) / cv,
                2.e-13));
  CHECK_THAT(e.compute_volume(pressure, t, p), WithinRel(v, 2.e-13));
}

TEST_CASE("Debye model derivatives and constant-gamma limits",
          "[modular_mgd][eos]") {
  auto p = parameters();
  std::shared_ptr<dt::DebyeTemperatureModelBase> model;
  const int type = GENERATE(0, 1, 2);
  if (type == 0)
    model = std::make_shared<dt::SLB>();
  else if (type == 1)
    model = std::make_shared<dt::PowerLawGammaSimple>();
  else
    model = std::make_shared<dt::PowerLawGamma>();
  model->validate_parameters(p);
  const double x = .7, h = 1.e-6;
  CHECK_THAT((model->value(x + h, p) - model->value(x - h, p)) / (2. * h),
             WithinRel(model->dVrel(x, p), 2.e-8));
  CHECK_THAT((model->dVrel(x + h, p) - model->dVrel(x - h, p)) / (2. * h),
             WithinRel(model->dVrel2(x, p), 2.e-8));
  p.q_0 = 0.;
  p.q_1 = 0.;
  p.q_2 = 0.;
  if (type != 0) {
    CHECK_THAT(model->value(x, p),
               WithinRel(*p.debye_0 * std::pow(x, -*p.grueneisen_0), 2.e-13));
    CHECK_THAT(model->gamma(x, p), WithinRel(*p.grueneisen_0, 2.e-13));
  }
}

TEST_CASE("Modular MGD zero temperature and rescaled anharmonicity",
          "[modular_mgd][eos]") {
  auto p = parameters();
  p.P_0 = 0.;
  eos::ModularMGD e;
  e.validate_parameters(p);
  const double v = .9 * *p.V_0, pressure = e.compute_pressure(0., v, p);
  CHECK(e.compute_entropy(pressure, 0., v, p) == 0.);
  CHECK(e.compute_molar_heat_capacity_v(pressure, 0., v, p) == 0.);
  CHECK(e.compute_thermal_expansivity(pressure, 0., v, p) == 0.);
  CHECK_THAT(e.compute_grueneisen_parameter(pressure, 0., v, p),
             WithinRel(p.debye_temperature_model->gamma(.9, p), 2.e-13));
  CHECK_THAT(
      e.compute_isentropic_bulk_modulus_reuss(pressure, 0., v, p),
      WithinRel(e.compute_isothermal_bulk_modulus_reuss(pressure, 0., v, p)));
  CHECK_THAT(e.compute_isentropic_bulk_modulus_reuss(pressure, 1.e-20, v, p),
             WithinRel(e.compute_isothermal_bulk_modulus_reuss(pressure, 1.e-20,
                                                               v, p)));
  eos::ModularMGDWithAnharmonicity a;
  REQUIRE_THROWS_AS(a.validate_parameters(p), std::invalid_argument);
  p.c_anh = 0.;
  a.validate_parameters(p);
  CHECK(a.compute_pressure(2000., v, p) == e.compute_pressure(2000., v, p));
  CHECK(a.compute_entropy(pressure, 2000., v, p) ==
        e.compute_entropy(pressure, 2000., v, p));
  p.c_anh = .05;
  CHECK(std::isfinite(a.compute_pressure(2000., v, p)));
  CHECK(a.compute_pressure(2000., v, p) != e.compute_pressure(2000., v, p));
  CHECK(std::isfinite(
      a.compute_isentropic_bulk_modulus_reuss(pressure, 0., v, p)));
  CHECK_THAT(a.compute_pressure(*p.T_0, v, p),
             WithinRel(e.compute_pressure(*p.T_0, v, p), 2.e-13));
  const double target = a.compute_pressure(2000., v, p);
  CHECK_THAT(a.compute_volume(target, 2000., p), WithinRel(v, 2.e-12));
}

TEST_CASE("Modular MGD reports invalid volumes and missing stable roots",
          "[modular_mgd][eos]") {
  auto p = parameters();
  eos::ModularMGD e;
  e.validate_parameters(p);
  CHECK_THROWS_AS(e.compute_pressure(300., 0., p), std::domain_error);
  CHECK_THROWS_AS(e.compute_pressure(-1., *p.V_0, p), std::domain_error);
  CHECK_THROWS_AS(e.compute_volume(-1.e14, 300., p), std::domain_error);
}

TEST_CASE("Modular MGD evaluates pressure-based reference moduli consistently",
          "[modular_mgd][eos]") {
  auto p = parameters();
  p.reference_eos = eos::make_eos(types::EOSType::MT);
  p.P_0 = 2.e9;
  eos::ModularMGD e;
  e.validate_parameters(p);
  const double v = .8 * *p.V_0, t = *p.T_0, dv = 1.e-5 * *p.V_0;
  const double pressure = e.compute_pressure(t, v, p);
  // Native MT uses pressure to compute K_T. Passing zero as a placeholder
  // would return the modulus at the wrong point on the reference isotherm.
  CHECK_THAT(e.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(-v *
                           (e.compute_pressure(t, v + dv, p) -
                            e.compute_pressure(t, v - dv, p)) /
                           (2. * dv),
                       2.e-8));
  CHECK_THAT(e.compute_helmholtz_free_energy(pressure, t, *p.V_0, p),
             WithinRel(*p.F_0, 2.e-13));
}
