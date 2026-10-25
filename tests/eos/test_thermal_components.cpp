// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/components/anharmonic_debye.hpp"
#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/debye.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/eos/components/einstein.hpp"
#include "burnman/utils/constants.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
using Function = double (*)(double, double, double);
struct Lattice {
  Function f, s, cv, dcv, df, ddf, ds;
};
const std::array<Lattice, 2> lattices{
    {{eos::debye::compute_helmholtz_free_energy, eos::debye::compute_entropy,
      eos::debye::compute_molar_heat_capacity_v,
      eos::debye::compute_dmolar_heat_capacity_v_dT,
      eos::debye::compute_dhelmholtz_dTheta,
      eos::debye::compute_d2helmholtz_dTheta2,
      eos::debye::compute_dentropy_dTheta},
     {eos::einstein::compute_helmholtz_free_energy,
      eos::einstein::compute_entropy,
      eos::einstein::compute_molar_heat_capacity_v,
      eos::einstein::compute_dmolar_heat_capacity_v_dT,
      eos::einstein::compute_dhelmholtz_dTheta,
      eos::einstein::compute_d2helmholtz_dTheta2,
      eos::einstein::compute_dentropy_dTheta}}};
types::MineralParams anharmonic_parameters(bool lognormal, bool sigmoid) {
  types::MineralParams p;
  p.V_0 = 1.e-5;
  p.T_0 = 300.;
  p.debye_0 = 1000.;
  p.grueneisen_0 = 1.2;
  p.q_0 = 1.1;
  p.a_anh = -.2;
  p.m_anh = 3.;
  p.b_anh = .2;
  p.c_anh = 2.;
  p.mu_anh = .2;
  p.sigma_anh = 1.2;
  p.debye_temperature_model =
      std::make_shared<eos::debye_temperature_models::PowerLawGammaSimple>();
  if (sigmoid)
    p.anharmonic_prefactor_model =
        std::make_shared<eos::anharmonic_prefactor_models::Sigmoid>();
  else
    p.anharmonic_prefactor_model =
        std::make_shared<eos::anharmonic_prefactor_models::PowerLaw>();
  if (lognormal)
    p.anharmonic_thermal_model =
        std::make_shared<eos::anharmonic_thermal_models::LogNormal>();
  else
    p.anharmonic_thermal_model =
        std::make_shared<eos::anharmonic_thermal_models::Pade>();
  eos::AnharmonicDebye::validate_parameters(p);
  return p;
}
} // namespace

TEST_CASE("Lattice thermal derivatives", "[thermal_components][eos]") {
  const double t = GENERATE(100., 500., 3000.), theta = 700., n = .5,
               step = .01;
  for (const auto &m : lattices) {
    CHECK_THAT(m.df(t, theta, n),
               WithinRel((m.f(t, theta + step, n) - m.f(t, theta - step, n)) /
                             (2. * step),
                         2.e-7));
    CHECK_THAT(m.ddf(t, theta, n),
               WithinRel((m.df(t, theta + step, n) - m.df(t, theta - step, n)) /
                             (2. * step),
                         2.e-7));
    CHECK_THAT(m.ds(t, theta, n),
               WithinRel((m.s(t, theta + step, n) - m.s(t, theta - step, n)) /
                             (2. * step),
                         2.e-7));
    CHECK_THAT(m.s(t, theta, n),
               WithinRel(-(m.f(t + step, theta, n) - m.f(t - step, theta, n)) /
                             (2. * step),
                         2.e-7));
    CHECK_THAT(
        m.cv(t, theta, n),
        WithinRel(t * (m.s(t + step, theta, n) - m.s(t - step, theta, n)) /
                      (2. * step),
                  2.e-7));
    CHECK_THAT(m.dcv(t, theta, n),
               WithinRel((m.cv(t + step, theta, n) - m.cv(t - step, theta, n)) /
                             (2. * step),
                         2.e-6));
  }
}

TEST_CASE("Lattice zero, low and high temperature limits",
          "[thermal_components][eos]") {
  for (const auto &m : lattices) {
    for (auto fn : {m.f, m.s, m.cv, m.dcv, m.df, m.ddf, m.ds}) {
      CHECK(fn(0., 700., .5) == 0.);
      CHECK(fn(1.e-16, 700., .5) == 0.);
      CHECK(std::isfinite(fn(1.e-4, 700., .5)));
      CHECK(std::isfinite(fn(1.e8, 1., .5)));
    }
    CHECK_THAT(m.cv(1.e8, 1., .5),
               WithinRel(1.5 * constants::physics::gas_constant, 1.e-14));
  }
  const double prefactor = 1.5 * constants::physics::gas_constant;
  CHECK_THAT(lattices[0].dcv(1.e8, 1., .5),
             WithinRel(prefactor / 1.e24 / 10., 1.e-14));
  CHECK_THAT(lattices[1].dcv(1.e8, 1., .5),
             WithinRel(prefactor / 1.e24 / 6., 1.e-14));
  CHECK_THAT(lattices[0].dcv(1.e-4, 700., .5),
             WithinRel(3. * lattices[0].cv(1.e-4, 700., .5) / 1.e-4, 1.e-14));
  CHECK(lattices[1].cv(1.e-4, 700., .5) == 0.);
}

TEST_CASE("AnharmonicDebye obeys Helmholtz derivatives",
          "[thermal_components][eos]") {
  const bool lognormal = GENERATE(false, true), sigmoid = GENERATE(false, true);
  auto p = anharmonic_parameters(lognormal, sigmoid);
  const double t = 1500., v = *p.V_0 * GENERATE(.8, 1.), dt = .02,
               dv = v * 1.e-5;
  const auto state = eos::AnharmonicDebye::evaluate(t, v, p);
  const auto f = [&](double T, double V) {
    return eos::AnharmonicDebye::helmholtz_energy(T, V, p);
  };
  CHECK_THAT(state.pressure,
             WithinRel(-(f(t, v + dv) - f(t, v - dv)) / (2. * dv), 2.e-7));
  CHECK_THAT(state.bulk_modulus,
             WithinRel(-v *
                           (eos::AnharmonicDebye::pressure(t, v + dv, p) -
                            eos::AnharmonicDebye::pressure(t, v - dv, p)) /
                           (2. * dv),
                       2.e-7));
  CHECK_THAT(state.entropy,
             WithinRel(-(f(t + dt, v) - f(t - dt, v)) / (2. * dt), 2.e-7));
  CHECK_THAT(state.heat_capacity_v,
             WithinRel(t *
                           (eos::AnharmonicDebye::entropy(t + dt, v, p) -
                            eos::AnharmonicDebye::entropy(t - dt, v, p)) /
                           (2. * dt),
                       2.e-7));
  CHECK_THAT(state.alpha_kt,
             WithinRel((eos::AnharmonicDebye::pressure(t + dt, v, p) -
                        eos::AnharmonicDebye::pressure(t - dt, v, p)) /
                           (2. * dt),
                       2.e-7));
  CHECK_THAT(state.alpha_kt,
             WithinRel((eos::AnharmonicDebye::entropy(t, v + dv, p) -
                        eos::AnharmonicDebye::entropy(t, v - dv, p)) /
                           (2. * dv),
                       2.e-7));
  const auto reference = eos::AnharmonicDebye::evaluate(*p.T_0, v, p);
  CHECK(reference.helmholtz == 0.);
  CHECK(reference.pressure == 0.);
  CHECK(reference.bulk_modulus == 0.);
  CHECK(std::isfinite(eos::AnharmonicDebye::evaluate(0., v, p).helmholtz));
}

TEST_CASE("AnharmonicDebye validates required models and states",
          "[thermal_components][eos]") {
  auto p = anharmonic_parameters(false, false);
  SECTION("Missing prefactor") {
    p.anharmonic_prefactor_model.reset();
    CHECK_THROWS_AS(eos::AnharmonicDebye::validate_parameters(p),
                    std::invalid_argument);
  }
  SECTION("Missing temperature model") {
    p.debye_temperature_model.reset();
    CHECK_THROWS_AS(eos::AnharmonicDebye::validate_parameters(p),
                    std::invalid_argument);
  }
  SECTION("Missing thermal model") {
    p.anharmonic_thermal_model.reset();
    CHECK_THROWS_AS(eos::AnharmonicDebye::validate_parameters(p),
                    std::invalid_argument);
  }
  SECTION("Missing reference") {
    p.T_0.reset();
    CHECK_THROWS_AS(eos::AnharmonicDebye::validate_parameters(p),
                    std::invalid_argument);
  }
  SECTION("Invalid state") {
    CHECK_THROWS_AS(eos::AnharmonicDebye::evaluate(-1., *p.V_0, p),
                    std::domain_error);
    CHECK_THROWS_AS(eos::AnharmonicDebye::evaluate(300., 0., p),
                    std::domain_error);
  }
}
