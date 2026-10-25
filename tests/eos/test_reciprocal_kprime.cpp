// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/murnaghan.hpp"
#include "burnman/eos/reciprocal_kprime.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
types::MineralParams rkprime_params() {
  types::MineralParams p;
  p.V_0 = 11.24e-6;
  p.K_0 = 161.e9;
  p.Kprime_0 = 4.21796;
  p.Kprime_inf = 2.1;
  return p;
}
} // namespace

TEST_CASE("RKprime factory and reference state", "[rkprime][eos]") {
  auto model = eos::make_eos(types::EOSType::RKprime);
  REQUIRE(dynamic_cast<eos::RKprime *>(model.get()));
  auto p = rkprime_params();
  model->validate_parameters(p);
  CHECK(*p.F_0 == 0.);
  CHECK(*p.P_0 == 0.);
  CHECK_FALSE(p.E_0);
  CHECK(std::isnan(model->compute_shear_modulus(0., 300., *p.V_0, p)));
  p.F_0 = -125000.;
  p.P_0 = 2.5e9;
  p.G_0 = 80.e9;
  p.Gprime_inf = 1.2;
  const double t = GENERATE(0., 300., 2000.), pressure = *p.P_0, v = *p.V_0;
  CHECK(model->compute_volume(pressure, t, p) == v);
  CHECK(model->compute_pressure(t, v, p) == pressure);
  CHECK(model->compute_isothermal_bulk_modulus_reuss(pressure, t, v, p) ==
        *p.K_0);
  CHECK(model->compute_isentropic_bulk_modulus_reuss(pressure, t, v, p) ==
        *p.K_0);
  CHECK_THAT(model->compute_shear_modulus(pressure, t, v, p),
             WithinRel(*p.G_0, 1.e-15));
  CHECK(model->compute_helmholtz_free_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_molar_internal_energy(pressure, t, v, p) == *p.F_0);
  CHECK(model->compute_gibbs_free_energy(pressure, t, v, p) ==
        *p.F_0 + pressure * v);
  CHECK(model->compute_enthalpy(pressure, t, v, p) == *p.F_0 + pressure * v);
  CHECK(model->compute_entropy(pressure, t, v, p) == 0.);
  CHECK(model->compute_thermal_expansivity(pressure, t, v, p) == 0.);
  CHECK(model->compute_grueneisen_parameter(pressure, t, v, p) == 0.);
  CHECK(model->compute_molar_heat_capacity_p(pressure, t, v, p) == 1.e-99);
  CHECK(model->compute_molar_heat_capacity_v(pressure, t, v, p) == 1.e-99);
}

TEST_CASE("RKprime independent reference values", "[rkprime][eos]") {
  // Eqs. 58 and 61 evaluated at xi=-.1,.01,.1,.25,.4. Energies are from
  // 60-digit quadrature of V*dP/dxi, independently of the gamma expression.
  struct Reference {
    double ratio, pressure, bulk, energy;
  };
  const auto row = GENERATE(Reference{1.0848744595707804, -10978604077.916214,
                                      109786040779.16216, 5638.9207603965324},
                            Reference{.98983808684526808, 1680116211.4816177,
                                      168011621148.16177, 95.106366499810576},
                            Reference{.8828480152934618, 25849207070.478264,
                                      258492070704.78262, 15336.122850211423},
                            Reference{.63135649129493121, 179532758933.20456,
                                      718131035732.81824, 257790.78762392257},
                            Reference{.25940673863357178, 2555362817294.0698,
                                      6388407043235.1748, 3660913.7010948798});
  eos::RKprime model;
  auto p = rkprime_params();
  model.validate_parameters(p);
  const double v = row.ratio * *p.V_0;
  CAPTURE(row.ratio);
  CHECK_THAT(model.compute_pressure(300., v, p),
             WithinRel(row.pressure, 1.e-12));
  CHECK_THAT(model.compute_volume(row.pressure, 300., p), WithinRel(v, 1.e-13));
  CHECK_THAT(
      model.compute_isothermal_bulk_modulus_reuss(row.pressure, 300., v, p),
      WithinRel(row.bulk, 1.e-13));
  CHECK_THAT(model.compute_helmholtz_free_energy(row.pressure, 300., v, p),
             WithinRel(row.energy, 1.e-11));
}

TEST_CASE("RKprime thermodynamic derivatives and integer gamma orders",
          "[rkprime][eos]") {
  eos::RKprime model;
  auto p = rkprime_params();
  p.Kprime_inf = 2.;
  p.Kprime_0 = GENERATE(3.8, 4., 4.21796, 8.);
  p.P_0 = GENERATE(0., 2.5e9);
  p.F_0 = -125000.;
  model.validate_parameters(p);
  const double ratio = GENERATE(.6, .95, 1.02), v = ratio * *p.V_0,
               dv = v * 1.e-5;
  const double pressure = model.compute_pressure(300., v, p);
  CAPTURE(*p.Kprime_0, *p.P_0, ratio);
  const auto f = [&](double volume) {
    return model.compute_helmholtz_free_energy(pressure, 300., volume, p);
  };
  CHECK_THAT(model.compute_volume(pressure, 300., p), WithinRel(v, 1.e-13));
  CHECK_THAT(-(f(v + dv) - f(v - dv)) / (2. * dv), WithinRel(pressure, 2.e-7));
  const double derivative = (model.compute_pressure(300., v + dv, p) -
                             model.compute_pressure(300., v - dv, p)) /
                            (2. * dv);
  CHECK_THAT(-v * derivative,
             WithinRel(model.compute_isothermal_bulk_modulus_reuss(pressure,
                                                                   300., v, p),
                       2.e-7));
  const auto g = [&](double at_pressure) {
    return model.compute_gibbs_free_energy(
        at_pressure, 300., model.compute_volume(at_pressure, 300., p), p);
  };
  CHECK_THAT((g(pressure + 1.e5) - g(pressure - 1.e5)) / 2.e5,
             WithinRel(v, 2.e-7));
}

TEST_CASE("RKprime constant-Kprime limit and small strain energy",
          "[rkprime][eos]") {
  eos::RKprime model;
  eos::Murnaghan reference;
  auto p = rkprime_params();
  p.Kprime_inf = p.Kprime_0 = GENERATE(2., 4.);
  p.F_0 = -125000.;
  p.P_0 = 2.5e9;
  model.validate_parameters(p);
  auto rp = p;
  reference.validate_parameters(rp);
  for (double ratio : {.6, .95, 1.1}) {
    const double v = ratio * *p.V_0,
                 pressure = reference.compute_pressure(300., v, rp);
    CHECK_THAT(model.compute_pressure(300., v, p), WithinRel(pressure, 1.e-13));
    CHECK_THAT(model.compute_volume(pressure, 300., p), WithinRel(v, 1.e-13));
    CHECK_THAT(
        model.compute_isothermal_bulk_modulus_reuss(pressure, 300., v, p),
        WithinRel(reference.compute_isothermal_bulk_modulus_reuss(pressure,
                                                                  300., v, rp),
                  1.e-13));
    CHECK_THAT(model.compute_helmholtz_free_energy(pressure, 300., v, p),
               WithinRel(reference.compute_helmholtz_free_energy(pressure, 300.,
                                                                 v, rp),
                         1.e-12));
  }
  p.P_0 = p.F_0 = 0.;
  p.Kprime_0 = GENERATE(4., 4. + 1.e-12);
  p.Kprime_inf = 2.;
  model.validate_parameters(p);
  const double v = *p.V_0 * GENERATE(1. - 1.e-8, 1. + 1.e-8), dv = v - *p.V_0;
  CHECK_THAT(model.compute_helmholtz_free_energy(0., 300., v, p),
             WithinRel(.5 * *p.K_0 / *p.V_0 * dv * dv, 1.e-7));
}

TEST_CASE("RKprime tension bounds and high-pressure asymptote",
          "[rkprime][eos]") {
  eos::RKprime model;
  auto p = rkprime_params();
  model.validate_parameters(p);
  const double a = *p.Kprime_inf, r = *p.Kprime_0 / a,
               xi = -1. / (*p.Kprime_0 - a);
  const double minimum = *p.K_0 * xi * std::pow(1. - a * xi, -r);
  const double maximum =
      *p.V_0 * std::pow(1. - a * xi, r / a) * std::exp((r - 1.) * xi);
  CHECK_THROWS_AS(model.compute_volume(minimum * 1.001, 300., p),
                  std::invalid_argument);
  CHECK_THROWS_AS(model.compute_pressure(300., maximum * 1.001, p),
                  std::invalid_argument);
  const double pressure = .999 * minimum,
               v = model.compute_volume(pressure, 300., p);
  CHECK_THAT(model.compute_pressure(300., v, p), WithinRel(pressure, 1.e-12));
  CHECK_THAT(model.compute_isothermal_bulk_modulus_reuss(1.e30, 300., 0., p) /
                 1.e30,
             WithinRel(a, 1.e-8));
  const double compressed = model.compute_volume(1.e30, 300., p);
  CHECK_THAT(model.compute_pressure(300., compressed, p),
             WithinRel(1.e30, 1.e-12));
}

TEST_CASE("RKprime parameter and state validation", "[rkprime][eos]") {
  eos::RKprime model;
  auto p = rkprime_params();
  SECTION("Required parameters") {
    const auto field = GENERATE(
        &types::MineralParams::V_0, &types::MineralParams::K_0,
        &types::MineralParams::Kprime_0, &types::MineralParams::Kprime_inf);
    (p.*field).reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Helmholtz reference") {
    p.E_0 = 0.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid shape") {
    p.Kprime_inf = GENERATE(0., 5., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid states") {
    model.validate_parameters(p);
    const double v =
        GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.compute_pressure(300., v, p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_helmholtz_free_energy(0., 300., v, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        model.compute_volume(std::numeric_limits<double>::quiet_NaN(), 300., p),
        std::invalid_argument);
  }
}
