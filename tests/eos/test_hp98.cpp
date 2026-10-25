// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/hp98.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/murnaghan.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;
namespace {
types::MineralParams hp98_params() {
  types::MineralParams p;
  p.V_0 = 2.445e-5;
  p.K_0 = 251.e9;
  p.Kprime_0 = 4.14;
  p.Cp = types::CpParams{149.3, .002918, -2983000., -799.1};
  p.a_0 = 1.87e-5;
  p.dKdT_0 = -1.e7;
  p.H_0 = -1443030.;
  p.S_0 = 62.6;
  p.napfu = 5.;
  p.molar_mass = .1003887;
  return p;
}
} // namespace

TEST_CASE("HP98 factory and zero-pressure reference", "[hp98][eos]") {
  auto model = eos::make_eos(types::EOSType::HP98);
  REQUIRE(dynamic_cast<eos::HP98 *>(model.get()));
  auto p = hp98_params();
  model->validate_parameters(p);
  CHECK(*p.T_0 == 298.15);
  CHECK(*p.P_0 == 0.);
  CHECK_FALSE(p.E_0);
  CHECK_FALSE(p.Kdprime_0);
  CHECK_FALSE(p.T_einstein);
  const double t = *p.T_0, v = *p.V_0;
  CHECK(model->compute_volume(0., t, p) == v);
  CHECK(model->compute_pressure(t, v, p) == 0.);
  CHECK(model->compute_isothermal_bulk_modulus_reuss(0., t, v, p) == *p.K_0);
  CHECK(model->compute_entropy(0., t, v, p) == *p.S_0);
  CHECK_THAT(model->compute_enthalpy(0., t, v, p), WithinRel(*p.H_0, 1.e-15));
  CHECK_THAT(model->compute_molar_internal_energy(0., t, v, p),
             WithinRel(*p.H_0, 1.e-15));
  CHECK(model->compute_gibbs_free_energy(0., t, v, p) == *p.H_0 - t * *p.S_0);
  CHECK(model->compute_helmholtz_free_energy(0., t, v, p) ==
        *p.H_0 - t * *p.S_0);
  CHECK_THAT(model->compute_thermal_expansivity(0., t, v, p),
             WithinRel(*p.a_0 * (1. - 10. / std::sqrt(t)), 1.e-15));
  CHECK(model->compute_shear_modulus(0., t, v, p) == 0.);
}

TEST_CASE("HP98 independent thermodynamic reference values", "[hp98][eos]") {
  // 60-digit evaluation of G(P,T), with independent temperature derivatives.
  // Includes the removable Kprime=0 and 1 limits absent from Python HP98.
  struct Reference {
    double q, pressure, t, gibbs, v, entropy, cp, alpha, bulk;
  };
  const auto row = GENERATE(
      Reference{4.14, 0., 298.15, -1461694.1899999999, 2.4450000000000001e-5,
                62.6, 70.333989032426132, 7.8701061545210213e-6, 251.e9},
      Reference{4.14, 1.e5, 1000., -1554896.652603209, 2.4639612909076454e-5,
                188.01919451341959, 123.96523198839428, 1.2688120590792251e-5,
                243981914000.},
      Reference{4.14, 1.e10, 3000., -1853010.1928389042, 2.4325135752288525e-5,
                331.7511078176048, 142.7985457998513, 1.3065256505458962e-5,
                265381500000.},
      Reference{4.14, 1.e11, 1200., 557448.76836451143, 1.9415210023011955e-5,
                191.33924832419291, 122.79411698138659, 6.8655015739649537e-6,
                655981500000.},
      Reference{0., 1.e10, 1200., -1352856.0871263109, 2.3703390510087479e-5,
                207.98345722545307, 127.04211810888202, 1.1457499319641676e-5,
                241981500000.},
      Reference{1., 1.e10, 1200., -1352789.2719041484, 2.3723098066099857e-5,
                207.97719451329669, 127.04098889170973, 1.1525273927600792e-5,
                251981500000.});
  auto p = hp98_params();
  p.Kprime_0 = row.q;
  eos::HP98 model;
  model.validate_parameters(p);
  CAPTURE(row.q, row.pressure, row.t);
  CHECK_THAT(model.compute_volume(row.pressure, row.t, p),
             WithinRel(row.v, 1.e-13));
  CHECK_THAT(model.compute_gibbs_free_energy(row.pressure, row.t, row.v, p),
             WithinRel(row.gibbs, 1.e-13));
  CHECK_THAT(model.compute_entropy(row.pressure, row.t, row.v, p),
             WithinRel(row.entropy, 1.e-13));
  CHECK_THAT(model.compute_molar_heat_capacity_p(row.pressure, row.t, row.v, p),
             WithinRel(row.cp, 1.e-13));
  CHECK_THAT(model.compute_thermal_expansivity(row.pressure, row.t, row.v, p),
             WithinRel(row.alpha, 1.e-13));
  CHECK_THAT(model.compute_isothermal_bulk_modulus_reuss(row.pressure, row.t,
                                                         row.v, p),
             WithinRel(row.bulk, 1.e-13));
}

TEST_CASE("HP98 Gibbs derivatives and response identities", "[hp98][eos]") {
  eos::HP98 model;
  auto p = hp98_params();
  p.Kprime_0 = GENERATE(-1., 0., 1. - 1.e-12, 1., 1. + 1.e-12, 4.14);
  model.validate_parameters(p);
  const double pressure = GENERATE(0., 1.e10), t = GENERATE(300., 2000.);
  const double v = model.compute_volume(pressure, t, p), dp = 1.e6, dt = .02;
  CAPTURE(*p.Kprime_0, pressure, t);
  const auto g = [&](double P, double T) {
    return model.compute_gibbs_free_energy(P, T, 0., p);
  };
  const auto s = [&](double P, double T) {
    return model.compute_entropy(P, T, 0., p);
  };
  const double alpha = model.compute_thermal_expansivity(pressure, t, v, p);
  const double kt =
      model.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p);
  const double cp = model.compute_molar_heat_capacity_p(pressure, t, v, p);
  const double cv = model.compute_molar_heat_capacity_v(pressure, t, v, p);
  CHECK_THAT(model.compute_pressure(t, v, p),
             WithinRel(pressure, 1.e-12) || WithinAbs(pressure, .01));
  CHECK_THAT((g(pressure + dp, t) - g(pressure - dp, t)) / (2. * dp),
             WithinRel(v, 2.e-7));
  CHECK_THAT(-(g(pressure, t + dt) - g(pressure, t - dt)) / (2. * dt),
             WithinRel(s(pressure, t), 2.e-7));
  CHECK_THAT(t * (s(pressure, t + dt) - s(pressure, t - dt)) / (2. * dt),
             WithinRel(cp, 2.e-7));
  CHECK_THAT((model.compute_volume(pressure, t + dt, p) -
              model.compute_volume(pressure, t - dt, p)) /
                 (2. * dt * v),
             WithinRel(alpha, 2.e-7));
  CHECK_THAT(-(s(pressure + dp, t) - s(pressure - dp, t)) / (2. * dp),
             WithinRel(alpha * v, 2.e-7));
  CHECK_THAT(cv, WithinRel(cp - v * t * alpha * alpha * kt, 1.e-13));
  CHECK_THAT(model.compute_isentropic_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(kt * cp / cv, 1.e-13));
  CHECK_THAT(model.compute_grueneisen_parameter(pressure, t, v, p),
             WithinRel(alpha * kt * v / cv, 1.e-13));
  CHECK_THAT(model.compute_enthalpy(pressure, t, v, p),
             WithinRel(g(pressure, t) + t * s(pressure, t), 1.e-13));
  CHECK_THAT(
      model.compute_molar_internal_energy(pressure, t, v, p),
      WithinRel(g(pressure, t) + t * s(pressure, t) - pressure * v, 1.e-13));
}

TEST_CASE("HP98 constant thermal parameters and pressure convention",
          "[hp98][eos]") {
  eos::HP98 model;
  auto p = hp98_params();
  model.validate_parameters(p);
  const double pressure = 1.e10, t = 1200.;
  const double g = model.compute_gibbs_free_energy(pressure, t, 0., p),
               v = model.compute_volume(pressure, t, p);
  p.P_0 = 2.5e9;
  p.F_0 = 1.e12;
  CHECK(model.compute_gibbs_free_energy(pressure, t, 0., p) == g);
  CHECK(model.compute_volume(pressure, t, p) == v);
  p.a_0 = p.dKdT_0 = 0.;
  auto rp = p;
  rp.P_0 = rp.F_0 = 0.;
  eos::Murnaghan reference;
  reference.validate_parameters(rp);
  const double static_v = reference.compute_volume(pressure, t, rp);
  CHECK_THAT(model.compute_volume(pressure, t, p), WithinRel(static_v, 1.e-13));
  CHECK(model.compute_entropy(pressure, t, static_v, p) ==
        model.compute_entropy(0., t, *p.V_0, p));
  CHECK(model.compute_molar_heat_capacity_p(pressure, t, static_v, p) ==
        model.compute_molar_heat_capacity_p(0., t, *p.V_0, p));
  CHECK(model.compute_thermal_expansivity(pressure, t, static_v, p) == 0.);
  CHECK_THAT(
      model.compute_gibbs_free_energy(pressure, t, static_v, p) -
          model.compute_gibbs_free_energy(0., t, *p.V_0, p),
      WithinRel(reference.compute_gibbs_free_energy(pressure, t, static_v, rp),
                1.e-12));
}

TEST_CASE("HP98 optional reference enthalpy and entropy", "[hp98][eos]") {
  eos::HP98 model;
  auto p = hp98_params();
  p.H_0.reset();
  p.S_0.reset();
  model.validate_parameters(p);
  const double v = model.compute_volume(1.e10, 1200., p);
  CHECK(std::isfinite(v));
  CHECK(std::isfinite(
      model.compute_isothermal_bulk_modulus_reuss(1.e10, 1200., v, p)));
  CHECK(std::isfinite(model.compute_molar_heat_capacity_p(1.e10, 1200., v, p)));
  CHECK(std::isnan(model.compute_gibbs_free_energy(1.e10, 1200., v, p)));
  CHECK(std::isnan(model.compute_entropy(1.e10, 1200., v, p)));
}

TEST_CASE("HP98 parameter and state validation", "[hp98][eos]") {
  eos::HP98 model;
  auto p = hp98_params();
  SECTION("Required scalars") {
    const auto field =
        GENERATE(&types::MineralParams::V_0, &types::MineralParams::K_0,
                 &types::MineralParams::Kprime_0, &types::MineralParams::a_0,
                 &types::MineralParams::dKdT_0, &types::MineralParams::napfu,
                 &types::MineralParams::molar_mass);
    (p.*field).reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Required Cp") {
    p.Cp.reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Enthalpy reference") {
    p.E_0 = 0.;
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid reference temperature") {
    p.T_0 = GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid heat capacity") {
    p.Cp->a = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Invalid states") {
    model.validate_parameters(p);
    const double value =
        GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN());
    CHECK_THROWS_AS(model.compute_volume(0., value, p), std::invalid_argument);
    CHECK_THROWS_AS(model.compute_pressure(300., value, p),
                    std::invalid_argument);
    CHECK_THROWS_AS(model.compute_volume(-2. * *p.K_0 / *p.Kprime_0, *p.T_0, p),
                    std::invalid_argument);
  }
  SECTION("Thermal bulk modulus becomes nonpositive") {
    model.validate_parameters(p);
    p.dKdT_0 = -1.e9;
    CHECK_THROWS_AS(model.compute_volume(0., *p.T_0 + 300., p),
                    std::invalid_argument);
  }
  SECTION("Thermal reference volume becomes nonpositive") {
    model.validate_parameters(p);
    p.a_0 = 1.;
    CHECK_THROWS_AS(model.compute_volume(0., 100., p), std::invalid_argument);
  }
}
