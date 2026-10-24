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
#include "burnman/eos/cork.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/utils/exceptions.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>

using namespace burnman;
using namespace Catch::Matchers;

namespace {
types::MineralParams co2_params() {
  types::MineralParams params;
  params.equation_of_state = types::EOSType::CORK;
  params.cork_params =
      types::CorkParams{54.5963,    -8.63920,    0.918301,     -0.0330558,
                        0.00230524, 0.000693054, -0.0000838293};
  params.cork_T = 304.2;
  params.cork_P = 0.0738e8;
  params.Cp = types::CpParams{87.8, -0.002644, 706400.0, -998.9};
  params.H_0 = -393510.0;
  params.S_0 = 213.7;
  params.molar_mass = 0.0440095;
  return params;
}
} // namespace

TEST_CASE("CORK parameter validation", "[cork][eos]") {
  eos::CORK cork;
  auto params = co2_params();
  SECTION("Required parameters") {
    SECTION("cork_params") { params.cork_params.reset(); }
    SECTION("cork_T") { params.cork_T.reset(); }
    SECTION("cork_P") { params.cork_P.reset(); }
    SECTION("Cp") { params.Cp.reset(); }
    REQUIRE_THROWS_AS(cork.validate_parameters(params), std::invalid_argument);
  }
  SECTION("Default reference state and derivatives") {
    params.H_0.reset();
    params.S_0.reset();
    cork.validate_parameters(params);
    CHECK(*params.T_0 == 298.15);
    CHECK(*params.P_0 == 0.0);
    CHECK(std::isnan(*params.H_0));
    CHECK(std::isnan(*params.S_0));
    REQUIRE(params.dcorkdT.has_value());
    CHECK_THAT((*params.dcorkdT)[0],
               WithinRel(-8.63920 * std::pow(304.2, 1.5) / 0.0738e8, 1e-14));
    CHECK((*params.dcorkdT)[1] == 0.0);
    CHECK_THAT((*params.dcorkdT)[2],
               WithinRel(0.00230524 / std::pow(0.0738e8, 1.5), 1e-14));
    CHECK_THAT((*params.dcorkdT)[3],
               WithinRel(-0.0000838293 / std::pow(0.0738e8, 2.0), 1e-14));
    CHECK(cork.compute_volume(1e8, 800.0, params) > 0.0);
  }
  SECTION("Explicit values are preserved") {
    params.T_0 = 350.0;
    params.P_0 = 2e7;
    const std::array<double, 4> derivatives{-0.008, 0.0, 2e-14, -1e-18};
    params.dcorkdT = derivatives;
    cork.validate_parameters(params);
    CHECK(*params.T_0 == 350.0);
    CHECK(*params.P_0 == 2e7);
    CHECK(*params.H_0 == -393510.0);
    CHECK(*params.S_0 == 213.7);
    CHECK(*params.dcorkdT == derivatives);
  }
}

TEST_CASE("CORK Python reference values", "[cork][eos]") {
  // Python BurnMan e2cfac31052273b3fa9d5ce905978e97723005e9,
  // HP_2011_fluids.CO2 evaluated with its CORK equation of state.
  struct State {
    double pressure, temperature, volume, bulk_modulus, alpha, cp, gibbs,
        entropy;
  };
  const auto state =
      GENERATE(State{1e5, 298.15, 0.024592667423870663, 99209.69850191954,
                     0.003434593711052317, 37.446339594844176,
                     -457244.3941039706, 213.56739902430309},
               State{1e8, 800.0, 8.949082367226386e-5, 153199621.4243088,
                     0.0011435131001850448, 60.030689470458086,
                     -529151.9636299624, 195.44856680143707},
               State{1e9, 1600.0, 4.2374052793937094e-5, 2539842573.019058,
                     0.0002562125702205587, 62.524578265140974,
                     -645677.7429926374, 215.03034228726855});
  auto cork = eos::make_eos(types::EOSType::CORK);
  REQUIRE(dynamic_cast<eos::CORK *>(cork.get()) != nullptr);
  auto params = co2_params();
  cork->validate_parameters(params);
  const double p = state.pressure, t = state.temperature;
  const double v = cork->compute_volume(p, t, params);
  CAPTURE(p, t);
  CHECK_THAT(v, WithinRel(state.volume, 2e-12));
  CHECK_THAT(cork->compute_isothermal_bulk_modulus_reuss(p, t, v, params),
             WithinRel(state.bulk_modulus, 2e-12));
  CHECK_THAT(cork->compute_thermal_expansivity(p, t, v, params),
             WithinRel(state.alpha, 2e-12));
  CHECK_THAT(cork->compute_molar_heat_capacity_p(p, t, v, params),
             WithinRel(state.cp, 2e-12));
  CHECK_THAT(cork->compute_gibbs_free_energy(p, t, v, params),
             WithinRel(state.gibbs, 2e-12));
  CHECK_THAT(cork->compute_entropy(p, t, v, params),
             WithinRel(state.entropy, 2e-12));
  CHECK(cork->compute_shear_modulus(p, t, v, params) == 0.0);
  CHECK_THROWS_AS(cork->compute_pressure(t, v, params),
                  exceptions::NotImplementedError);

  Mineral material;
  material.params = params;
  material.set_method(types::EOSType::Auto);
  material.set_state(p, t);
  CHECK_THAT(material.get_molar_gibbs(), WithinRel(state.gibbs, 2e-12));
  CHECK_THAT(material.get_molar_enthalpy(),
             WithinRel(state.gibbs + t * state.entropy, 2e-12));
  CHECK_THAT(material.get_molar_internal_energy(),
             WithinRel(state.gibbs + t * state.entropy - p * v, 2e-12));
  CHECK_THAT(material.get_density(), WithinRel(*params.molar_mass / v, 2e-12));
  CHECK(material.get_shear_wave_velocity() == 0.0);
}

TEST_CASE("CORK zero critical temperature", "[cork][eos]") {
  // HP_2011_fluids.O2 at 1 bar and 298.15 K.
  auto params = co2_params();
  params.cork_T = 0.0;
  params.cork_P = 1e5;
  params.Cp = types::CpParams{48.3, -0.000691, 499200.0, -420.7};
  params.H_0 = 0.0;
  params.S_0 = 205.2;
  eos::CORK cork;
  cork.validate_parameters(params);
  const double p = 1e5, t = 298.15;
  const double v = cork.compute_volume(p, t, params);
  CHECK_THAT(v, WithinRel(0.02479619343202593, 2e-12));
  CHECK_THAT(cork.compute_isothermal_bulk_modulus_reuss(p, t, v, params),
             WithinRel(100039.57711457305, 2e-12));
  CHECK_THAT(cork.compute_thermal_expansivity(p, t, v, params),
             WithinRel(0.0033540164346805303, 2e-12));
  CHECK_THAT(cork.compute_molar_heat_capacity_p(p, t, v, params),
             WithinRel(29.34532647227213, 2e-12));
  CHECK_THAT(cork.compute_gibbs_free_energy(p, t, v, params),
             WithinRel(-61180.38, 2e-12));
  CHECK(cork.compute_entropy(p, t, v, params) == 205.2);
}
