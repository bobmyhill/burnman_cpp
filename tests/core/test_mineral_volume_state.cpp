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
#include "burnman/minerals/datasets.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE(
    "Volume states preserve direct EOS volumes and include modifier volumes",
    "[mineral][volume_state]") {
  auto mineral = minerals::SLB_2011::forsterite();
  SECTION("Unmodified EOS") {}
  SECTION("Linear Gibbs modifier") {
    mineral->set_property_modifier_params(
        {eos::excesses::LinearParams{2.e-7, 0.5, 100.}});
  }
  for (const double pressure : {-2.e9, 2.e9, 42.e9}) {
    CAPTURE(pressure);
    mineral->set_state(pressure, 700.);
    const double volume = mineral->get_molar_volume();
    const double gibbs = mineral->get_molar_gibbs();
    mineral->set_state(1.e9, 300.);
    mineral->set_state_with_volume(volume, 700.);
    CHECK_THAT(mineral->get_pressure(), WithinRel(pressure, 1.e-10));
    CHECK(mineral->get_molar_volume() == volume);
    CHECK_THAT(mineral->get_molar_gibbs(), WithinRel(gibbs, 1.e-12));
    CHECK_THAT(
        mineral->get_molar_volume_unmodified(),
        WithinRel(volume - mineral->get_property_modifiers().dGdP, 1.e-14));
  }
}

TEST_CASE(
    "Volume states invert EOS implementations without a pressure function",
    "[mineral][volume_state]") {
  class VolumeOnlyEOS : public EquationOfState {
  public:
    void validate_parameters(types::MineralParams &) override {}
    double compute_volume(double p, double,
                          const types::MineralParams &params) const override {
      return *params.V_0 / (1. + p / *params.K_0);
    }
  };
  Mineral mineral;
  mineral.params.V_0 = 1.e-5;
  mineral.params.K_0 = 1.e11;
  mineral.set_method(std::make_shared<VolumeOnlyEOS>());
  mineral.set_state_with_volume(0.83e-5, 500.);
  CHECK_THAT(mineral.get_pressure(),
             WithinRel(1.e11 * (1. / 0.83 - 1.), 1.e-12));
  CHECK(mineral.get_molar_volume() == 0.83e-5);
  CHECK(mineral.get_temperature() == 500.);
}

TEST_CASE("Invalid independent volume inputs do not change a mineral state",
          "[mineral][volume_state]") {
  auto mineral = minerals::HP_2011_ds62::fo();
  mineral->set_state(1.e9, 600.);
  for (const double invalid :
       {0., -1., std::numeric_limits<double>::quiet_NaN()})
    CHECK_THROWS_AS(mineral->set_state_with_volume(invalid, 300.),
                    std::invalid_argument);
  CHECK_THROWS_AS(mineral->set_state_with_volume(1.e-5, -1.),
                  std::invalid_argument);
  CHECK(mineral->get_pressure() == 1.e9);
  CHECK(mineral->get_temperature() == 600.);
}
