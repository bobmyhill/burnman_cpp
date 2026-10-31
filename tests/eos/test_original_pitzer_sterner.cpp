// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Port of test_reference_properties_and_stable_roots; see
// tests/reference/burnman.json.

#include "burnman/eos/pitzer_sterner.hpp"
#include "burnman/minerals/water.hpp"
#include "ps1994_reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE("Original PS1994 golden states require their reference gas constant",
          "[eos][pitzer_sterner][original]") {
  // Upstream patches Python's R. A Python attribute patch cannot configure a
  // compiled EOS: use the existing alternate-R constructor explicitly.
  const bool ideal_reference = GENERATE(false, true);
  auto phase = ideal_reference
                   ? minerals::water_fluid()
                   : minerals::Pitzer_Sterner_1994::H2O_Pitzer_Sterner();
  if (!ideal_reference)
    phase->set_method(std::make_shared<eos::PitzerSterner>(8.31451));
  auto reference = phase->copy();
  for (const auto &s : ps1994_reference::states) {
    INFO("HP reference=" << ideal_reference << " P=" << s.pressure
                         << " T=" << s.temperature);
    phase->set_state(s.pressure, s.temperature);
    reference->set_state(1.e5, s.temperature);
    CHECK_THAT(phase->get_molar_volume(), WithinRel(s.volume, 2.e-12));
    CHECK_THAT(phase->get_isothermal_bulk_modulus_reuss(),
               WithinRel(s.bulk_modulus, 2.e-8));
    CHECK_THAT(phase->get_thermal_expansivity(),
               WithinRel(s.expansivity, 2.e-6));
    CHECK_THAT(
        phase->get_molar_gibbs() - reference->get_molar_gibbs(),
        WithinAbs(s.delta_gibbs, 2.e-8 + 2.e-11 * std::abs(s.delta_gibbs)));
    CHECK_THAT(
        phase->get_molar_entropy() - reference->get_molar_entropy(),
        WithinAbs(s.delta_entropy, 2.e-7 + 2.e-7 * std::abs(s.delta_entropy)));
    CHECK_THAT(phase->get_molar_heat_capacity_p() -
                   reference->get_molar_heat_capacity_p(),
               WithinAbs(s.delta_cp, 2.e-3 + 2.e-5 * std::abs(s.delta_cp)));
    CHECK(phase->get_shear_modulus() == 0.);
  }
}
