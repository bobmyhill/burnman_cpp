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

/* Absolute energy benchmarks from Holland & Powell (2011), Table 2a, and
 * https://hpxeosandthermocalc.org/wp-content/uploads/2020/09/
 * 0_metapelite_benchmarks_2020-09-10.zip.
 * See tests/reference/thermocalc_metapelite.json and
 * docs/thermocalc_reference.md for source versions and unit conversions.
 */
#include "burnman/minerals/datasets.hpp"
#include "burnman/minerals/water.hpp"
#include "thermocalc_reference.hpp"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <map>
#include <string>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE("Water reference enthalpy matches Holland-Powell Table 2a",
          "[water][ps1994][hp2011_reference][thermocalc]") {
  // The table gives the gas thermal reference, not stable liquid water at
  // 298.15 K. Evaluate it without extrapolating the PS1994 fluid EOS domain.
  const auto gas = minerals::water_ideal_gas_reference(298.15);
  CHECK_THAT(gas.enthalpy, WithinAbs(-241810., 10.));
  CHECK_THAT(gas.entropy, WithinAbs(188.80, .005));
  CHECK_THAT(gas.gibbs, WithinAbs(-298100.72, 10.));
}

TEST_CASE("Pure phase energies match THERMOCALC metapelite benchmarks",
          "[native_datasets][ps1994][thermocalc]") {
  const std::map<std::string, std::shared_ptr<Mineral>> phases = {
      {"H2O", minerals::water_fluid()},
      {"q", minerals::HP11::q()},
      {"and", minerals::HP11::andalusite()},
      {"sill", minerals::HP11::sill()}};
  for (const auto &state : thermocalc_reference::phases) {
    DYNAMIC_SECTION(state.source << ": " << state.phase) {
      REQUIRE(phases.count(state.phase) == 1);
      auto phase = phases.at(state.phase);
      phase->set_state(state.pressure, state.temperature);
      CHECK_THAT(
          phase->get_molar_gibbs(),
          WithinAbs(state.gibbs, thermocalc_reference::energy_tolerance));
      CHECK_THAT(
          phase->get_molar_enthalpy(),
          WithinAbs(state.enthalpy, thermocalc_reference::energy_tolerance));
    }
  }
}

TEST_CASE("Hydrous and anhydrous endmember energies match THERMOCALC",
          "[native_datasets][thermocalc]") {
  namespace mp = minerals::MP14;
  const std::map<std::string, std::shared_ptr<Solution>> phases = {
      {"mu", mp::mu()},   {"bi", mp::bi()},     {"chl", mp::chl()},
      {"ep", mp::ep()},   {"g", mp::g()},       {"opx", mp::opx()},
      {"cd", mp::cd()},   {"plc", mp::plc()},   {"pl", mp::pl4tr()},
      {"ksp", mp::ksp()}, {"ilmm", mp::ilmm()}, {"mt1", mp::mt1()}};
  for (const auto &state : thermocalc_reference::endmembers) {
    DYNAMIC_SECTION(state.source << ": " << state.phase << "/"
                                 << state.endmember) {
      std::string name = state.phase;
      // Coexisting muscovite/paragonite and ilmenite/hematite solutions share
      // standard endmembers. Their compositions and activities are not used.
      if (name == "pa")
        name = "mu";
      if (name == "ilm" || name == "hem")
        name = "ilmm";
      REQUIRE(phases.count(name) == 1);
      auto phase = phases.at(name);
      phase->set_state(state.pressure, state.temperature);
      const auto &names = phase->get_endmember_names();
      const auto found = std::find(names.begin(), names.end(), state.endmember);
      REQUIRE(found != names.end());
      const auto index = static_cast<Eigen::Index>(found - names.begin());
      const auto energies =
          phase->map_endmembers_to_array(&Mineral::get_molar_gibbs);
      CHECK_THAT(
          energies(index),
          WithinAbs(state.gibbs, thermocalc_reference::energy_tolerance));
    }
  }
}
