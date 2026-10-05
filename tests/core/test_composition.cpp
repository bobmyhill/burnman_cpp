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

#include "burnman/core/composition.hpp"
#include "burnman/utils/chemistry_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>

using namespace burnman;
using namespace Catch::Matchers;

namespace {
types::FormulaMap amounts(const Composition::ComponentAmounts &entries) {
  return {entries.begin(), entries.end()};
}
} // namespace

TEST_CASE("Composition converts kg, component moles and atom moles",
          "[composition]") {
  const Composition c({{"MgSiO3", 0.5}, {"FeSiO3", 0.5}}, "molar");
  const auto mass = amounts(c.get_mass_composition());
  CHECK_THAT(mass.at("MgSiO3"),
             WithinRel(0.5 * (0.024305 + 0.0280855 + 3.0 * 0.0159994), 1.e-14));
  CHECK(c.get_weight_composition() == c.get_mass_composition());
  const auto atoms = amounts(c.get_atomic_composition());
  CHECK_THAT(atoms.at("Mg"), WithinAbs(0.5, 1.e-14));
  CHECK_THAT(atoms.at("Si"), WithinAbs(1.0, 1.e-14));
  CHECK_THAT(atoms.at("O"), WithinAbs(3.0, 1.e-14));
  CHECK(c.get_element_list() ==
        (std::vector<std::string>{"Mg", "Si", "O", "Fe"}));
}

TEST_CASE("Composition normalizes in the requested basis", "[composition]") {
  Composition c({{"MgO", 2.0}, {"SiO2", 2.0}}, "molar", true);
  CHECK_THAT(amounts(c.get_molar_composition()).at("MgO"),
             WithinAbs(0.5, 1.e-14));
  c.renormalize("atomic", "O", 3.0);
  CHECK_THAT(amounts(c.get_molar_composition()).at("MgO"),
             WithinAbs(1.0, 1.e-14));
  c.renormalize("mass", "total", 1.0);
  const auto mass = c.get_mass_composition();
  CHECK_THAT(mass[0].second + mass[1].second, WithinAbs(1.0, 1.e-14));
  CHECK_THROWS_AS(c.renormalize("molar", "FeO", 1.0), std::invalid_argument);
  CHECK(c.get_mass_composition() == mass);
}

TEST_CASE("Composition arithmetic preserves signed amounts and value semantics",
          "[composition]") {
  Composition c({{"MgO", 1.0}, {"SiO2", 1.0}}, "molar");
  const Composition d({{"MgO", 2.0}, {"FeO", 3.0}}, "molar");
  auto difference = c - d;
  CHECK_THAT(amounts(difference.get_molar_composition()).at("MgO"),
             WithinAbs(-1.0, 1.e-14));
  CHECK_THAT(amounts(difference.get_molar_composition()).at("FeO"),
             WithinAbs(-3.0, 1.e-14));
  difference += d;
  CHECK_THAT(amounts(difference.get_molar_composition()).at("MgO"),
             WithinAbs(1.0, 1.e-14));
  const auto original = c.get_mass_composition();
  auto scaled = 2.0 * c;
  scaled /= 2.0;
  CHECK(scaled.get_mass_composition() == original);
  CHECK(c.get_mass_composition() == original);
  CHECK_THROWS_AS(c *= std::numeric_limits<double>::infinity(),
                  std::invalid_argument);
  CHECK(c.get_mass_composition() == original);
  c += c;
  CHECK_THAT(amounts(c.get_molar_composition()).at("SiO2"),
             WithinAbs(2.0, 1.e-14));
  c -= c;
  c.remove_null_components();
  CHECK(c.get_mass_composition().empty());
  CHECK(c.get_element_list().empty());
}

TEST_CASE("Native NNLS changes composition bases while preserving the bulk",
          "[composition]") {
  for (const double scale : {1.e-18, 1.0, 1.e18}) {
    Composition c({{"MgSiO3", 0.5 * scale}, {"FeSiO3", 0.5 * scale}}, "molar");
    const auto atoms = amounts(c.get_atomic_composition());
    c.change_component_set({"MgO", "SiO2", "FeO", "Al2O3"});
    const auto molar = amounts(c.get_molar_composition());
    CHECK_THAT(molar.at("MgO"), WithinRel(0.5 * scale, 1.e-12));
    CHECK_THAT(molar.at("SiO2"), WithinRel(scale, 1.e-12));
    CHECK_THAT(molar.at("FeO"), WithinRel(0.5 * scale, 1.e-12));
    CHECK_THAT(molar.at("Al2O3"), WithinAbs(0.0, 1.e-30));
    const auto new_atoms = amounts(c.get_atomic_composition());
    for (const auto &[element, amount] : atoms)
      CHECK_THAT(new_atoms.at(element), WithinRel(amount, 1.e-12));
  }
}

TEST_CASE("Native NNLS accepts redundant bases and rejects incompatible bases "
          "atomically",
          "[composition]") {
  Composition c({{"MgO", 1.0}, {"SiO2", 1.0}}, "molar");
  c.change_component_set({"MgO", "SiO2", "MgSiO3", "Mg2Si2O6"});
  auto atoms = amounts(c.get_atomic_composition());
  CHECK_THAT(atoms.at("Mg"), WithinAbs(1.0, 1.e-12));
  CHECK_THAT(atoms.at("Si"), WithinAbs(1.0, 1.e-12));
  CHECK_THAT(atoms.at("O"), WithinAbs(3.0, 1.e-12));
  for (const auto &entry : c.get_molar_composition())
    CHECK(entry.second >= 0.0);
  const auto original = c.get_mass_composition();
  CHECK_THROWS_AS(c.change_component_set({"MgSiO2"}), std::invalid_argument);
  CHECK_THROWS_AS(c.change_component_set({"MgO", "SiO2", "SiO2"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(c.change_component_set({}), std::invalid_argument);
  CHECK(c.get_mass_composition() == original);
  c.change_component_set({"MgO2", "SiO"});
  CHECK_THAT(amounts(c.get_molar_composition()).at("SiO"),
             WithinAbs(1.0, 1.e-12));
}

TEST_CASE("Composition parses fractional counts and rejects malformed formulae",
          "[composition]") {
  const Composition c({{"Mg1/2Fe0.5SiO3", 2.0}}, "molar");
  CHECK_THAT(amounts(c.get_atomic_composition()).at("Mg"),
             WithinAbs(1.0, 1.e-14));
  CHECK_THAT(
      utils::formula_mass(utils::dictionarize_formula("CH3COOH")),
      WithinRel(2.0 * 0.0120107 + 4.0 * 0.00100794 + 2.0 * 0.0159994, 1.e-14));
  for (const std::string formula :
       {"", "2MgO", "XxO", "MgO!", "Mg-1O", "Mg1/0O", "Vc"}) {
    CHECK_THROWS_AS(Composition({{formula, 1.0}}, "molar"),
                    std::invalid_argument);
  }
  CHECK_THROWS_AS(Composition({{"MgO", 1.0}, {"MgO", 2.0}}, "molar"),
                  std::invalid_argument);
  CHECK_THROWS_AS(Composition({{"MgO", 1.0}}, "atomic"), std::invalid_argument);
  CHECK_THROWS_AS(Composition({{"MgO", 0.0}}, "molar", true),
                  std::invalid_argument);
}

TEST_CASE("Formatting a composition does not renormalize it", "[composition]") {
  const Composition c({{"SiO2", 1.0}, {"MgO", 1.0}}, "molar");
  const auto original = c.get_mass_composition();
  CHECK(c.format("molar", 2, "total", 100.0) ==
        "Molar composition\nMgO: 50.00\nSiO2: 50.00\n");
  CHECK(c.get_mass_composition() == original);
}
