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

#include "burnman/utils/chemistry_utils.hpp"
#include "burnman/utils/string_utils.hpp"
#include <cmath>
#include <regex>

namespace burnman::utils {

types::FormulaMap
dictionarize_formula(const std::string &formula,
                     std::vector<std::string> *element_order) {
  static const std::string number =
      R"((?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)";
  static const std::regex token("([A-Z][a-z]*)(" + number + "(?:/" + number +
                                ")?)?");
  if (formula.empty())
    throw std::invalid_argument("Chemical formula must not be empty.");
  types::FormulaMap result;
  std::vector<std::string> order;
  auto position = formula.cbegin();
  while (position != formula.cend()) {
    std::match_results<std::string::const_iterator> match;
    if (!std::regex_search(position, formula.cend(), match, token,
                           std::regex_constants::match_continuous)) {
      throw std::invalid_argument("Malformed chemical formula: " + formula);
    }
    const std::string element = match[1].str();
    if (constants::chemistry::atomic_masses.count(element) == 0) {
      throw std::invalid_argument("Unknown element '" + element +
                                  "' in formula: " + formula);
    }
    double atoms = 1.0;
    try {
      if (match[2].matched)
        atoms = utils::stod(match[2].str());
    } catch (const std::out_of_range &) {
      throw std::invalid_argument(
          "Atom count is outside the supported range in formula: " + formula);
    }
    if (!std::isfinite(atoms))
      throw std::invalid_argument("Nonfinite atom count in formula: " +
                                  formula);
    if (result.count(element) == 0)
      order.push_back(element);
    result[element] += atoms;
    if (!std::isfinite(result[element]))
      throw std::invalid_argument("Atom count overflow in formula: " + formula);
    position = match[0].second;
  }
  if (element_order != nullptr)
    *element_order = std::move(order);
  return result;
}

double formula_mass(const types::FormulaMap &formula) {
  double mass = 0.0;
  for (const auto &[element, atoms] : formula) {
    const auto it = constants::chemistry::atomic_masses.find(element);
    if (it == constants::chemistry::atomic_masses.end()) {
      throw std::invalid_argument("No atomic mass for element: " + element);
    }
    if (!std::isfinite(atoms) || atoms < 0.0) {
      throw std::invalid_argument(
          "Formula atom counts must be finite and nonnegative.");
    }
    mass += atoms * it->second;
  }
  if (!std::isfinite(mass))
    throw std::invalid_argument("Formula mass is not finite.");
  return mass;
}

} // namespace burnman::utils
