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

#ifndef BURNMAN_UTILS_TYPES_SIMPLE_TYPES_HPP_INCLUDED
#define BURNMAN_UTILS_TYPES_SIMPLE_TYPES_HPP_INCLUDED

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace burnman {

/**
 * Forward declarations
 */
class Mineral;

namespace types {

/**
 * Custom type alias for chemical formulae.
 */
using FormulaMap = std::unordered_map<std::string, double>;

/**
 * Enum used to define EOS Type
 */
enum class EOSType {
  Auto,   // Used to set EOS from params
  Custom, // Used when user passes custom EOSType
  Vinet,
  MT,
  BM3,
  BM2,
  MGD2,
  MGD3,
  HPTMT,
  HPTMTL,
  SLB2,
  SLB3,
  SLB3Conductive,
  SLB3Stishovite
};

/**
 * Enum used to define averaging scheme type
 */
enum class AveragingType {
  Voigt,
  Reuss,
  VRH,
  HashinShtrikmanLower,
  HashinShtrikmanUpper,
  HashinShtrikman
};

/**
 * Enum used to define fraction type
 */
enum class FractionType { Molar, Mass, Volume };

/**
 * Type aliases for constructing solid solution models.
 */
using MineralFormulaPair = std::pair<Mineral, std::string>;
using PairedEndmemberList = std::vector<MineralFormulaPair>;

} // namespace types

// Overloaded FormulaMap operators to add and weight formulae
// In namespace burnman so they can be found
inline types::FormulaMap operator+(const types::FormulaMap &a,
                                   const types::FormulaMap &b) {
  types::FormulaMap result = a;
  for (const auto &[elem, count] : b) {
    result[elem] += count;
  }
  return result;
}

inline types::FormulaMap &operator+=(types::FormulaMap &a,
                                     const types::FormulaMap &b) {
  for (const auto &[elem, count] : b) {
    a[elem] += count;
  }
  return a;
}

inline types::FormulaMap operator*(const types::FormulaMap &a, double scalar) {
  types::FormulaMap result;
  for (const auto &[elem, count] : a) {
    result[elem] = scalar * count;
  }
  return result;
}

inline types::FormulaMap operator*(double scalar, const types::FormulaMap &a) {
  return a * scalar;
}

inline types::FormulaMap &operator*=(types::FormulaMap &a, double scalar) {
  for (auto &[_, count] : a) {
    count *= scalar;
  }
  return a;
}

} // namespace burnman

#endif // BURNMAN_UTILS_TYPES_SIMPLE_TYPES_HPP_INCLUDED
