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

#ifndef BURNMAN_UTILS_TYPES_HPP_INCLUDED
#define BURNMAN_UTILS_TYPES_HPP_INCLUDED

#include "burnman/eos/components/excess_params.hpp"
#include "burnman/utils/types/mineral_params.hpp"
#include "burnman/utils/types/ndarray.hpp"
#include "burnman/utils/types/simple_types.hpp"

namespace burnman {
/**
 * @namespace burnman::types
 * @brief Custom types used in burnman
 *
 * TODO: docs on main user types - FormulaMap, MineralParams,
 * ExcessParams, etc. Also NDArray etc.
 */
namespace types {
// Aliases to expose excess_params types to user via burnman::types
using LandauParams = burnman::eos::excesses::LandauParams;
using LandauSLB2022Params = burnman::eos::excesses::LandauSLB2022Params;
using LandauHPParams = burnman::eos::excesses::LandauHPParams;
using LinearParams = burnman::eos::excesses::LinearParams;
using BraggWilliamsParams = burnman::eos::excesses::BraggWilliamsParams;
using MagneticChsParams = burnman::eos::excesses::MagneticChsParams;
using DebyeParams = burnman::eos::excesses::DebyeParams;
using DebyeDeltaParams = burnman::eos::excesses::DebyeDeltaParams;
using EinsteinParams = burnman::eos::excesses::EinsteinParams;
using EinsteinDeltaParams = burnman::eos::excesses::EinsteinDeltaParams;
using ExcessParamVector = burnman::eos::excesses::ExcessParamVector;
} // namespace types
} // namespace burnman

#endif // BURNMAN_UTILS_TYPES_HPP_INCLUDED
