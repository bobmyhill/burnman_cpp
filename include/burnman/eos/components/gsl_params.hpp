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

#ifndef BURNMAN_EOS_TYPES_GSL_PARAMS_HPP_INCLUDED
#define BURNMAN_EOS_TYPES_GSL_PARAMS_HPP_INCLUDED

#include "burnman/utils/types/mineral_params.hpp"

namespace burnman {
namespace eos {

/**
 * @namespace burnman::eos::gsl_params
 * @brief Structs to hols parameters when making GSL function objects
 */
namespace gsl_params {
/**
 * Struct for GSL Brent root finding
 * Used for volume finding in EOS where only pressure needed
 * as an additional argument:
 *   bm, bm4, vinet, macaw, morse_potential, spock
 */
struct SolverParams_P {
  const types::MineralParams &params;
  double pressure;
};
/**
 * Struct for GSL Brent root finding
 * Used for volume finding in EOS where pressure and
 * temperature needed as additional arguments:
 *   mgd3
 */
struct SolverParams_PT {
  const types::MineralParams &params;
  double pressure;
  double temperature;
};
/**
 * Struct for GSL root finding
 * Used for volume finding in EOS where
 * P, T needed along with SLB specific params.
 *  SLB2, SLB3, etc.
 */
struct SolverParams_SLB {
  const types::MineralParams &params;
  double pressure;
  double temperature;
  double a1_ii, a2_iikk;
  double b_iikk, b_iikkmm;
  double bel_0, gel;
};
/**
 * Struct for GSL Brent root finding
 * Used in Bragg Williams excess function to find
 * Q in Gibbs calculation.
 */
struct BWReactParams {
  double delta_H, temperature, W;
  int n;
  double f_0, f_1;
};

} // namespace gsl_params
} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_TYPES_GSL_PARAMS_HPP_INCLUDED
