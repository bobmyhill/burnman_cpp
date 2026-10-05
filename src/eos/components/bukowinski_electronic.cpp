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

#include "burnman/eos/components/bukowinski_electronic.hpp"
#include <cmath>

namespace burnman::eos {

double bukowinski::compute_helmholtz_el(double temperature, double volume,
                                        const types::MineralParams &params) {
  return -0.5 * (*params.bel_0) * std::pow(volume / *params.V_0, *params.gel) *
         (temperature * temperature - (*params.T_0) * (*params.T_0));
}

double bukowinski::compute_pressure_el(double temperature, double volume,
                                       const types::MineralParams &params) {
  return 0.5 * (*params.gel) * (*params.bel_0) *
         std::pow(volume / *params.V_0, *params.gel) *
         (temperature * temperature - (*params.T_0) * (*params.T_0)) / volume;
}

double bukowinski::compute_entropy_el(double temperature, double volume,
                                      const types::MineralParams &params) {
  return *params.bel_0 * temperature *
         std::pow(volume / *params.V_0, *params.gel);
}

double bukowinski::compute_KT_over_V(double temperature, double volume,
                                     const types::MineralParams &params) {
  return -(*params.gel - 1.0) *
         compute_pressure_el(temperature, volume, params) / volume;
}

double bukowinski::compute_CV_over_T(double volume,
                                     const types::MineralParams &params) {
  return *params.bel_0 * std::pow(volume / *params.V_0, *params.gel);
}

double bukowinski::compute_alpha_KT(double temperature, double volume,
                                    const types::MineralParams &params) {
  return *params.gel * (*params.bel_0) * temperature *
         std::pow(volume / *params.V_0, *params.gel) / volume;
}

} // namespace burnman::eos
