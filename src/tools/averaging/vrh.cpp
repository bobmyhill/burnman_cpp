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

#include "burnman/tools/averaging/vrh.hpp"
#include "burnman/tools/averaging/averaging_utils.hpp"

namespace burnman::averaging {

double VoigtReussHill::average_bulk_moduli(const Eigen::ArrayXd &volumes,
                                           const Eigen::ArrayXd &bulk_moduli,
                                           const Eigen::ArrayXd &shear_moduli
                                           [[maybe_unused]]) const {
  return utils::voigt_reuss_hill_fn(volumes, bulk_moduli);
}

double
VoigtReussHill::average_shear_moduli(const Eigen::ArrayXd &volumes,
                                     const Eigen::ArrayXd &bulk_moduli
                                     [[maybe_unused]],
                                     const Eigen::ArrayXd &shear_moduli) const {
  return utils::voigt_reuss_hill_fn(volumes, shear_moduli);
}

} // namespace burnman::averaging
