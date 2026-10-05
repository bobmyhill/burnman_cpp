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

#include "burnman/tools/averaging/hs_lower.hpp"
#include "burnman/tools/averaging/averaging_utils.hpp"

namespace burnman::averaging {

double HashinShtrikmanLower::average_bulk_moduli(
    const Eigen::ArrayXd &volumes, const Eigen::ArrayXd &bulk_moduli,
    const Eigen::ArrayXd &shear_moduli) const {
  return utils::lower_hs_bulk_fn(volumes, bulk_moduli, shear_moduli);
}

double HashinShtrikmanLower::average_shear_moduli(
    const Eigen::ArrayXd &volumes, const Eigen::ArrayXd &bulk_moduli,
    const Eigen::ArrayXd &shear_moduli) const {
  return utils::lower_hs_shear_fn(volumes, bulk_moduli, shear_moduli);
}

} // namespace burnman::averaging
