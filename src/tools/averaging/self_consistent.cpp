// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan.
// Detailed license information is in the LICENSE document.
//
// ------------------------------------------------------

#include "burnman/tools/averaging/self_consistent.hpp"
#include "burnman/tools/averaging/averaging_utils.hpp"

namespace burnman::averaging {

double
SelfConsistent::average_bulk_moduli(const Eigen::ArrayXd &volumes,
                                    const Eigen::ArrayXd &bulk_moduli,
                                    const Eigen::ArrayXd &shear_moduli) const {
  return utils::self_consistent_fn(volumes, bulk_moduli, shear_moduli).first;
}

double
SelfConsistent::average_shear_moduli(const Eigen::ArrayXd &volumes,
                                     const Eigen::ArrayXd &bulk_moduli,
                                     const Eigen::ArrayXd &shear_moduli) const {
  return utils::self_consistent_fn(volumes, bulk_moduli, shear_moduli).second;
}

} // namespace burnman::averaging
