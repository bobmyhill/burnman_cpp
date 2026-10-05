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

#include "burnman/tools/averaging/averaging_base.hpp"

namespace burnman::averaging {

double AveragingScheme::average_density(const Eigen::ArrayXd &volumes,
                                        const Eigen::ArrayXd &densities) const {
  return (volumes * densities).sum() / volumes.sum();
}

double AveragingScheme::average_thermal_expansivity(
    const Eigen::ArrayXd &volumes, const Eigen::ArrayXd &alphas) const {
  return (volumes * alphas).sum() / volumes.sum();
}

double
AveragingScheme::average_heat_capacity_v(const Eigen::ArrayXd &fractions,
                                         const Eigen::ArrayXd &c_v) const {
  return (fractions * c_v).sum();
}

double
AveragingScheme::average_heat_capacity_p(const Eigen::ArrayXd &fractions,
                                         const Eigen::ArrayXd &c_p) const {
  return (fractions * c_p).sum();
}

} // namespace burnman::averaging
