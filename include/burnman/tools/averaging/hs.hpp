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

#ifndef BURNMAN_TOOLS_AVERAGING_HS_HPP_INCLUDED
#define BURNMAN_TOOLS_AVERAGING_HS_HPP_INCLUDED

#include "burnman/tools/averaging/averaging_base.hpp"

namespace burnman {
namespace averaging {

/**
 * @class HashinShtrikman
 * @brief Hashin-Shtrikman averaging scheme.
 *
 * Class for computing the Hashin-Shtrikman average for elastic properties.
 * Defined as arithmetic mean of upper and lower bounds.
 * Overrides bulk and shear moduli averaging in `AveragingScheme'.
 */
class HashinShtrikman : public AveragingScheme {
public:
  double average_bulk_moduli(const Eigen::ArrayXd &volumes,
                             const Eigen::ArrayXd &bulk_moduli,
                             const Eigen::ArrayXd &shear_moduli) const override;

  double
  average_shear_moduli(const Eigen::ArrayXd &volumes,
                       const Eigen::ArrayXd &bulk_moduli,
                       const Eigen::ArrayXd &shear_moduli) const override;
};

} // namespace averaging
} // namespace burnman

#endif // BURNMAN_TOOLS_AVERAGING_HS_HPP_INCLUDED
