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

#ifndef BURNMAN_TOOLS_AVERAGING_HS_UPPER_HPP_INCLUDED
#define BURNMAN_TOOLS_AVERAGING_HS_UPPER_HPP_INCLUDED

#include "burnman/tools/averaging/averaging_base.hpp"

namespace burnman {
namespace averaging {

/**
 * @class HashinShtrikmanUpper
 * @brief Upper Hashin-Shtrikman bound.
 *
 * Class for computing the upper Hashin-Shtrikman bound for
 * elastic properties.

 * Overrides bulk and shear moduli averaging in `AveragingScheme'.
 * Uses formulae from Watt, 1976.
 *
 * @note The Hashin-Shtrikman bounds are tighter than the Voigt and Reuss
 * bounds because they make the additional assumption that the orientation
 * of the phases are statistically isotropic. In some cases this may be a
 * good assumption, and in others it may not be.
 */
class HashinShtrikmanUpper : public AveragingScheme {
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

#endif // BURNMAN_TOOLS_AVERAGING_HS_UPPER_HPP_INCLUDED
