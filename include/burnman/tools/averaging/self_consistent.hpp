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

#ifndef BURNMAN_TOOLS_AVERAGING_SELF_CONSISTENT_HPP_INCLUDED
#define BURNMAN_TOOLS_AVERAGING_SELF_CONSISTENT_HPP_INCLUDED

#include "burnman/tools/averaging/averaging_base.hpp"

namespace burnman::averaging {

/**
 * @class SelfConsistent
 * @brief Spherical self-consistent elastic moduli for scalar phases.
 *
 * Each phase is represented by its bulk and shear moduli and by spherical
 * grains embedded in the effective isotropic medium. All phases are treated
 * equally. The coupled equations of Budiansky (1965),
 * https://doi.org/10.1016/0022-5096(65)90011-6, are solved together by
 * utils::self_consistent_fn().
 *
 * Nonnegative phase moduli are supported, including fluids and empty pores.
 * Zero-volume phases are ignored. Phase volumes need not sum to one.
 */
class SelfConsistent : public AveragingScheme {
public:
  double average_bulk_moduli(const Eigen::ArrayXd &volumes,
                             const Eigen::ArrayXd &bulk_moduli,
                             const Eigen::ArrayXd &shear_moduli) const override;

  double
  average_shear_moduli(const Eigen::ArrayXd &volumes,
                       const Eigen::ArrayXd &bulk_moduli,
                       const Eigen::ArrayXd &shear_moduli) const override;
};

} // namespace burnman::averaging

#endif // BURNMAN_TOOLS_AVERAGING_SELF_CONSISTENT_HPP_INCLUDED
