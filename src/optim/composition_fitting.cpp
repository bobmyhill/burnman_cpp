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

#include "burnman/optim/composition_fitting.hpp"

namespace burnman::optim {
LeastSquaresResult fit_phase_proportions_to_bulk_composition(
    const Eigen::MatrixXd &phase_compositions,
    const Eigen::VectorXd &bulk_composition) {
  const auto n_phases = phase_compositions.cols();
  const LinearConstraints inequality_constraints{
      -Eigen::MatrixXd::Identity(n_phases, n_phases),
      Eigen::VectorXd::Zero(n_phases)};
  return weighted_constrained_least_squares(
      phase_compositions, bulk_composition, std::nullopt, std::nullopt,
      inequality_constraints);
}
} // namespace burnman::optim
