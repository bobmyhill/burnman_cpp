// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/solution_models/mechanical_solution.hpp"

namespace burnman::solution_models {

Eigen::ArrayXd MechanicalSolution::compute_excess_partial_gibbs_free_energies(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::ArrayXd::Zero(molar_fractions.size());
}

Eigen::ArrayXd MechanicalSolution::compute_excess_partial_volumes(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::ArrayXd::Zero(molar_fractions.size());
}

Eigen::ArrayXd MechanicalSolution::compute_excess_partial_entropies(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::ArrayXd::Zero(molar_fractions.size());
}

Eigen::ArrayXd MechanicalSolution::compute_activities(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::ArrayXd::Ones(molar_fractions.size());
}

Eigen::ArrayXd MechanicalSolution::compute_activity_coefficients(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::ArrayXd::Ones(molar_fractions.size());
}

Eigen::MatrixXd MechanicalSolution::compute_gibbs_hessian(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::MatrixXd::Zero(molar_fractions.size(), molar_fractions.size());
}

Eigen::MatrixXd MechanicalSolution::compute_entropy_hessian(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::MatrixXd::Zero(molar_fractions.size(), molar_fractions.size());
}

Eigen::MatrixXd MechanicalSolution::compute_volume_hessian(
    double, double, const Eigen::ArrayXd &molar_fractions) const {
  return Eigen::MatrixXd::Zero(molar_fractions.size(), molar_fractions.size());
}

} // namespace burnman::solution_models
