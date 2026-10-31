// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once

#include "burnman/core/solution_models/solution_model_base.hpp"

namespace burnman::solution_models {

/**
 * Linear mixing of endmember Gibbs energies, without interactions or
 * configurational entropy. Activities and activity coefficients are unity.
 * Endmembers need no common crystal structure; site formulae are labels only.
 * As in Python BurnMan, Solution accepts unnormalised and signed coefficients
 * with this model, allowing linear combinations of endmember formula units.
 */
class MechanicalSolution : public SolutionModel {
public:
  explicit MechanicalSolution(const types::PairedEndmemberList &endmember_list)
      : SolutionModel(endmember_list, false) {}

  std::shared_ptr<SolutionModel> clone() const override {
    return std::make_shared<MechanicalSolution>(*this);
  }

  Eigen::ArrayXd compute_excess_partial_gibbs_free_energies(
      double, double, const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::ArrayXd compute_excess_partial_volumes(
      double, double, const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::ArrayXd compute_excess_partial_entropies(
      double, double, const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::ArrayXd
  compute_activities(double, double,
                     const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::ArrayXd compute_activity_coefficients(
      double, double, const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::MatrixXd
  compute_gibbs_hessian(double, double,
                        const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::MatrixXd
  compute_entropy_hessian(double, double,
                          const Eigen::ArrayXd &molar_fractions) const override;
  Eigen::MatrixXd
  compute_volume_hessian(double, double,
                         const Eigen::ArrayXd &molar_fractions) const override;
};

} // namespace burnman::solution_models
