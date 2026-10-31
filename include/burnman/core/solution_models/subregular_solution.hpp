// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once

#include "burnman/core/solution_models/ideal_solution.hpp"
#include "burnman/utils/types/ndarray.hpp"
#include <array>

namespace burnman::solution_models {

/// Helffrich and Wood (1989) subregular mixing with ideal site entropy.
/// Binary rows contain [W_ij, W_ji] pairs for j > i. Ternary terms are
/// [i, j, k, value]. Energy, entropy and volume coefficients are in SI units.
/// The cubic tensors retain the multicomponent terms induced by binary pairs,
/// following Python BurnMan's formulation.
class SubregularSolution : public IdealSolution {
public:
  using Interactions = std::vector<std::vector<std::array<double, 2>>>;
  using TernaryTerms = std::vector<std::array<double, 4>>;
  using InteractionTensor = types::NDArray<double>;

  SubregularSolution(const types::PairedEndmemberList &endmember_list,
                     const Interactions &energy_interaction,
                     const Interactions &volume_interaction = {},
                     const Interactions &entropy_interaction = {},
                     const TernaryTerms &energy_ternary_terms = {},
                     const TernaryTerms &volume_ternary_terms = {},
                     const TernaryTerms &entropy_ternary_terms = {});

  std::shared_ptr<SolutionModel> clone() const override {
    return std::make_shared<SubregularSolution>(*this);
  }
  const InteractionTensor &get_energy_interactions() const { return Wijke; }
  const InteractionTensor &get_entropy_interactions() const { return Wijks; }
  const InteractionTensor &get_volume_interactions() const { return Wijkv; }

  Eigen::ArrayXd compute_excess_partial_gibbs_free_energies(
      double, double, const Eigen::ArrayXd &) const override;
  Eigen::ArrayXd
  compute_excess_partial_entropies(double, double,
                                   const Eigen::ArrayXd &) const override;
  Eigen::ArrayXd
  compute_excess_partial_volumes(double, double,
                                 const Eigen::ArrayXd &) const override;
  Eigen::MatrixXd compute_gibbs_hessian(double, double,
                                        const Eigen::ArrayXd &) const override;
  Eigen::MatrixXd
  compute_entropy_hessian(double, double,
                          const Eigen::ArrayXd &) const override;
  Eigen::MatrixXd compute_volume_hessian(double, double,
                                         const Eigen::ArrayXd &) const override;
  Eigen::ArrayXd
  compute_activity_coefficients(double, double,
                                const Eigen::ArrayXd &) const override;
  Eigen::ArrayXd compute_activities(double, double,
                                    const Eigen::ArrayXd &) const override;

private:
  InteractionTensor Wijke, Wijks, Wijkv;

  InteractionTensor make_interaction_tensor(const Interactions &binary,
                                            const TernaryTerms &ternary,
                                            bool optional = false) const;
  Eigen::ArrayXd compute_non_ideal_interactions(const InteractionTensor &,
                                                const Eigen::ArrayXd &) const;
  Eigen::ArrayXd
  compute_non_ideal_excess_partial_gibbs(double, double,
                                         const Eigen::ArrayXd &) const;
  Eigen::MatrixXd compute_non_ideal_hessian(const InteractionTensor &,
                                            const Eigen::ArrayXd &) const;
};

} // namespace burnman::solution_models
