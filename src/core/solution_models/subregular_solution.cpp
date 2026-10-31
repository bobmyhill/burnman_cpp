// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/solution_models/subregular_solution.hpp"
#include "burnman/utils/constants.hpp"
#include <cmath>

namespace burnman::solution_models {

SubregularSolution::SubregularSolution(
    const types::PairedEndmemberList &endmember_list,
    const Interactions &energy_interaction,
    const Interactions &volume_interaction,
    const Interactions &entropy_interaction,
    const TernaryTerms &energy_ternary_terms,
    const TernaryTerms &volume_ternary_terms,
    const TernaryTerms &entropy_ternary_terms)
    : IdealSolution(endmember_list),
      Wijke(make_interaction_tensor(energy_interaction, energy_ternary_terms)),
      Wijks(make_interaction_tensor(entropy_interaction, entropy_ternary_terms,
                                    true)),
      Wijkv(make_interaction_tensor(volume_interaction, volume_ternary_terms,
                                    true)) {}

SubregularSolution::InteractionTensor
SubregularSolution::make_interaction_tensor(const Interactions &binary,
                                            const TernaryTerms &ternary,
                                            bool optional) const {
  const auto n = static_cast<std::size_t>(n_endmembers);
  InteractionTensor W({n, n, n});
  if (!optional || !binary.empty()) {
    if (binary.size() != n - 1)
      throw std::invalid_argument("Subregular interactions require n-1 rows of "
                                  "lengths n-1, n-2, ..., 1.");
    for (std::size_t i = 0; i < n - 1; ++i) {
      if (binary[i].size() != n - i - 1)
        throw std::invalid_argument(
            "Subregular interaction row has the wrong length.");
      for (std::size_t j = i + 1; j < n; ++j) {
        const auto &pair = binary[i][j - i - 1];
        if (!std::isfinite(pair[0]) || !std::isfinite(pair[1]))
          throw std::invalid_argument(
              "Subregular interactions must be finite.");
        const double w0 = pair[0] / 2., w1 = pair[1] / 2.;
        for (std::size_t k = 0; k < n; ++k) {
          W({k, i, j}) += w0;
          W({k, j, i}) += w1;
        }
        W({i, j, j}) += w0;
        W({j, i, i}) += w1;
        W({i, j, i}) -= w0;
        W({j, i, j}) -= w1;
      }
    }
  }
  for (const auto &term : ternary) {
    for (std::size_t i = 0; i < 3; ++i)
      if (!std::isfinite(term[i]) || term[i] < 0. ||
          term[i] >= static_cast<double>(n) || std::floor(term[i]) != term[i])
        throw std::invalid_argument(
            "Ternary indices must be integers within the endmember range.");
    if (!std::isfinite(term[3]))
      throw std::invalid_argument("Ternary interactions must be finite.");
    W({static_cast<std::size_t>(term[0]), static_cast<std::size_t>(term[1]),
       static_cast<std::size_t>(term[2])}) += term[3];
  }
  return W;
}

Eigen::ArrayXd SubregularSolution::compute_excess_partial_gibbs_free_energies(
    double P, double T, const Eigen::ArrayXd &p) const {
  return IdealSolution::compute_excess_partial_gibbs_free_energies(P, T, p) +
         compute_non_ideal_excess_partial_gibbs(P, T, p);
}

Eigen::ArrayXd SubregularSolution::compute_excess_partial_entropies(
    double P, double T, const Eigen::ArrayXd &p) const {
  return IdealSolution::compute_excess_partial_entropies(P, T, p) +
         compute_non_ideal_interactions(Wijks, p);
}

Eigen::ArrayXd SubregularSolution::compute_excess_partial_volumes(
    double, double, const Eigen::ArrayXd &p) const {
  return compute_non_ideal_interactions(Wijkv, p);
}

Eigen::MatrixXd
SubregularSolution::compute_gibbs_hessian(double P, double T,
                                          const Eigen::ArrayXd &p) const {
  return IdealSolution::compute_gibbs_hessian(P, T, p) +
         compute_non_ideal_hessian(Wijke, p) -
         T * compute_non_ideal_hessian(Wijks, p) +
         P * compute_non_ideal_hessian(Wijkv, p);
}

Eigen::MatrixXd
SubregularSolution::compute_entropy_hessian(double P, double T,
                                            const Eigen::ArrayXd &p) const {
  return IdealSolution::compute_entropy_hessian(P, T, p) +
         compute_non_ideal_hessian(Wijks, p);
}

Eigen::MatrixXd
SubregularSolution::compute_volume_hessian(double, double,
                                           const Eigen::ArrayXd &p) const {
  return compute_non_ideal_hessian(Wijkv, p);
}

Eigen::ArrayXd SubregularSolution::compute_activity_coefficients(
    double P, double T, const Eigen::ArrayXd &p) const {
  if (T <= 1.e-10)
    throw std::domain_error("Activity coefficients undefined at 0 K.");
  return (compute_non_ideal_excess_partial_gibbs(P, T, p) /
          (constants::physics::gas_constant * T))
      .exp();
}

Eigen::ArrayXd
SubregularSolution::compute_activities(double P, double T,
                                       const Eigen::ArrayXd &p) const {
  return IdealSolution::compute_activities(P, T, p) *
         compute_activity_coefficients(P, T, p);
}

Eigen::ArrayXd SubregularSolution::compute_non_ideal_interactions(
    const InteractionTensor &Wijk, const Eigen::ArrayXd &p) const {
  Eigen::ArrayXd gradient = Eigen::ArrayXd::Zero(n_endmembers);
  double value = 0.;
  std::size_t index = 0;
  for (Eigen::Index i = 0; i < n_endmembers; ++i)
    for (Eigen::Index j = 0; j < n_endmembers; ++j)
      for (Eigen::Index k = 0; k < n_endmembers; ++k) {
        const double w = Wijk[index++];
        value += w * p[i] * p[j] * p[k];
        gradient[i] += w * p[j] * p[k];
        gradient[j] += w * p[i] * p[k];
        gradient[k] += w * p[i] * p[j];
      }
  // Partial molar derivatives of N*f(n/N), with cubic f(p) = W_ijk p_i p_j p_k.
  return gradient - 2. * value;
}

Eigen::ArrayXd SubregularSolution::compute_non_ideal_excess_partial_gibbs(
    double P, double T, const Eigen::ArrayXd &p) const {
  return compute_non_ideal_interactions(Wijke, p) -
         T * compute_non_ideal_interactions(Wijks, p) +
         P * compute_non_ideal_interactions(Wijkv, p);
}

Eigen::MatrixXd
SubregularSolution::compute_non_ideal_hessian(const InteractionTensor &Wijk,
                                              const Eigen::ArrayXd &p) const {
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(n_endmembers, n_endmembers);
  std::size_t index = 0;
  for (Eigen::Index i = 0; i < n_endmembers; ++i)
    for (Eigen::Index j = 0; j < n_endmembers; ++j)
      for (Eigen::Index k = 0; k < n_endmembers; ++k) {
        const double w = Wijk[index++];
        hessian(i, j) += w * p[k];
        hessian(j, i) += w * p[k];
        hessian(i, k) += w * p[j];
        hessian(k, i) += w * p[j];
        hessian(j, k) += w * p[i];
        hessian(k, j) += w * p[i];
      }
  // Project composition derivatives onto changes in endmember mole amounts.
  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n_endmembers, n_endmembers);
  J.colwise() -= p.matrix();
  return (J.transpose() * hessian).eval() * J;
}

} // namespace burnman::solution_models
