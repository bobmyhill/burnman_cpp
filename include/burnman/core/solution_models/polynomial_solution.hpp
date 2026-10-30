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

#pragma once
#include "burnman/core/solution_models/ideal_solution.hpp"
#include <optional>

namespace burnman::solution_models {
/// Ideal configurational entropy plus polynomials in transformed proportions.
/// Interaction coefficients may include the thermodynamics of extra minerals.
class PolynomialSolution : public IdealSolution {
public:
  using Interactions = std::vector<std::vector<double>>;
  PolynomialSolution(
      const types::PairedEndmemberList &endmembers,
      const Interactions &ESV_interactions = {},
      const std::vector<std::shared_ptr<Mineral>> &interaction_endmembers = {},
      const Interactions &endmember_coefficients_and_interactions = {},
      const std::optional<Eigen::MatrixXd> &transformation_matrix =
          std::nullopt);
  std::shared_ptr<SolutionModel> clone() const override;
  const std::vector<std::shared_ptr<Mineral>> &
  get_interaction_endmembers() const;
  const Eigen::MatrixXd &get_dqdp() const;
  const Interactions &get_ESV_interactions() const;
  const Interactions &get_endmember_coefficients_and_interactions() const;

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
  double compute_Cp_excess(double, double,
                           const Eigen::ArrayXd &) const override;
  double compute_alphaV_excess(double, double,
                               const Eigen::ArrayXd &) const override;
  double compute_VoverKT_excess(double, double,
                                const Eigen::ArrayXd &) const override;

private:
  enum class Property { Gibbs, Entropy, Volume, Cp, AlphaV, VoverKT };
  struct Term {
    Eigen::VectorXd coefficients, exponents;
    bool mineral;
  };
  struct Derivatives {
    double value;
    Eigen::ArrayXd gradient;
    Eigen::MatrixXd hessian;
  };
  Derivatives evaluate(double, double, const Eigen::ArrayXd &, Property) const;
  std::vector<Term> terms_;
  std::vector<std::shared_ptr<Mineral>> interaction_endmembers_;
  Eigen::MatrixXd dqdp_;
  Interactions ESV_interactions_, endmember_coefficients_and_interactions_;
};
} // namespace burnman::solution_models
