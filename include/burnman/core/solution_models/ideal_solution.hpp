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

#ifndef BURNMAN_CORE_SOLUTION_MODELS_IDEAL_HPP_INCLUDED
#define BURNMAN_CORE_SOLUTION_MODELS_IDEAL_HPP_INCLUDED

#include "burnman/core/solution_models/solution_model_base.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include <Eigen/Dense>

namespace burnman {
namespace solution_models {

/**
 * @class IdealSolution
 * @brief Ideal solution model.
 *
 * Derived from SolutionModel.
 * Calculates the excess gibbs free energy and etropy due to configurational
 * entropy. Excess internal energy and volume are equal to zero.
 *
 * The multiplicity of each type of site in the structure is allowed to change
 * linearly as a function of endmember proportions. This makes the model
 * equivalent to the entropic part of a Temkin-type model (Temkin, 1945).
 *
 */
class IdealSolution : public SolutionModel {

public:
  // Extend constructor
  IdealSolution(const types::PairedEndmemberList &endmember_list);
  std::shared_ptr<SolutionModel> clone() const override {
    return std::make_shared<IdealSolution>(*this);
  }

  // Public functions overriden from base class
  Eigen::ArrayXd compute_excess_partial_gibbs_free_energies(
      double pressure, double temperature,
      const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::ArrayXd compute_excess_partial_entropies(
      double pressure, double temperature,
      const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::ArrayXd compute_excess_partial_volumes(
      double pressure, double temperature,
      const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::ArrayXd
  compute_activities(double pressure, double temperature,
                     const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::ArrayXd compute_activity_coefficients(
      double pressure, double temperature,
      const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::MatrixXd
  compute_gibbs_hessian(double pressure, double temperature,
                        const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::MatrixXd
  compute_entropy_hessian(double pressure, double temperature,
                          const Eigen::ArrayXd &molar_fractions) const override;

  Eigen::MatrixXd
  compute_volume_hessian(double pressure, double temperature,
                         const Eigen::ArrayXd &molar_fractions) const override;

private:
  // Public or protected member variable?
  Eigen::ArrayXd endmember_configurational_entropies;
  Eigen::ArrayXd compute_endmember_configurational_entropies() const;

  // This is unused in python?
  double
  compute_configurational_entropy(const Eigen::ArrayXd &molar_fractions) const;

  Eigen::ArrayXd compute_ideal_excess_partial_gibbs(
      double temperature, const Eigen::ArrayXd &molar_fractions) const;

  Eigen::ArrayXd compute_ideal_excess_partial_entropies(
      const Eigen::ArrayXd &molar_fractions) const;

  Eigen::ArrayXd
  compute_ideal_activities(const Eigen::ArrayXd &molar_fractions) const;

  Eigen::ArrayXd
  compute_log_ideal_activities(const Eigen::ArrayXd &molar_fractions) const;

  Eigen::MatrixXd compute_log_ideal_activity_derivatives(
      const Eigen::ArrayXd &molar_fractions) const;

  Eigen::MatrixXd
  compute_ideal_entropy_hessian(const Eigen::ArrayXd &molar_fractions) const;

  // Want to make and cache, ones, eyeones, eye
  // Not really used --> would make protected and create in setup
};

} // namespace solution_models
} // namespace burnman

#endif // BURNMAN_CORE_SOLUTION_MODELS_IDEAL_HPP_INCLUDED
