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

#ifndef BURNMAN_CORE_RELAXED_ANISOTROPIC_SOLUTION_HPP_INCLUDED
#define BURNMAN_CORE_RELAXED_ANISOTROPIC_SOLUTION_HPP_INCLUDED

#include "burnman/core/anisotropic_solution.hpp"

namespace burnman {
/// Minimize rapid isochemical modes, then relax the strain/temperature Hessian.
/// Tensor susceptibilities describe an interior structural minimum.
class RelaxedAnisotropicSolution : public AnisotropicSolution {
public:
  RelaxedAnisotropicSolution(std::shared_ptr<AnisotropicSolution> solution,
                             const Eigen::MatrixXd &relaxation_vectors,
                             const Eigen::MatrixXd &unrelaxed_vectors);
  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;
  void set_composition(const Eigen::ArrayXd &fractions) override;
  void set_composition(const Eigen::ArrayXd &fractions,
                       const std::optional<Eigen::VectorXd> &q_initial,
                       bool relaxed = true);
  void set_state(double pressure, double temperature) override;
  void set_state(double pressure, double temperature, bool relaxed);
  void set_state_with_volume(double volume, double temperature,
                             const Eigen::Vector2d &pressure_guesses =
                                 Eigen::Vector2d(0., 1.e10)) override;
  void set_method(types::EOSType method) override;
  void set_method(std::shared_ptr<EquationOfState> method) override;
  void relax_at_PTX();

  const std::shared_ptr<AnisotropicSolution> &get_unrelaxed() const;
  const Eigen::MatrixXd &get_dndq() const;
  const Eigen::MatrixXd &get_dndx() const;
  const Eigen::ArrayXd &get_unrelaxed_vectors() const;
  const Eigen::VectorXd &get_q_initial() const;
  void set_q_initial(const Eigen::VectorXd &q);
  const CompositionTensors &get_depsdq_fixed_VT() const;
  const Eigen::Matrix3d &get_depsdT_fixed_volume() const;
  types::ElasticTensor get_dbarepsdeps() const;
  const Eigen::VectorXd &get_dSdq() const;
  const Eigen::VectorXd &get_dPdq() const;
  const Eigen::MatrixXd &get_structural_hessian_fixed_VT() const;
  const Eigen::MatrixXd &get_structural_hessian_fixed_epsT() const;
  const Eigen::MatrixXd &get_susceptibility() const;
  const Eigen::MatrixXd &get_coupling() const;
  const Eigen::MatrixXd &get_dqdz_relaxed() const;
  const types::Matrix7d &get_frozen_hessian() const;
  const types::Matrix7d &get_relaxed_hessian() const;

  const types::Matrix6d &get_isothermal_stiffness_tensor() const override;
  const types::Matrix6d &get_isothermal_compliance_tensor() const override;
  const Eigen::Matrix3d &get_thermal_expansivity_tensor() const override;
  Eigen::Matrix3d get_thermal_stress_tensor() const override;
  double get_molar_isometric_heat_capacity() const override;

protected:
  double compute_isothermal_bulk_modulus_reuss() const override;
  double compute_isentropic_bulk_modulus_reuss() const override;
  double compute_isothermal_compressibility_reuss() const override;
  double compute_isentropic_compressibility_reuss() const override;
  double compute_thermal_expansivity() const override;
  double compute_molar_heat_capacity_p() const override;

private:
  struct Responses {
    CompositionTensors depsdq;
    Eigen::Matrix3d depsdT, pi, alpha;
    Eigen::VectorXd dSdq, dPdq;
    Eigen::MatrixXd structural_VT, structural, susceptibility, coupling, dqdz;
    types::Matrix7d frozen, relaxed;
    types::Matrix6d stiffness, compliance;
    double isometric_heat_capacity;
  };
  const Responses &responses() const;
  void require_composition() const;
  void minimize(bool fixed_volume, double volume, double pressure,
                double temperature, const Eigen::Vector2d &pressure_guesses);
  void synchronize(const Eigen::ArrayXd &fractions, double pressure,
                   double temperature);
  std::shared_ptr<AnisotropicSolution> unrelaxed_;
  Eigen::MatrixXd dndq_, dndx_, inverse_basis_;
  Eigen::ArrayXd independent_;
  Eigen::VectorXd q_initial_;
  mutable std::optional<Responses> responses_;
};
} // namespace burnman

#endif
