// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/assemblage.hpp"

namespace burnman {
/// Equilibrate a fixed assemblage, then relax Gibbs derivatives along selected
/// isochemical reactions. Reaction vectors are rows in endmember order, as in
/// Python BurnMan's RelaxedComposite; they do not restrict the equilibrium
/// solve.
class RelaxedComposite : public Assemblage {
public:
  RelaxedComposite(std::shared_ptr<Assemblage> composite,
                   const Eigen::MatrixXd &relaxation_vectors);
  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;
  void set_state(double pressure, double temperature) override;
  void set_state(double pressure, double temperature,
                 const types::FormulaMap &bulk_composition);
  void copy_state_from_unrelaxed();
  void clear_state() { Material::clear_state(); }

  std::shared_ptr<Assemblage> get_unrelaxed() const { return unrelaxed_; }
  Eigen::Index get_n_relaxation_vectors() const { return dndq_.cols(); }
  const Eigen::MatrixXd &get_dndq() const { return dndq_; }
  const Eigen::MatrixXd &get_dqdz_relaxed() const;
  const Eigen::MatrixXd &get_structural_hessian() const;
  const Eigen::MatrixXd &get_susceptibility() const;
  const Eigen::MatrixXd &get_coupling() const;
  const Eigen::Matrix2d &get_frozen_hessian() const;
  const Eigen::Matrix2d &get_relaxed_hessian() const;

protected:
  double compute_isothermal_bulk_modulus_reuss() const override;
  double compute_isothermal_compressibility_reuss() const override;
  double compute_thermal_expansivity() const override;
  double compute_molar_heat_capacity_p() const override;

private:
  std::shared_ptr<Assemblage> unrelaxed_;
  Eigen::MatrixXd dndq_;
  struct Derivatives {
    Eigen::MatrixXd structural, susceptibility, coupling, dqdz;
    Eigen::Matrix2d frozen, relaxed;
  };
  mutable std::optional<Derivatives> derivatives_;
  const Derivatives &derivatives() const;
};
} // namespace burnman
