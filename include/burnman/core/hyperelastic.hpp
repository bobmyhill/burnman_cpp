// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan. See LICENSE for details.
//
// ------------------------------------------------------

#pragma once
#include "burnman/core/anisotropic_mineral.hpp"

namespace burnman {
enum class OutputFrame { crystal = 1, current = 2 };

/// Finite-strain Helmholtz model: F_hyd(V,T) + V/2 H:C_hyd:H.
/// H is the right isochoric Hencky strain from the hydrostatic cell.
/// Tangents are Cauchy-stress derivatives, with minor but not major symmetry.
class HyperelasticMineral : public AnisotropicMineral {
public:
  explicit HyperelasticMineral(std::shared_ptr<AnisotropicMineral> hydrostatic);
  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;
  void set_state(double, double) override;
  void set_state_with_volume(
      double, double,
      const Eigen::Vector2d & = Eigen::Vector2d(0., 1.e10)) override;
  void set_state_with_molar_cell_vectors(const Eigen::Matrix3d &cell_vectors,
                                         double T);
  void set_method(types::EOSType method) override;
  void set_method(std::shared_ptr<EquationOfState> method) override;
  void set_output_frame(OutputFrame frame);
  void set_output_frame(const std::string &frame);
  OutputFrame get_output_frame() const;
  const std::shared_ptr<AnisotropicMineral> &get_hydrostatic() const;
  const Eigen::Matrix3d &get_current_cell_vectors() const;
  const Eigen::Matrix3d &get_rotation_to_crystallographic_frame() const;
  const Eigen::Matrix3d &get_rotation_matrix() const override;
  Eigen::Matrix3d get_unrotated_cell_vectors() const override;
  Eigen::Matrix3d get_cell_vectors() const override;
  const Eigen::Matrix3d &get_deformation_gradient_tensor() const override;
  utils::CellParameters get_cell_parameters() const override;
  const Eigen::Matrix3d &get_hencky_strain() const;
  double get_helmholtz_elastic() const;
  const Eigen::Matrix3d &get_cauchy_stress_unrotated() const;
  const Eigen::Matrix3d &get_cauchy_stress_tensor() const;
  double get_pressure() const override;
  const types::Matrix6d &
  get_unrotated_isothermal_stiffness_tensor() const override;
  const types::Matrix6d &get_isothermal_stiffness_tensor() const override;
  const types::Matrix6d &get_isothermal_compliance_tensor() const override;
  const types::Matrix6d &get_isentropic_stiffness_tensor() const override;
  const types::Matrix6d &get_isentropic_compliance_tensor() const override;
  const Eigen::Matrix3d &get_thermal_stress_unrotated() const;
  Eigen::Matrix3d get_thermal_stress_tensor() const override;
  const Eigen::Matrix3d &get_thermal_expansivity_tensor() const override;
  double get_molar_isometric_heat_capacity() const override;
  double get_molar_isostress_heat_capacity() const;

protected:
  std::optional<Eigen::Matrix3d> get_cauchy_stress() const override;
  PsiState compute_psi_state() const override;
  double compute_molar_volume() const override;
  double compute_molar_volume_unmodified() const override;
  double compute_molar_mass() const override;
  double compute_molar_helmholtz() const override;
  double compute_molar_entropy() const override;
  double compute_molar_internal_energy() const override;
  double compute_molar_gibbs() const override;
  double compute_molar_enthalpy() const override;
  double compute_isothermal_compressibility_reuss() const override;
  double compute_isothermal_bulk_modulus_reuss() const override;
  double compute_isentropic_compressibility_reuss() const override;
  double compute_isentropic_bulk_modulus_reuss() const override;
  // Preserve the reference's scalar-EOS alpha/Cp convention. Finite-strain
  // thermal derivatives have explicit isometric and isostress accessors.
  double compute_thermal_expansivity() const override;
  double compute_molar_heat_capacity_p() const override;
  double compute_molar_heat_capacity_v() const override;

private:
  struct Geometry {
    Eigen::Matrix3d cell, unrotated_M, isochoric_rotation, hencky;
    double volume;
  };
  struct Responses {
    std::optional<double> elastic, entropy, Ceps;
    std::optional<Eigen::Matrix3d> stress_unrotated, stress, pi_unrotated,
        alpha;
    std::optional<types::Matrix6d> unrotated_CT, CT, ST, CS, SS;
  };
  const Geometry &geometry() const;
  void update_frame();
  double temperature_step() const;
  double elastic_energy_at(const Eigen::Matrix3d &cell, double T,
                           AnisotropicMineral &hydrostatic) const;
  std::shared_ptr<AnisotropicMineral> probe() const;
  std::shared_ptr<AnisotropicMineral> hydrostatic_;
  OutputFrame frame_ = OutputFrame::current;
  std::optional<Geometry> geometry_;
  Eigen::Matrix3d crystal_rotation_, rotation_, deformation_gradient_;
  mutable Responses responses_;
};
} // namespace burnman
