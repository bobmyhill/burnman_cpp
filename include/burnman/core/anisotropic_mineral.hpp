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

#ifndef BURNMAN_CORE_ANISOTROPIC_MINERAL_HPP_INCLUDED
#define BURNMAN_CORE_ANISOTROPIC_MINERAL_HPP_INCLUDED

#include "burnman/core/anisotropy.hpp"
#include "burnman/core/mineral.hpp"
#include "burnman/utils/types/ndarray.hpp"
#include "burnman/utils/unitcell.hpp"
#include <functional>
#include <tuple>

namespace burnman {
using PsiDerivatives =
    std::tuple<types::Matrix6d, types::Matrix6d, types::Matrix6d>;
using PsiFunction =
    std::function<PsiDerivatives(double f, double thermal_pressure)>;

std::pair<Eigen::Matrix3d, Eigen::Matrix3d>
matrix_exponential_frechet(const Eigen::Matrix3d &matrix,
                           const Eigen::Matrix3d &direction);

std::tuple<types::Matrix6d, Eigen::Matrix3d, Eigen::Matrix3d>
convert_f_Pth_to_f_T_derivatives(const types::Matrix6d &dPsidf_Pth,
                                 const types::Matrix6d &dPsidPth_f,
                                 double alpha_KT, double dPthdf);
std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, types::Matrix6d>
deformation_gradient_alpha_and_compliance(double alpha, double beta,
                                          const Eigen::Matrix3d &PsiI,
                                          const types::Matrix6d &dPsidf_T,
                                          const Eigen::Matrix3d &dPsiIdf_T,
                                          const Eigen::Matrix3d &dPsiIdT_f);

/// Myhill (2022) anisotropic EOS, retaining the reference mineral's scalar EOS.
class AnisotropicMineral : public Mineral, public AnisotropicMaterial {
public:
  /// Coefficients have shape (6,6,m,n), m >= 2 and n >= 1.
  AnisotropicMineral(
      std::shared_ptr<Mineral> isotropic_mineral,
      const utils::CellParameters &cell_parameters,
      const types::NDArray<double> &anisotropic_parameters,
      std::optional<utils::FrameConvention> frame_convention = std::nullopt);
  /// Custom Psi returns (Psi, dPsi/df at Pth, dPsi/dPth at f), in Voigt form.
  AnisotropicMineral(
      std::shared_ptr<Mineral> isotropic_mineral,
      const utils::CellParameters &cell_parameters, PsiFunction psi_function,
      bool orthotropic,
      std::optional<utils::FrameConvention> frame_convention = std::nullopt);

  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;
  void set_state(double pressure, double temperature) override;
  static bool
  check_standard_parameters(const types::NDArray<double> &coefficients);
  static PsiDerivatives
  standard_psi_function(double f, double Pth,
                        const types::NDArray<double> &coefficients);
  PsiDerivatives evaluate_psi(double f, double Pth) const;

  const std::shared_ptr<Mineral> &get_isotropic_mineral() const;
  bool get_orthotropic() const;
  const utils::FrameConvention &get_frame_convention() const;
  const Eigen::Matrix3d &get_cell_vectors_0() const;
  const Eigen::Matrix3d &get_deformation_gradient_tensor() const;
  Eigen::Matrix3d get_unrotated_cell_vectors() const;
  const Eigen::Matrix3d &get_rotation_matrix() const;
  Eigen::Matrix3d get_cell_vectors() const;
  utils::CellParameters get_cell_parameters() const;
  double get_thermal_pressure() const;
  double get_dPthdf() const;
  const Eigen::Matrix3d &get_PsiI() const;
  const types::Matrix6d &get_dPsidf_T_Voigt() const;
  const Eigen::Matrix3d &get_dPsiIdf_T() const;
  const Eigen::Matrix3d &get_dPsiIdT_f() const;
  const Eigen::Matrix3d &get_unrotated_dFdf_T() const;
  Eigen::Matrix3d get_unrotated_alpha() const;
  const types::Matrix6d &get_unrotated_isothermal_compliance_tensor() const;

  virtual const types::Matrix6d &get_isothermal_compliance_tensor() const;
  virtual const types::Matrix6d &get_isothermal_stiffness_tensor() const;
  const types::Matrix6d &get_unrotated_isothermal_stiffness_tensor() const;
  const types::Matrix6d &get_isentropic_compliance_tensor() const override;
  const types::Matrix6d &get_isentropic_stiffness_tensor() const override;
  types::ElasticTensor get_full_isothermal_compliance_tensor() const;
  types::ElasticTensor get_full_isothermal_stiffness_tensor() const;
  virtual const Eigen::Matrix3d &get_thermal_expansivity_tensor() const;
  Eigen::Matrix3d get_isothermal_compressibility_tensor() const;
  Eigen::Matrix3d get_isentropic_compressibility_tensor() const;
  Eigen::Matrix3d get_grueneisen_tensor() const;
  virtual Eigen::Matrix3d get_thermal_stress_tensor() const;
  virtual double get_molar_isometric_heat_capacity() const;
  double get_isothermal_bulk_modulus_voigt() const;
  double get_isothermal_compressibility_voigt() const;
  double get_isentropic_compressibility_voigt() const;

protected:
  /// Derived solution classes supply their own scalar EOS and reference cell.
  AnisotropicMineral();
  struct PsiState {
    double Pth, dPthdf;
    Eigen::Matrix3d PsiI, dPsiIdf_T, dPsiIdT_f;
    types::Matrix6d dPsidf_T;
  };
  virtual PsiState compute_psi_state() const;
  /// Scalar expansion and compressibility for deformation at frozen structure.
  virtual std::pair<double, double> compute_deformation_alpha_beta() const;
  void set_reference_cell(const Eigen::Matrix3d &vectors,
                          const utils::FrameConvention &frame);
  double compute_density() const override;
  double compute_isentropic_bulk_modulus_reuss() const override;
  double compute_shear_modulus() const override;
  double compute_molar_heat_capacity_v() const override;
  double compute_grueneisen_parameter() const override;

private:
  struct Tensors {
    PsiState psi;
    Eigen::Matrix3d F, dFdf, rotation, alpha;
    types::Matrix6d unrotated_ST, ST;
    std::optional<types::Matrix6d> unrotated_CT, CT, SS, CS;
  };
  void initialize(const utils::CellParameters &cell_parameters,
                  std::optional<utils::FrameConvention> frame_convention);
  Tensors &tensors() const;
  std::shared_ptr<Mineral> isotropic_mineral_;
  types::NDArray<double> coefficients_;
  PsiFunction psi_function_;
  bool orthotropic_;
  utils::FrameConvention frame_convention_;
  Eigen::Matrix3d cell_vectors_0_;
  mutable std::optional<Tensors> tensors_;
};
} // namespace burnman

#endif
