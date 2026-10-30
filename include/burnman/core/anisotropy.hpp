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

#ifndef BURNMAN_CORE_ANISOTROPY_HPP_INCLUDED
#define BURNMAN_CORE_ANISOTROPY_HPP_INCLUDED

#include "burnman/core/material.hpp"
#include "burnman/utils/anisotropy.hpp"

namespace burnman {
/** Elastic material at a fixed density and isentropic stiffness (SI units).
 * Subclasses can override the stiffness and density for state dependent models;
 * reset_cache() invalidates compliance when their state changes.
 */
class AnisotropicMaterial : public virtual Material {
public:
  AnisotropicMaterial(double rho, const types::Matrix6d &cijs);
  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;

  virtual const types::Matrix6d &get_isentropic_stiffness_tensor() const;
  types::ElasticTensor get_full_isentropic_stiffness_tensor() const;
  virtual const types::Matrix6d &get_isentropic_compliance_tensor() const;
  types::ElasticTensor get_full_isentropic_compliance_tensor() const;
  double get_isentropic_bulk_modulus_voigt() const;
  double get_isentropic_bulk_modulus_vrh() const;
  double get_isentropic_shear_modulus_voigt() const;
  double get_isentropic_shear_modulus_reuss() const;
  double get_isentropic_shear_modulus_vrh() const;
  double get_isentropic_universal_elastic_anisotropy() const;
  double get_isentropic_isotropic_poisson_ratio() const;

  Eigen::Matrix3d
  christoffel_tensor(const Eigen::Vector3d &propagation_direction) const;
  double
  isentropic_linear_compressibility(const Eigen::Vector3d &direction) const;
  double isentropic_youngs_modulus(const Eigen::Vector3d &direction) const;
  double isentropic_shear_modulus(const Eigen::Vector3d &plane_normal,
                                  const Eigen::Vector3d &shear_direction) const;
  double
  isentropic_poissons_ratio(const Eigen::Vector3d &axial_direction,
                            const Eigen::Vector3d &lateral_direction) const;
  /// Speeds in descending order; columns are the matching unit polarizations.
  std::pair<Eigen::Vector3d, Eigen::Matrix3d>
  wave_velocities(const Eigen::Vector3d &propagation_direction) const;

protected:
  double compute_density() const override;
  double compute_isentropic_bulk_modulus_reuss() const override;
  /// Optional tension-positive Cauchy stress for nonhydrostatic subclasses.
  virtual std::optional<Eigen::Matrix3d> get_cauchy_stress() const;

private:
  double rho_;
  types::Matrix6d stiffness_;
  mutable std::optional<types::Matrix6d> compliance_;
};

/// Cijs: [C12, C44] (Lame parameters lambda, mu).
class IsotropicMaterial : public AnisotropicMaterial {
public:
  IsotropicMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C44].
class CubicMaterial : public AnisotropicMaterial {
public:
  CubicMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C13, C33, C44].
class HexagonalMaterial : public AnisotropicMaterial {
public:
  HexagonalMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C13, (C16), C33, C44, C66].
class TetragonalMaterial : public AnisotropicMaterial {
public:
  TetragonalMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C13, C14, (C15), C33, C44, C66].
class RhombohedralMaterial : public AnisotropicMaterial {
public:
  RhombohedralMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C13, C22, C23, C33, C44, C55, C66].
class OrthorhombicMaterial : public AnisotropicMaterial {
public:
  OrthorhombicMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: [C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66].
class MonoclinicMaterial : public AnisotropicMaterial {
public:
  MonoclinicMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};

/// Cijs: upper triangular entries, in row order (21 coefficients).
class TriclinicMaterial : public AnisotropicMaterial {
public:
  TriclinicMaterial(double rho, const std::vector<double> &cijs);
  std::shared_ptr<Material> copy() const override;
};
} // namespace burnman

#endif
