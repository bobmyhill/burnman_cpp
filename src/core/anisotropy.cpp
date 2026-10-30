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

#include "burnman/core/anisotropy.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

namespace burnman {
namespace {
Eigen::Vector3d unit_normalize(const Eigen::Vector3d &direction) {
  if (!direction.allFinite())
    throw std::invalid_argument("Directions must be finite.");
  return direction.normalized();
}

double contract(const types::ElasticTensor &tensor, const Eigen::Vector3d &a,
                const Eigen::Vector3d &b, const Eigen::Vector3d &c,
                const Eigen::Vector3d &d) {
  double result = 0.;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          result += tensor[i][j][k][l] * a.data()[i] * b.data()[j] *
                    c.data()[k] * d.data()[l];
  return result;
}

void check_orthogonal(const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
  if (std::abs(a.dot(b)) >= 1.e-12)
    throw std::invalid_argument("Directions must be orthogonal.");
}

types::Matrix6d cubic_tensor(const std::vector<double> &cijs) {
  return utils::voigt_array_from_cijs(cijs, {{{0, 0}, {1, 1}, {2, 2}},
                                             {{0, 1}, {0, 2}, {1, 2}},
                                             {{3, 3}, {4, 4}, {5, 5}}});
}

types::Matrix6d tetragonal_tensor(std::vector<double> cijs) {
  if (cijs.size() == 6)
    return utils::voigt_array_from_cijs(cijs, {{{0, 0}, {1, 1}},
                                               {{0, 1}},
                                               {{0, 2}, {1, 2}},
                                               {{2, 2}},
                                               {{3, 3}, {4, 4}},
                                               {{5, 5}}});
  if (cijs.size() != 7)
    throw std::invalid_argument("Tetragonal materials require 6 or 7 Cijs.");
  cijs.insert(cijs.begin() + 4, -cijs[3]);
  return utils::voigt_array_from_cijs(cijs, {{{0, 0}, {1, 1}},
                                             {{0, 1}},
                                             {{0, 2}, {1, 2}},
                                             {{0, 5}},
                                             {{1, 5}},
                                             {{2, 2}},
                                             {{3, 3}, {4, 4}},
                                             {{5, 5}}});
}

types::Matrix6d rhombohedral_tensor(std::vector<double> cijs) {
  if (cijs.size() != 7 && cijs.size() != 8)
    throw std::invalid_argument("Rhombohedral materials require 7 or 8 Cijs.");
  cijs.insert(cijs.begin() + 4, -cijs[3]);
  if (cijs.size() == 8)
    return utils::voigt_array_from_cijs(cijs, {{{0, 0}, {1, 1}},
                                               {{0, 1}},
                                               {{0, 2}, {1, 2}},
                                               {{0, 3}, {4, 5}},
                                               {{1, 3}},
                                               {{2, 2}},
                                               {{3, 3}, {4, 4}},
                                               {{5, 5}}});
  cijs.insert(cijs.begin() + 6, -cijs[5]);
  return utils::voigt_array_from_cijs(cijs, {{{0, 0}, {1, 1}},
                                             {{0, 1}},
                                             {{0, 2}, {1, 2}},
                                             {{0, 3}, {4, 5}},
                                             {{1, 3}},
                                             {{0, 4}},
                                             {{1, 4}, {3, 5}},
                                             {{2, 2}},
                                             {{3, 3}, {4, 4}},
                                             {{5, 5}}});
}
} // namespace

AnisotropicMaterial::AnisotropicMaterial(double rho,
                                         const types::Matrix6d &cijs)
    : rho_(rho), stiffness_(cijs) {
  if (!std::isfinite(rho) || rho <= 0.)
    throw std::invalid_argument("Density must be positive and finite.");
  if (!cijs.allFinite() || !cijs.isApprox(cijs.transpose(), 1.e-5))
    throw std::invalid_argument(
        "Stiffness tensor must be finite and symmetric.");
}

std::shared_ptr<Material> AnisotropicMaterial::copy() const {
  return std::make_shared<AnisotropicMaterial>(*this);
}

void AnisotropicMaterial::reset_cache() {
  Material::reset_cache();
  compliance_.reset();
}

const types::Matrix6d &
AnisotropicMaterial::get_isentropic_stiffness_tensor() const {
  return stiffness_;
}

types::ElasticTensor
AnisotropicMaterial::get_full_isentropic_stiffness_tensor() const {
  return utils::voigt_notation_to_stiffness_tensor(
      get_isentropic_stiffness_tensor());
}

const types::Matrix6d &
AnisotropicMaterial::get_isentropic_compliance_tensor() const {
  if (!compliance_) {
    const Eigen::FullPivLU<types::Matrix6d> lu(
        get_isentropic_stiffness_tensor());
    if (!lu.isInvertible())
      throw std::runtime_error("Stiffness tensor is singular.");
    compliance_ = lu.inverse();
  }
  return *compliance_;
}

types::ElasticTensor
AnisotropicMaterial::get_full_isentropic_compliance_tensor() const {
  return utils::voigt_notation_to_compliance_tensor(
      get_isentropic_compliance_tensor());
}

double AnisotropicMaterial::compute_density() const { return rho_; }

double AnisotropicMaterial::get_isentropic_bulk_modulus_voigt() const {
  return get_isentropic_stiffness_tensor().topLeftCorner<3, 3>().sum() / 9.;
}

double AnisotropicMaterial::compute_isentropic_bulk_modulus_reuss() const {
  return 1. / get_isentropic_compliance_tensor().topLeftCorner<3, 3>().sum();
}

double AnisotropicMaterial::get_isentropic_bulk_modulus_vrh() const {
  return 0.5 * (get_isentropic_bulk_modulus_voigt() +
                get_isentropic_bulk_modulus_reuss());
}

double AnisotropicMaterial::get_isentropic_shear_modulus_voigt() const {
  const auto &c = get_isentropic_stiffness_tensor();
  return (c.diagonal().head<3>().sum() + 3. * c.diagonal().tail<3>().sum() -
          c(0, 1) - c(1, 2) - c(2, 0)) /
         15.;
}

double AnisotropicMaterial::get_isentropic_shear_modulus_reuss() const {
  const auto &s = get_isentropic_compliance_tensor();
  return 15. / (4. * s.diagonal().head<3>().sum() +
                3. * s.diagonal().tail<3>().sum() -
                4. * (s(0, 1) + s(1, 2) + s(2, 0)));
}

double AnisotropicMaterial::get_isentropic_shear_modulus_vrh() const {
  return 0.5 * (get_isentropic_shear_modulus_voigt() +
                get_isentropic_shear_modulus_reuss());
}

double
AnisotropicMaterial::get_isentropic_universal_elastic_anisotropy() const {
  return 5. * get_isentropic_shear_modulus_voigt() /
             get_isentropic_shear_modulus_reuss() +
         get_isentropic_bulk_modulus_voigt() /
             get_isentropic_bulk_modulus_reuss() -
         6.;
}

double AnisotropicMaterial::get_isentropic_isotropic_poisson_ratio() const {
  const double k = get_isentropic_bulk_modulus_vrh();
  const double g = get_isentropic_shear_modulus_vrh();
  return (3. * k - 2. * g) / (6. * k + 2. * g);
}

std::optional<Eigen::Matrix3d> AnisotropicMaterial::get_cauchy_stress() const {
  return std::nullopt;
}

Eigen::Matrix3d AnisotropicMaterial::christoffel_tensor(
    const Eigen::Vector3d &propagation_direction) const {
  const auto n = unit_normalize(propagation_direction);
  const auto c = get_full_isentropic_stiffness_tensor();
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          result.data()[i + 3 * k] += c[i][j][k][l] * n.data()[j] * n.data()[l];
  // Wallace's equation-of-motion tensor under nonhydrostatic stress.
  if (const auto stress = get_cauchy_stress())
    result = (0.5 * (result + result.transpose()) +
              0.5 * (n.dot(*stress * n) * Eigen::Matrix3d::Identity() -
                     n.squaredNorm() * *stress))
                 .eval();
  return result;
}

double AnisotropicMaterial::isentropic_linear_compressibility(
    const Eigen::Vector3d &direction) const {
  const auto n = unit_normalize(direction);
  const auto s = get_full_isentropic_compliance_tensor();
  double result = 0.;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        result += s[i][j][k][k] * n.data()[i] * n.data()[j];
  return result;
}

double AnisotropicMaterial::isentropic_youngs_modulus(
    const Eigen::Vector3d &direction) const {
  const auto n = unit_normalize(direction);
  return 1. / contract(get_full_isentropic_compliance_tensor(), n, n, n, n);
}

double AnisotropicMaterial::isentropic_shear_modulus(
    const Eigen::Vector3d &plane_normal,
    const Eigen::Vector3d &shear_direction) const {
  const auto n = unit_normalize(plane_normal);
  const auto d = unit_normalize(shear_direction);
  check_orthogonal(n, d);
  return 0.25 / contract(get_full_isentropic_compliance_tensor(), d, n, d, n);
}

double AnisotropicMaterial::isentropic_poissons_ratio(
    const Eigen::Vector3d &axial_direction,
    const Eigen::Vector3d &lateral_direction) const {
  const auto a = unit_normalize(axial_direction);
  const auto l = unit_normalize(lateral_direction);
  check_orthogonal(a, l);
  return -isentropic_youngs_modulus(a) *
         contract(get_full_isentropic_compliance_tensor(), l, l, a, a);
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d>
AnisotropicMaterial::wave_velocities(
    const Eigen::Vector3d &propagation_direction) const {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(
      christoffel_tensor(propagation_direction));
  if (eig.info() != Eigen::Success)
    throw std::runtime_error("Christoffel eigendecomposition failed.");
  return {(eig.eigenvalues().reverse().array() / get_density()).sqrt().matrix(),
          eig.eigenvectors().rowwise().reverse()};
}

IsotropicMaterial::IsotropicMaterial(double rho,
                                     const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, [&] {
        if (cijs.size() != 2)
          throw std::invalid_argument("Isotropic materials require 2 Cijs.");
        return cubic_tensor({cijs[0] + 2. * cijs[1], cijs[0], cijs[1]});
      }()) {}

CubicMaterial::CubicMaterial(double rho, const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, cubic_tensor(cijs)) {}

HexagonalMaterial::HexagonalMaterial(double rho,
                                     const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, [&] {
        if (cijs.size() != 5)
          throw std::invalid_argument("Hexagonal materials require 5 Cijs.");
        auto values = cijs;
        values.push_back((cijs[0] - cijs[1]) / 2.);
        return tetragonal_tensor(values);
      }()) {}

TetragonalMaterial::TetragonalMaterial(double rho,
                                       const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, tetragonal_tensor(cijs)) {}

RhombohedralMaterial::RhombohedralMaterial(double rho,
                                           const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, rhombohedral_tensor(cijs)) {}

OrthorhombicMaterial::OrthorhombicMaterial(double rho,
                                           const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, utils::voigt_array_from_cijs(cijs, {{{0, 0}},
                                                                   {{0, 1}},
                                                                   {{0, 2}},
                                                                   {{1, 1}},
                                                                   {{1, 2}},
                                                                   {{2, 2}},
                                                                   {{3, 3}},
                                                                   {{4, 4}},
                                                                   {{5, 5}}})) {
}

MonoclinicMaterial::MonoclinicMaterial(double rho,
                                       const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, utils::voigt_array_from_cijs(cijs, {{{0, 0}},
                                                                   {{0, 1}},
                                                                   {{0, 2}},
                                                                   {{0, 4}},
                                                                   {{1, 1}},
                                                                   {{1, 2}},
                                                                   {{1, 4}},
                                                                   {{2, 2}},
                                                                   {{2, 4}},
                                                                   {{3, 3}},
                                                                   {{3, 5}},
                                                                   {{4, 4}},
                                                                   {{5, 5}}})) {
}

TriclinicMaterial::TriclinicMaterial(double rho,
                                     const std::vector<double> &cijs)
    : AnisotropicMaterial(rho, [&] {
        std::vector<std::vector<std::pair<int, int>>> indices;
        for (int i = 0; i < 6; ++i)
          for (int j = i; j < 6; ++j)
            indices.push_back({{i, j}});
        return utils::voigt_array_from_cijs(cijs, indices);
      }()) {}

#define COPY_MATERIAL(Class)                                                   \
  std::shared_ptr<Material> Class::copy() const {                              \
    return std::make_shared<Class>(*this);                                     \
  }
COPY_MATERIAL(IsotropicMaterial)
COPY_MATERIAL(CubicMaterial)
COPY_MATERIAL(HexagonalMaterial)
COPY_MATERIAL(TetragonalMaterial)
COPY_MATERIAL(RhombohedralMaterial)
COPY_MATERIAL(OrthorhombicMaterial)
COPY_MATERIAL(MonoclinicMaterial)
COPY_MATERIAL(TriclinicMaterial)
#undef COPY_MATERIAL
} // namespace burnman
