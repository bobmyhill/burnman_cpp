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

#include "burnman/core/anisotropic_mineral.hpp"
#include "burnman/utils/exceptions.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace burnman {
namespace {
Eigen::Matrix3d contract_identity(const types::ElasticTensor &tensor) {
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        result.data()[i + 3 * j] += tensor[i][j][k][k];
  return result;
}

Eigen::Matrix3d contract_second(const types::ElasticTensor &tensor,
                                const Eigen::Matrix3d &second) {
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          result.data()[i + 3 * j] +=
              tensor[i][j][k][l] * second.data()[k + 3 * l];
  return result;
}

types::Matrix6d inverse(const types::Matrix6d &matrix) {
  const Eigen::FullPivLU<types::Matrix6d> lu(matrix);
  if (!matrix.allFinite() || !lu.isInvertible())
    throw std::runtime_error("Elastic tensor must be finite and invertible.");
  return lu.inverse();
}

const Mineral &require_mineral(const std::shared_ptr<Mineral> &mineral) {
  if (!mineral || !mineral->eos_method || !mineral->params.V_0 ||
      !mineral->params.T_0 || *mineral->params.V_0 <= 0.)
    throw std::invalid_argument(
        "Require a reference mineral with an EOS, V_0 > 0 and T_0.");
  return *mineral;
}

} // namespace

// The upper-right block of exp([[A,E],[0,A]]) is the Frechet derivative.
std::pair<Eigen::Matrix3d, Eigen::Matrix3d>
matrix_exponential_frechet(const Eigen::Matrix3d &a, const Eigen::Matrix3d &e) {
  types::Matrix6d block = types::Matrix6d::Zero();
  block.topLeftCorner<3, 3>() = block.bottomRightCorner<3, 3>() = a;
  block.topRightCorner<3, 3>() = e;
  const types::Matrix6d exponential = block.exp();
  return {exponential.topLeftCorner<3, 3>(),
          exponential.topRightCorner<3, 3>()};
}

namespace {
void check_shape(const types::NDArray<double> &c) {
  const auto &shape = c.shape();
  if (shape.size() != 4 || shape[0] != 6 || shape[1] != 6 || shape[2] < 2 ||
      shape[3] == 0)
    throw std::invalid_argument(
        "Anisotropic coefficients require shape (6,6,m,n), m >= 2, n >= 1.");
  for (double value : c.data())
    if (!std::isfinite(value))
      throw std::invalid_argument("Anisotropic coefficients must be finite.");
}
} // namespace

std::tuple<types::Matrix6d, Eigen::Matrix3d, Eigen::Matrix3d>
convert_f_Pth_to_f_T_derivatives(const types::Matrix6d &df,
                                 const types::Matrix6d &dp, double alpha_KT,
                                 double dPthdf) {
  const auto dfI =
      contract_identity(utils::voigt_notation_to_compliance_tensor(df));
  const auto dpI =
      contract_identity(utils::voigt_notation_to_compliance_tensor(dp));
  return {df + dp * dPthdf, dfI + dpI * dPthdf, alpha_KT * dpI};
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Matrix3d, types::Matrix6d>
deformation_gradient_alpha_and_compliance(double alpha, double beta,
                                          const Eigen::Matrix3d &PsiI,
                                          const types::Matrix6d &dPsidf_T,
                                          const Eigen::Matrix3d &dPsiIdf_T,
                                          const Eigen::Matrix3d &dPsiIdT_f) {
  const auto [F, dFdf] = matrix_exponential_frechet(PsiI, dPsiIdf_T);
  const auto dFdT_f = matrix_exponential_frechet(PsiI, dPsiIdT_f).second;
  const Eigen::Matrix3d invF = F.inverse();
  const Eigen::Matrix3d LP = -beta * dFdf * invF.transpose();
  const Eigen::Matrix3d beta_T = -0.5 * (LP + LP.transpose());
  const Eigen::Matrix3d LT = (dFdT_f + alpha * dFdf) * invF.transpose();
  const Eigen::Matrix3d expansivity = 0.5 * (LT + LT.transpose());
  types::Matrix6d compliance = beta * dPsidf_T;
  for (const auto &ijk :
       {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{1, 2, 0},
        std::array<int, 3>{2, 0, 1}}) {
    const auto [i, j, k] = ijk;
    compliance(i, j) =
        0.5 * (-compliance(i, i) - compliance(j, j) + compliance(k, k) +
               beta_T(i, i) + beta_T(j, j) - beta_T(k, k));
    compliance(j, i) = compliance(i, j);
    compliance(i, i + 3) =
        2. * beta_T(j, k) - compliance(j, i + 3) - compliance(k, i + 3);
    compliance(i + 3, i) = compliance(i, i + 3);
  }
  return {F, dFdf, expansivity, compliance};
}

AnisotropicMineral::AnisotropicMineral(
    std::shared_ptr<Mineral> mineral, const utils::CellParameters &cell,
    const types::NDArray<double> &coefficients,
    std::optional<utils::FrameConvention> frame)
    : Material(), Mineral(require_mineral(mineral)),
      AnisotropicMaterial(1., types::Matrix6d::Identity()),
      isotropic_mineral_(std::move(mineral)), coefficients_(coefficients),
      orthotropic_(check_standard_parameters(coefficients)) {
  initialize(cell, frame);
}

AnisotropicMineral::AnisotropicMineral(
    std::shared_ptr<Mineral> mineral, const utils::CellParameters &cell,
    PsiFunction psi_function, bool orthotropic,
    std::optional<utils::FrameConvention> frame)
    : Material(), Mineral(require_mineral(mineral)),
      AnisotropicMaterial(1., types::Matrix6d::Identity()),
      isotropic_mineral_(std::move(mineral)),
      psi_function_(std::move(psi_function)), orthotropic_(orthotropic) {
  if (!psi_function_)
    throw std::invalid_argument("A custom Psi function must be callable.");
  initialize(cell, frame);
}

void AnisotropicMineral::initialize(
    const utils::CellParameters &cell,
    std::optional<utils::FrameConvention> frame) {
  if (!orthotropic_ && !frame)
    throw std::invalid_argument(
        "A frame convention is required for non-orthotropic minerals.");
  frame_convention_ = orthotropic_ ? utils::FrameConvention{0, 1, 2} : *frame;
  if (!std::get<0>(evaluate_psi(0., 0.)).isZero(0.))
    throw std::invalid_argument("Psi must vanish at the reference state.");
  cell_vectors_0_ = utils::cell_parameters_to_vectors(cell, frame_convention_);
  if (std::abs(cell_vectors_0_.determinant() - *params.V_0) >
      std::numeric_limits<double>::epsilon())
    throw std::invalid_argument(
        "Reference cell volume must equal V_0; rescale the cell lengths.");
}

bool AnisotropicMineral::check_standard_parameters(
    const types::NDArray<double> &c) {
  check_shape(c);
  bool orthotropic = true;
  const auto &shape = c.shape();
  types::Matrix6d first;
  for (std::size_t m = 0; m < shape[2]; ++m)
    for (std::size_t n = 0; n < shape[3]; ++n) {
      double sum = 0.;
      for (std::size_t i = 0; i < 6; ++i) {
        double row_sum = 0.;
        for (std::size_t j = 0; j < 6; ++j) {
          const double value = c({i, j, m, n});
          if (m == 0 && n == 0 && value != 0.)
            throw std::invalid_argument(
                "All coefficients at m=n=0 must vanish.");
          if (std::abs(value - c({j, i, m, n})) >
              1.e-8 + 1.e-8 * std::abs(c({j, i, m, n})))
            throw std::invalid_argument(
                "Each anisotropic coefficient matrix must be symmetric.");
          if (i < 3 && j < 3)
            sum += value;
          if (j < 3)
            row_sum += value;
          if (m == 1 && n == 0)
            first(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                value;
        }
        if (i >= 3 && std::abs(row_sum) >= 1.e-10)
          orthotropic = false;
      }
      const double expected = (m == 1 && n == 0) ? 1. : 0.;
      const double tolerance = (m == 1 && n == 0) ? 1.e-5 : 1.e-10;
      if (std::abs(sum - expected) > tolerance)
        throw std::invalid_argument(
            "Coefficient normal-block sums are inconsistent with the volume.");
    }
  if (!Eigen::FullPivLU<types::Matrix6d>(first).isInvertible())
    throw std::invalid_argument(
        "The coefficient matrix at m=1,n=0 is singular.");
  return orthotropic;
}

PsiDerivatives
AnisotropicMineral::standard_psi_function(double f, double Pth,
                                          const types::NDArray<double> &c) {
  check_shape(c);
  types::Matrix6d psi = types::Matrix6d::Zero(), df = psi, dp = psi;
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      for (std::size_t n = 0; n < c.shape()[3]; ++n) {
        double x = c({i, j, 0, n}) + c({i, j, 1, n}) * f;
        double derivative = c({i, j, 1, n});
        for (std::size_t m = 2; m < c.shape()[2]; ++m) {
          x += c({i, j, m, n}) * std::pow(f, static_cast<double>(m)) /
               static_cast<double>(m);
          derivative +=
              c({i, j, m, n}) * std::pow(f, static_cast<double>(m - 1));
        }
        const auto row = static_cast<Eigen::Index>(i),
                   col = static_cast<Eigen::Index>(j);
        psi(row, col) += x * std::pow(Pth, static_cast<double>(n));
        df(row, col) += derivative * std::pow(Pth, static_cast<double>(n));
        if (n > 0)
          dp(row, col) += x * static_cast<double>(n) *
                          std::pow(Pth, static_cast<double>(n - 1));
      }
  return {psi, df, dp};
}

PsiDerivatives AnisotropicMineral::evaluate_psi(double f, double Pth) const {
  auto result = psi_function_ ? psi_function_(f, Pth)
                              : standard_psi_function(f, Pth, coefficients_);
  if (!std::get<0>(result).allFinite() || !std::get<1>(result).allFinite() ||
      !std::get<2>(result).allFinite())
    throw std::invalid_argument(
        "Psi and its derivatives must be finite 6x6 matrices.");
  return result;
}

std::shared_ptr<Material> AnisotropicMineral::copy() const {
  auto result = std::make_shared<AnisotropicMineral>(*this);
  result->isotropic_mineral_ =
      std::dynamic_pointer_cast<Mineral>(isotropic_mineral_->copy());
  return result;
}

void AnisotropicMineral::reset_cache() {
  AnisotropicMaterial::reset_cache();
  tensors_.reset();
}

void AnisotropicMineral::set_state(double P, double T) {
  if (!std::isfinite(P) || !std::isfinite(T) || T < 0.)
    throw std::invalid_argument(
        "Require finite pressure and finite temperature >= 0.");
  Mineral::set_state(P, T);
  tensors();
}

AnisotropicMineral::AnisotropicMineral()
    : AnisotropicMaterial(1., types::Matrix6d::Identity()), orthotropic_(false),
      frame_convention_{0, 1, 2}, cell_vectors_0_(Eigen::Matrix3d::Zero()) {}

void AnisotropicMineral::set_reference_cell(
    const Eigen::Matrix3d &vectors, const utils::FrameConvention &frame) {
  cell_vectors_0_ = vectors;
  frame_convention_ = frame;
  reset_cache();
}

AnisotropicMineral::PsiState AnisotropicMineral::compute_psi_state() const {
  isotropic_mineral_->set_state(get_pressure(), get_temperature());
  const double volume = isotropic_mineral_->get_molar_volume();
  const double kt = isotropic_mineral_->get_isothermal_bulk_modulus_reuss();
  auto cold = *isotropic_mineral_;
  cold.set_state_with_volume(volume, *params.T_0);
  PsiState state;
  state.Pth = get_pressure() - cold.get_pressure();
  state.dPthdf = cold.get_isothermal_bulk_modulus_reuss() - kt;
  const double f = std::log(get_molar_volume() / *params.V_0);
  const auto [psi, df, dp] = evaluate_psi(f, state.Pth);
  state.PsiI =
      contract_identity(utils::voigt_notation_to_compliance_tensor(psi));
  const double alpha_KT =
      get_thermal_expansivity() * get_isothermal_bulk_modulus_reuss();
  std::tie(state.dPsidf_T, state.dPsiIdf_T, state.dPsiIdT_f) =
      convert_f_Pth_to_f_T_derivatives(df, dp, alpha_KT, state.dPthdf);
  return state;
}

std::pair<double, double>
AnisotropicMineral::compute_deformation_alpha_beta() const {
  return {get_thermal_expansivity(), get_isothermal_compressibility_reuss()};
}

AnisotropicMineral::Tensors &AnisotropicMineral::tensors() const {
  if (!has_state())
    throw std::runtime_error(
        "Set the mineral state before querying anisotropic properties.");
  if (!tensors_) {
    Tensors state;
    state.psi = compute_psi_state();
    const auto &psi = state.psi;
    const auto [alpha, beta] = compute_deformation_alpha_beta();
    if (orthotropic_) {
      state.F = psi.PsiI.exp();
      state.dFdf = state.F * psi.dPsiIdf_T;
      state.alpha = psi.dPsiIdT_f + alpha * psi.dPsiIdf_T;
      state.unrotated_ST = beta * psi.dPsidf_T;
      state.rotation = Eigen::Matrix3d::Identity();
    } else {
      std::tie(state.F, state.dFdf, state.alpha, state.unrotated_ST) =
          deformation_gradient_alpha_and_compliance(alpha, beta, psi.PsiI,
                                                    psi.dPsidf_T, psi.dPsiIdf_T,
                                                    psi.dPsiIdT_f);
      state.rotation = utils::rotation_to_crystallographic_frame(
          cell_vectors_0_ * state.F.transpose(), frame_convention_);
    }
    state.ST = orthotropic_
                   ? state.unrotated_ST
                   : utils::contract_compliances(utils::rotate_elastic_tensor(
                         utils::voigt_notation_to_compliance_tensor(
                             state.unrotated_ST),
                         state.rotation));
    if (!orthotropic_)
      state.alpha =
          (state.rotation * state.alpha * state.rotation.transpose()).eval();
    tensors_ = std::move(state);
  }
  return *tensors_;
}

const Eigen::Matrix3d &AnisotropicMineral::get_PsiI() const {
  return tensors().psi.PsiI;
}
const types::Matrix6d &AnisotropicMineral::get_dPsidf_T_Voigt() const {
  return tensors().psi.dPsidf_T;
}
const Eigen::Matrix3d &AnisotropicMineral::get_dPsiIdf_T() const {
  return tensors().psi.dPsiIdf_T;
}
const Eigen::Matrix3d &AnisotropicMineral::get_dPsiIdT_f() const {
  return tensors().psi.dPsiIdT_f;
}
const Eigen::Matrix3d &AnisotropicMineral::get_unrotated_dFdf_T() const {
  return tensors().dFdf;
}
Eigen::Matrix3d AnisotropicMineral::get_unrotated_alpha() const {
  const auto &state = tensors();
  return state.rotation.transpose() * state.alpha * state.rotation;
}
const types::Matrix6d &
AnisotropicMineral::get_unrotated_isothermal_compliance_tensor() const {
  return tensors().unrotated_ST;
}

const std::shared_ptr<Mineral> &
AnisotropicMineral::get_isotropic_mineral() const {
  return isotropic_mineral_;
}
bool AnisotropicMineral::get_orthotropic() const { return orthotropic_; }
const utils::FrameConvention &AnisotropicMineral::get_frame_convention() const {
  return frame_convention_;
}
const Eigen::Matrix3d &AnisotropicMineral::get_cell_vectors_0() const {
  return cell_vectors_0_;
}
const Eigen::Matrix3d &
AnisotropicMineral::get_deformation_gradient_tensor() const {
  return tensors().F;
}
Eigen::Matrix3d AnisotropicMineral::get_unrotated_cell_vectors() const {
  return cell_vectors_0_ * tensors().F.transpose();
}
const Eigen::Matrix3d &AnisotropicMineral::get_rotation_matrix() const {
  return tensors().rotation;
}
Eigen::Matrix3d AnisotropicMineral::get_cell_vectors() const {
  return get_unrotated_cell_vectors() * get_rotation_matrix().transpose();
}
utils::CellParameters AnisotropicMineral::get_cell_parameters() const {
  return utils::cell_vectors_to_parameters(get_cell_vectors(),
                                           frame_convention_);
}
double AnisotropicMineral::get_thermal_pressure() const {
  return tensors().psi.Pth;
}
double AnisotropicMineral::get_dPthdf() const { return tensors().psi.dPthdf; }
const Eigen::Matrix3d &
AnisotropicMineral::get_thermal_expansivity_tensor() const {
  return tensors().alpha;
}
const types::Matrix6d &
AnisotropicMineral::get_isothermal_compliance_tensor() const {
  return tensors().ST;
}

const types::Matrix6d &
AnisotropicMineral::get_isothermal_stiffness_tensor() const {
  auto &state = tensors();
  if (!state.CT)
    state.CT = inverse(state.ST);
  return *state.CT;
}

const types::Matrix6d &
AnisotropicMineral::get_unrotated_isothermal_stiffness_tensor() const {
  auto &state = tensors();
  if (!state.unrotated_CT)
    state.unrotated_CT = inverse(state.unrotated_ST);
  return *state.unrotated_CT;
}

const types::Matrix6d &
AnisotropicMineral::get_isentropic_compliance_tensor() const {
  auto &state = tensors();
  if (!state.SS) {
    const auto alpha =
        utils::contract_strains(get_thermal_expansivity_tensor());
    state.SS =
        get_isothermal_compliance_tensor() -
        (get_molar_volume() * get_temperature() / get_molar_heat_capacity_p()) *
            alpha * alpha.transpose();
  }
  return *state.SS;
}

const types::Matrix6d &
AnisotropicMineral::get_isentropic_stiffness_tensor() const {
  auto &state = tensors();
  if (!state.CS)
    state.CS = inverse(get_isentropic_compliance_tensor());
  return *state.CS;
}

types::ElasticTensor
AnisotropicMineral::get_full_isothermal_compliance_tensor() const {
  return utils::voigt_notation_to_compliance_tensor(
      get_isothermal_compliance_tensor());
}
types::ElasticTensor
AnisotropicMineral::get_full_isothermal_stiffness_tensor() const {
  return utils::voigt_notation_to_stiffness_tensor(
      get_isothermal_stiffness_tensor());
}
Eigen::Matrix3d
AnisotropicMineral::get_isothermal_compressibility_tensor() const {
  return contract_identity(get_full_isothermal_compliance_tensor());
}
Eigen::Matrix3d
AnisotropicMineral::get_isentropic_compressibility_tensor() const {
  return contract_identity(get_full_isentropic_compliance_tensor());
}
Eigen::Matrix3d AnisotropicMineral::get_grueneisen_tensor() const {
  return contract_second(get_full_isentropic_stiffness_tensor(),
                         get_thermal_expansivity_tensor()) *
         get_molar_volume() / get_molar_heat_capacity_p();
}
Eigen::Matrix3d AnisotropicMineral::get_thermal_stress_tensor() const {
  return -contract_second(get_full_isothermal_stiffness_tensor(),
                          get_thermal_expansivity_tensor());
}
double AnisotropicMineral::get_molar_isometric_heat_capacity() const {
  return get_molar_heat_capacity_p() +
         get_molar_volume() * get_temperature() *
             (get_thermal_expansivity_tensor().array() *
              get_thermal_stress_tensor().array())
                 .sum();
}
double AnisotropicMineral::get_isothermal_bulk_modulus_voigt() const {
  return get_isothermal_stiffness_tensor().topLeftCorner<3, 3>().sum() / 9.;
}
double AnisotropicMineral::get_isothermal_compressibility_voigt() const {
  return 1. / get_isothermal_bulk_modulus_voigt();
}
double AnisotropicMineral::get_isentropic_compressibility_voigt() const {
  return 1. / get_isentropic_bulk_modulus_voigt();
}
double AnisotropicMineral::compute_density() const {
  return Mineral::compute_density();
}
double AnisotropicMineral::compute_isentropic_bulk_modulus_reuss() const {
  return Mineral::compute_isentropic_bulk_modulus_reuss();
}
double AnisotropicMineral::compute_molar_heat_capacity_v() const {
  return Mineral::compute_molar_heat_capacity_v();
}
double AnisotropicMineral::compute_grueneisen_parameter() const {
  return get_thermal_expansivity() * get_molar_volume() /
         (get_isentropic_compressibility_reuss() * get_molar_heat_capacity_p());
}
double AnisotropicMineral::compute_shear_modulus() const {
  throw exceptions::NotImplementedError(
      "AnisotropicMineral",
      "shear_modulus: query directional or tensor shear properties");
}
} // namespace burnman
