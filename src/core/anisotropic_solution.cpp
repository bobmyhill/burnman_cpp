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

#include "burnman/core/anisotropic_solution.hpp"
#include "burnman/utils/exceptions.hpp"
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace burnman {
namespace {
Eigen::Matrix3d weighted_sum(const CompositionTensors &matrices,
                             const Eigen::ArrayXd &fractions) {
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  for (Eigen::Index i = 0; i < fractions.size(); ++i)
    result += fractions[i] * matrices[static_cast<std::size_t>(i)];
  return result;
}
} // namespace

AnisotropicSolution::AnisotropicSolution(
    const std::string &solution_name,
    std::shared_ptr<solution_models::SolutionModel> model,
    PsiExcessFunction psi_excess_function,
    std::optional<Eigen::ArrayXd> fractions)
    : psi_excess_function_(std::move(psi_excess_function)) {
  if (!model || model->endmembers.empty() || !psi_excess_function_ ||
      model->polymorphic_endmembers.size() != model->endmembers.size())
    throw std::invalid_argument("Require a solution model with anisotropic "
                                "endmembers and a Psi excess function.");
  auto owned = model->clone();
  owned->polymorphic_endmembers.clear();
  for (std::size_t i = 0; i < model->endmembers.size(); ++i) {
    const auto endmember = std::dynamic_pointer_cast<AnisotropicMineral>(
        model->polymorphic_endmembers[i]);
    if (!endmember || !endmember->get_isotropic_mineral())
      throw std::invalid_argument(
          "Every endmember must be an AnisotropicMineral with a scalar "
          "reference mineral.");
    if (i && endmember->get_frame_convention() !=
                 endmembers_[0]->get_frame_convention())
      throw std::invalid_argument(
          "Anisotropic endmembers must share a frame convention.");
    auto copy =
        std::dynamic_pointer_cast<AnisotropicMineral>(endmember->copy());
    endmembers_.push_back(copy);
    owned->polymorphic_endmembers.push_back(copy);
    owned->endmembers[i] = *copy->get_isotropic_mineral();
    const Eigen::Matrix3d cell = copy->get_cell_vectors_0().transpose();
    logm_M0_mbr_.push_back(cell.log());
    if (!logm_M0_mbr_.back().allFinite())
      throw std::invalid_argument(
          "Reference cells must have finite real matrix logarithms.");
  }
  Solution::set_solution_model(owned);
  scalar_solution_ = std::make_shared<Solution>();
  auto scalar_model = owned->clone();
  scalar_model->polymorphic_endmembers.clear();
  scalar_solution_->set_solution_model(scalar_model);
  set_name(solution_name);
  scalar_solution_->set_name(solution_name);
  if (fractions)
    set_composition(*fractions);
}

std::shared_ptr<Material> AnisotropicSolution::copy() const {
  auto result = std::make_shared<AnisotropicSolution>(*this);
  auto model = get_solution_model()->clone();
  model->polymorphic_endmembers.clear();
  result->endmembers_.clear();
  for (const auto &endmember : endmembers_) {
    const auto copied =
        std::dynamic_pointer_cast<AnisotropicMineral>(endmember->copy());
    result->endmembers_.push_back(copied);
    model->polymorphic_endmembers.push_back(copied);
  }
  result->set_solution_model(model);
  result->scalar_solution_ =
      std::dynamic_pointer_cast<Solution>(scalar_solution_->copy());
  return result;
}

void AnisotropicSolution::reset_cache() {
  Solution::reset_cache();
  AnisotropicMineral::reset_cache();
  PsiI_mbr_.clear();
  dPsiIdp_excess_.clear();
}

void AnisotropicSolution::require_composition() const {
  if (get_molar_fractions().size() != get_n_endmembers())
    throw std::runtime_error("Call set_composition() before setting or "
                             "querying the solution state.");
}

void AnisotropicSolution::set_composition(const Eigen::ArrayXd &fractions) {
  if (fractions.size() != get_n_endmembers() || !fractions.isFinite().all() ||
      std::abs(fractions.sum() - 1.) > 1.e-12 ||
      (get_endmember_occupancies().matrix().transpose() * fractions.matrix())
              .minCoeff() < -1.e-10)
    throw std::invalid_argument("Require finite unit-sum fractions giving "
                                "nonnegative site occupancies.");
  Solution::set_composition(fractions);
  scalar_solution_->set_composition(fractions);
  const Eigen::Matrix3d logarithm = weighted_sum(logm_M0_mbr_, fractions);
  set_reference_cell(logarithm.exp().transpose(),
                     endmembers_[0]->get_frame_convention());
  if (has_state())
    compute_base_properties();
}

void AnisotropicSolution::set_state(double P, double T) {
  require_composition();
  if (!std::isfinite(P) || !std::isfinite(T) || T < 0.)
    throw std::invalid_argument(
        "Require finite pressure and finite temperature >= 0.");
  scalar_solution_->set_state(P, T);
  Solution::set_state(P, T);
  compute_base_properties();
}

void AnisotropicSolution::set_state_with_volume(
    double volume, double T, const Eigen::Vector2d &guesses) {
  require_composition();
  auto scalar = std::dynamic_pointer_cast<Solution>(scalar_solution_->copy());
  scalar->set_state_with_volume(volume, T, guesses);
  set_state(scalar->get_pressure(), T);
}

void AnisotropicSolution::set_method(types::EOSType method) {
  Solution::set_method(method);
  scalar_solution_->set_method(method);
  for (auto &endmember : endmembers_) {
    endmember->set_method(method);
    endmember->get_isotropic_mineral()->set_method(method);
  }
  reset_cache();
}
void AnisotropicSolution::set_method(std::shared_ptr<EquationOfState> method) {
  Solution::set_method(method);
  scalar_solution_->set_method(method);
  for (auto &endmember : endmembers_) {
    endmember->set_method(method);
    endmember->get_isotropic_mineral()->set_method(method);
  }
  reset_cache();
}

void AnisotropicSolution::compute_base_properties() {
  require_composition();
  reset_cache();
  AnisotropicMineral::get_deformation_gradient_tensor();
}

AnisotropicMineral::PsiState AnisotropicSolution::compute_psi_state() const {
  require_composition();
  const double P = get_pressure(), T = get_temperature();
  auto &scalar = *scalar_solution_;
  scalar.set_state(P, T);
  const double volume = scalar.get_molar_volume();
  const double kt = scalar.get_isothermal_bulk_modulus_reuss();
  const auto fractions = get_molar_fractions();
  const Eigen::Vector2d guesses(std::max(0., P - 2.e9), P + 2.e9);
  PsiState result;
  result.PsiI.setZero();
  result.dPsidf_T.setZero();
  result.dPsiIdf_T.setZero();
  result.dPsiIdT_f.setZero();
  PsiI_mbr_.clear();
  for (std::size_t i = 0; i < endmembers_.size(); ++i) {
    auto &endmember = *endmembers_[i];
    endmember.set_state_with_volume(volume, T, guesses);
    const double fraction = fractions[static_cast<Eigen::Index>(i)];
    PsiI_mbr_.push_back(endmember.get_PsiI());
    result.PsiI += fraction * endmember.get_PsiI();
    result.dPsidf_T += fraction * endmember.get_dPsidf_T_Voigt();
    result.dPsiIdf_T += fraction * endmember.get_dPsiIdf_T();
    result.dPsiIdT_f += fraction * endmember.get_dPsiIdT_f();
    endmember.set_state(P, T);
  }
  auto cold = std::dynamic_pointer_cast<Solution>(scalar.copy());
  cold->set_state_with_volume(volume, T_0);
  const double cold_pressure = cold->get_pressure();
  result.Pth = P - cold_pressure;
  result.dPthdf = cold->get_isothermal_bulk_modulus_reuss() - kt;
  const auto [psi, df, dp, dPsiIdp] =
      psi_excess_function_(std::log(volume), result.Pth, fractions);
  if (!psi.allFinite() || !df.allFinite() || !dp.allFinite() ||
      dPsiIdp.size() != endmembers_.size())
    throw std::invalid_argument("Psi excess requires three finite 6x6 matrices "
                                "and one 3x3 derivative per endmember.");
  for (const auto &derivative : dPsiIdp)
    if (!derivative.allFinite())
      throw std::invalid_argument(
          "Psi composition derivatives must be finite.");
  dPsiIdp_excess_ = dPsiIdp;
  const auto full = utils::voigt_notation_to_compliance_tensor(psi);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        result.PsiI(static_cast<Eigen::Index>(i),
                    static_cast<Eigen::Index>(j)) += full[i][j][k][k];
  const auto [df_T, dfI_T, dTI_f] = convert_f_Pth_to_f_T_derivatives(
      df, dp, scalar.get_thermal_expansivity() * kt, result.dPthdf);
  result.dPsidf_T += df_T;
  result.dPsiIdf_T += dfI_T;
  result.dPsiIdT_f += dTI_f;
  return result;
}

const std::vector<std::shared_ptr<AnisotropicMineral>> &
AnisotropicSolution::get_endmembers() const {
  return endmembers_;
}
const std::shared_ptr<Solution> &
AnisotropicSolution::get_scalar_solution() const {
  return scalar_solution_;
}
const CompositionTensors &AnisotropicSolution::get_logm_M0_mbr() const {
  return logm_M0_mbr_;
}
const CompositionTensors &AnisotropicSolution::get_PsiI_mbr() const {
  get_PsiI();
  return PsiI_mbr_;
}
const CompositionTensors &AnisotropicSolution::get_dPsiIdp_excess() const {
  get_PsiI();
  return dPsiIdp_excess_;
}

CompositionTensors AnisotropicSolution::get_dM0dp_fixed_VT() const {
  require_composition();
  const auto logarithm = weighted_sum(logm_M0_mbr_, get_molar_fractions());
  CompositionTensors result;
  for (const auto &direction : logm_M0_mbr_)
    result.push_back(matrix_exponential_frechet(logarithm, direction).second);
  return result;
}
CompositionTensors AnisotropicSolution::get_unrotated_dFdp_fixed_VT() const {
  const auto &psi = get_PsiI();
  CompositionTensors result;
  for (std::size_t i = 0; i < endmembers_.size(); ++i)
    result.push_back(
        matrix_exponential_frechet(psi, PsiI_mbr_[i] + dPsiIdp_excess_[i])
            .second);
  return result;
}
CompositionTensors AnisotropicSolution::get_unrotated_dMdn_fixed_VT() const {
  const auto dM0dp = get_dM0dp_fixed_VT();
  const auto dFdp = get_unrotated_dFdp_fixed_VT();
  const auto fractions = get_molar_fractions();
  const double n = fractions.sum();
  const Eigen::Matrix3d M0 = get_cell_vectors_0().transpose();
  const auto &F = get_deformation_gradient_tensor();
  const auto mean_dM0 = weighted_sum(dM0dp, fractions),
             mean_dF = weighted_sum(dFdp, fractions);
  CompositionTensors result;
  for (std::size_t i = 0; i < endmembers_.size(); ++i) {
    const Eigen::Matrix3d dM0dn =
        (M0 / 3. + dM0dp[i] - mean_dM0) / std::pow(n, 2. / 3.);
    const Eigen::Matrix3d dFdn =
        (-get_unrotated_dFdf_T() + dFdp[i] - mean_dF) / n;
    result.push_back(dFdn * M0 + F * dM0dn);
  }
  return result;
}
CompositionTensors AnisotropicSolution::get_dMdn_fixed_VT() const {
  auto result = get_unrotated_dMdn_fixed_VT();
  const auto &rotation = get_rotation_matrix();
  for (auto &derivative : result)
    derivative = (rotation * derivative).eval();
  return result;
}
CompositionTensors AnisotropicSolution::get_depsdn_fixed_VT() const {
  auto result = get_dMdn_fixed_VT();
  const Eigen::Matrix3d invM = get_cell_vectors().transpose().inverse();
  for (auto &derivative : result) {
    const Eigen::Matrix3d L = derivative * invM;
    derivative = 0.5 * (L + L.transpose());
  }
  return result;
}

#define SOLUTION_PROPERTY(name)                                                \
  double AnisotropicSolution::compute_##name() const {                         \
    return Solution::compute_##name();                                         \
  }
SOLUTION_PROPERTY(molar_internal_energy)
SOLUTION_PROPERTY(molar_gibbs)
SOLUTION_PROPERTY(molar_helmholtz)
SOLUTION_PROPERTY(molar_mass)
SOLUTION_PROPERTY(molar_volume)
SOLUTION_PROPERTY(density)
SOLUTION_PROPERTY(molar_entropy)
SOLUTION_PROPERTY(molar_enthalpy)
SOLUTION_PROPERTY(isothermal_bulk_modulus_reuss)
SOLUTION_PROPERTY(isentropic_bulk_modulus_reuss)
SOLUTION_PROPERTY(isothermal_compressibility_reuss)
SOLUTION_PROPERTY(isentropic_compressibility_reuss)
SOLUTION_PROPERTY(p_wave_velocity)
SOLUTION_PROPERTY(bulk_sound_velocity)
SOLUTION_PROPERTY(shear_wave_velocity)
SOLUTION_PROPERTY(grueneisen_parameter)
SOLUTION_PROPERTY(thermal_expansivity)
SOLUTION_PROPERTY(molar_heat_capacity_v)
SOLUTION_PROPERTY(molar_heat_capacity_p)
SOLUTION_PROPERTY(isentropic_thermal_gradient)
#undef SOLUTION_PROPERTY
types::FormulaMap AnisotropicSolution::compute_formula() const {
  return Solution::compute_formula();
}
double AnisotropicSolution::compute_shear_modulus() const {
  return AnisotropicMineral::compute_shear_modulus();
}
double AnisotropicSolution::compute_molar_volume_unmodified() const {
  throw exceptions::NotImplementedError("AnisotropicSolution",
                                        "molar_volume_unmodified");
}
std::pair<double, double>
AnisotropicSolution::compute_deformation_alpha_beta() const {
  return {scalar_solution_->get_thermal_expansivity(),
          scalar_solution_->get_isothermal_compressibility_reuss()};
}
} // namespace burnman
