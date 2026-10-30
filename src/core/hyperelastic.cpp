// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/hyperelastic.hpp"
#include "burnman/utils/exceptions.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>

namespace burnman {
namespace {
types::Matrix6d inverse(const types::Matrix6d &matrix) {
  Eigen::FullPivLU<types::Matrix6d> lu(matrix);
  if (!matrix.allFinite() || !lu.isInvertible())
    throw std::domain_error(
        "Hyperelastic stiffness or compliance is singular.");
  return lu.inverse();
}
Eigen::Matrix3d strain(Eigen::Index i, double magnitude) {
  return utils::expand_strains(types::Vector6d::Unit(i) * magnitude);
}
} // namespace

HyperelasticMineral::HyperelasticMineral(
    std::shared_ptr<AnisotropicMineral> hydrostatic)
    : hydrostatic_(std::move(hydrostatic)) {
  if (!hydrostatic_ || dynamic_cast<HyperelasticMineral *>(hydrostatic_.get()))
    throw std::invalid_argument(
        "A hydrostatic AnisotropicMineral reference is required.");
  params = hydrostatic_->params;
  set_name(hydrostatic_->get_name());
  set_reference_cell(hydrostatic_->get_cell_vectors_0(),
                     hydrostatic_->get_frame_convention());
}
std::shared_ptr<Material> HyperelasticMineral::copy() const {
  auto result = std::make_shared<HyperelasticMineral>(*this);
  result->hydrostatic_ = probe();
  return result;
}
std::shared_ptr<AnisotropicMineral> HyperelasticMineral::probe() const {
  return std::dynamic_pointer_cast<AnisotropicMineral>(hydrostatic_->copy());
}
void HyperelasticMineral::reset_cache() {
  AnisotropicMineral::reset_cache();
  responses_ = Responses{};
}
void HyperelasticMineral::set_state(double, double) {
  throw exceptions::NotImplementedError(
      "HyperelasticMineral", "Use set_state_with_molar_cell_vectors() to "
                             "specify the full strain state.");
}
void HyperelasticMineral::set_state_with_volume(double, double,
                                                const Eigen::Vector2d &) {
  set_state(0., 0.);
}
const HyperelasticMineral::Geometry &HyperelasticMineral::geometry() const {
  if (!geometry_ || !has_state())
    throw std::runtime_error("Set molar cell vectors and temperature before "
                             "querying hyperelastic properties.");
  return *geometry_;
}
void HyperelasticMineral::set_state_with_molar_cell_vectors(
    const Eigen::Matrix3d &cell, double T) {
  const double volume = cell.determinant();
  if (!cell.allFinite() || !std::isfinite(volume) || volume <= 0. ||
      !std::isfinite(T) || T < 0.)
    throw std::invalid_argument(
        "Require finite cell vectors with positive volume and finite T >= 0.");
  hydrostatic_->set_state_with_volume(volume, T);
  const Eigen::Matrix3d F =
      cell.transpose() *
      hydrostatic_->get_unrotated_cell_vectors().transpose().inverse();
  const auto [rotation, hencky] = utils::polar_decomposition(F, "H");
  Geometry state{cell, rotation.transpose() * cell.transpose(), rotation,
                 hencky, volume};
  // Keep the finite-difference reference pressure separate from mean stress.
  Material::set_state(hydrostatic_->get_pressure(), T);
  geometry_ = std::move(state);
  update_frame();
}
void HyperelasticMineral::set_method(types::EOSType method) {
  hydrostatic_->set_method(method);
  if (geometry_)
    set_state_with_molar_cell_vectors(geometry_->cell, get_temperature());
  else
    reset_cache();
}
void HyperelasticMineral::set_method(std::shared_ptr<EquationOfState> method) {
  hydrostatic_->set_method(std::move(method));
  if (geometry_)
    set_state_with_molar_cell_vectors(geometry_->cell, get_temperature());
  else
    reset_cache();
}
void HyperelasticMineral::set_output_frame(const std::string &frame) {
  if (frame == "current")
    set_output_frame(OutputFrame::current);
  else if (frame == "crystal")
    set_output_frame(OutputFrame::crystal);
  else
    throw std::invalid_argument("Output frame must be current or crystal.");
}
void HyperelasticMineral::set_output_frame(OutputFrame frame) {
  if (frame != OutputFrame::current && frame != OutputFrame::crystal)
    throw std::invalid_argument("Output frame must be current or crystal.");
  frame_ = frame;
  reset_cache();
  if (geometry_)
    update_frame();
}
void HyperelasticMineral::update_frame() {
  crystal_rotation_ = utils::rotation_to_crystallographic_frame(
      geometry_->cell, get_frame_convention());
  rotation_ = frame_ == OutputFrame::current
                  ? geometry_->isochoric_rotation
                  : (crystal_rotation_ * geometry_->isochoric_rotation).eval();
  deformation_gradient_ = get_cell_vectors().transpose() *
                          get_cell_vectors_0().transpose().inverse();
}
OutputFrame HyperelasticMineral::get_output_frame() const { return frame_; }
const std::shared_ptr<AnisotropicMineral> &
HyperelasticMineral::get_hydrostatic() const {
  return hydrostatic_;
}
const Eigen::Matrix3d &HyperelasticMineral::get_current_cell_vectors() const {
  return geometry().cell;
}
const Eigen::Matrix3d &
HyperelasticMineral::get_rotation_to_crystallographic_frame() const {
  geometry();
  return crystal_rotation_;
}
const Eigen::Matrix3d &HyperelasticMineral::get_rotation_matrix() const {
  geometry();
  return rotation_;
}
Eigen::Matrix3d HyperelasticMineral::get_unrotated_cell_vectors() const {
  return geometry().unrotated_M.transpose();
}
Eigen::Matrix3d HyperelasticMineral::get_cell_vectors() const {
  const auto &cell = geometry().cell;
  return frame_ == OutputFrame::current
             ? cell
             : (cell * crystal_rotation_.transpose()).eval();
}
const Eigen::Matrix3d &
HyperelasticMineral::get_deformation_gradient_tensor() const {
  geometry();
  return deformation_gradient_;
}
utils::CellParameters HyperelasticMineral::get_cell_parameters() const {
  return utils::cell_vectors_to_parameters(geometry().cell,
                                           get_frame_convention(), true);
}
const Eigen::Matrix3d &HyperelasticMineral::get_hencky_strain() const {
  return geometry().hencky;
}

double HyperelasticMineral::get_helmholtz_elastic() const {
  if (!responses_.elastic) {
    const auto h = utils::contract_strains(geometry().hencky);
    responses_.elastic =
        .5 * hydrostatic_->get_molar_volume() *
        h.dot(hydrostatic_->get_unrotated_isothermal_stiffness_tensor() * h);
  }
  return *responses_.elastic;
}
double
HyperelasticMineral::elastic_energy_at(const Eigen::Matrix3d &cell, double T,
                                       AnisotropicMineral &hydrostatic) const {
  hydrostatic.set_state_with_volume(cell.determinant(), T);
  const Eigen::Matrix3d F =
      cell.transpose() *
      hydrostatic.get_unrotated_cell_vectors().transpose().inverse();
  const auto H = utils::polar_decomposition(F, "H").second;
  const auto h = utils::contract_strains(H);
  return .5 * hydrostatic.get_molar_volume() *
         h.dot(hydrostatic.get_unrotated_isothermal_stiffness_tensor() * h);
}
const Eigen::Matrix3d &
HyperelasticMineral::get_cauchy_stress_unrotated() const {
  if (!responses_.stress_unrotated) {
    const auto &M = geometry().unrotated_M;
    const double de = 1.e-4, t = get_temperature(), v = get_molar_volume();
    auto hydrostatic = probe();
    types::Vector6d stress;
    for (Eigen::Index i = 0; i < 6; ++i) {
      const Eigen::Matrix3d E = strain(i, de);
      const double plus =
          elastic_energy_at((E.exp() * M).transpose(), t, *hydrostatic);
      const double minus =
          elastic_energy_at(((-E).exp() * M).transpose(), t, *hydrostatic);
      stress[i] = (plus - minus) / (2. * de * v);
    }
    responses_.stress_unrotated =
        utils::expand_stresses(stress) -
        hydrostatic_->get_pressure() * Eigen::Matrix3d::Identity();
  }
  return *responses_.stress_unrotated;
}
const Eigen::Matrix3d &HyperelasticMineral::get_cauchy_stress_tensor() const {
  if (!responses_.stress)
    responses_.stress = get_rotation_matrix() * get_cauchy_stress_unrotated() *
                        get_rotation_matrix().transpose();
  return *responses_.stress;
}
std::optional<Eigen::Matrix3d> HyperelasticMineral::get_cauchy_stress() const {
  return get_cauchy_stress_tensor();
}
double HyperelasticMineral::get_pressure() const {
  return -get_cauchy_stress_tensor().trace() / 3.;
}

const types::Matrix6d &
HyperelasticMineral::get_unrotated_isothermal_stiffness_tensor() const {
  if (!responses_.unrotated_CT) {
    const auto &M = geometry().unrotated_M;
    const double de = 1.e-4, t = get_temperature(), v = get_molar_volume();
    auto hydrostatic = probe();
    types::Matrix6d stiffness;
    for (Eigen::Index i = 0; i < 6; ++i)
      for (Eigen::Index j = 0; j < 6; ++j) {
        const Eigen::Matrix3d Ei = strain(i, de), Ej = strain(j, de);
        const double pp = elastic_energy_at(
            (Ei.exp() * Ej.exp() * M).transpose(), t, *hydrostatic);
        const double pm = elastic_energy_at(
            (Ei.exp() * (-Ej).exp() * M).transpose(), t, *hydrostatic);
        const double mp = elastic_energy_at(
            ((-Ei).exp() * Ej.exp() * M).transpose(), t, *hydrostatic);
        const double mm = elastic_energy_at(
            ((-Ei).exp() * (-Ej).exp() * M).transpose(), t, *hydrostatic);
        stiffness(i, j) = (pp - pm - mp + mm) / (4. * de * de * v);
      }
    stiffness.topLeftCorner<3, 3>().array() +=
        hydrostatic_->get_isothermal_bulk_modulus_reuss();
    const auto elastic_stress = utils::contract_stresses(
        get_cauchy_stress_unrotated() +
        hydrostatic_->get_pressure() * Eigen::Matrix3d::Identity());
    for (Eigen::Index j = 0; j < 3; ++j)
      stiffness.col(j) -= elastic_stress;
    responses_.unrotated_CT = stiffness;
  }
  return *responses_.unrotated_CT;
}
const types::Matrix6d &
HyperelasticMineral::get_isothermal_stiffness_tensor() const {
  if (!responses_.CT)
    responses_.CT = utils::contract_stiffnesses(utils::rotate_elastic_tensor(
        utils::voigt_notation_to_stiffness_tensor(
            get_unrotated_isothermal_stiffness_tensor()),
        get_rotation_matrix()));
  return *responses_.CT;
}
const types::Matrix6d &
HyperelasticMineral::get_isothermal_compliance_tensor() const {
  if (!responses_.ST)
    responses_.ST = inverse(get_isothermal_stiffness_tensor());
  return *responses_.ST;
}
double HyperelasticMineral::temperature_step() const {
  if (!(get_temperature() > 0.))
    throw std::domain_error(
        "Hyperelastic thermal derivatives require positive temperature.");
  return std::min(.1, .5 * get_temperature());
}
double HyperelasticMineral::compute_molar_entropy() const {
  if (!responses_.entropy) {
    const auto &cell = geometry().cell;
    const double t = get_temperature(), dt = temperature_step();
    auto hydrostatic = probe();
    const double plus = elastic_energy_at(cell, t + dt, *hydrostatic);
    const double minus = elastic_energy_at(cell, t - dt, *hydrostatic);
    responses_.entropy =
        hydrostatic_->get_molar_entropy() - (plus - minus) / (2. * dt);
  }
  return *responses_.entropy;
}
double HyperelasticMineral::get_molar_isometric_heat_capacity() const {
  if (!responses_.Ceps) {
    const auto &cell = geometry().cell;
    const double t = get_temperature(), dt = temperature_step();
    auto hydrostatic = probe();
    const double plus = elastic_energy_at(cell, t + dt, *hydrostatic);
    const double minus = elastic_energy_at(cell, t - dt, *hydrostatic);
    responses_.Ceps =
        hydrostatic_->get_molar_heat_capacity_v() -
        t * (plus - 2. * get_helmholtz_elastic() + minus) / (dt * dt);
  }
  return *responses_.Ceps;
}
const Eigen::Matrix3d &
HyperelasticMineral::get_thermal_stress_unrotated() const {
  if (!responses_.pi_unrotated) {
    const auto &M = geometry().unrotated_M;
    const double t = get_temperature(), dt = temperature_step(), de = 1.e-4;
    auto hydrostatic = probe();
    types::Vector6d pi;
    for (Eigen::Index i = 0; i < 6; ++i) {
      const Eigen::Matrix3d E = strain(i, de);
      const Eigen::Matrix3d plus = (E.exp() * M).transpose(),
                            minus = ((-E).exp() * M).transpose();
      const double pp = elastic_energy_at(plus, t + dt, *hydrostatic);
      const double pm = elastic_energy_at(plus, t - dt, *hydrostatic);
      const double mp = elastic_energy_at(minus, t + dt, *hydrostatic);
      const double mm = elastic_energy_at(minus, t - dt, *hydrostatic);
      pi[i] = ((pp - pm) - (mp - mm)) / (4. * de * dt * get_molar_volume());
    }
    responses_.pi_unrotated =
        utils::expand_stresses(pi) -
        hydrostatic_->get_thermal_expansivity() *
            hydrostatic_->get_isothermal_bulk_modulus_reuss() *
            Eigen::Matrix3d::Identity();
  }
  return *responses_.pi_unrotated;
}
Eigen::Matrix3d HyperelasticMineral::get_thermal_stress_tensor() const {
  return get_rotation_matrix() * get_thermal_stress_unrotated() *
         get_rotation_matrix().transpose();
}
const Eigen::Matrix3d &
HyperelasticMineral::get_thermal_expansivity_tensor() const {
  if (!responses_.alpha)
    responses_.alpha = utils::expand_strains(
        -get_isothermal_compliance_tensor() *
        utils::contract_stresses(get_thermal_stress_tensor()));
  return *responses_.alpha;
}
double HyperelasticMineral::get_molar_isostress_heat_capacity() const {
  return get_molar_isometric_heat_capacity() -
         get_molar_volume() * get_temperature() *
             (get_thermal_expansivity_tensor().array() *
              get_thermal_stress_tensor().array())
                 .sum();
}
const types::Matrix6d &
HyperelasticMineral::get_isentropic_compliance_tensor() const {
  if (!responses_.SS) {
    const auto &s = get_isothermal_compliance_tensor();
    const auto pi = utils::contract_stresses(get_thermal_stress_tensor());
    const types::Vector6d left = -s * pi, right = -s.transpose() * pi;
    responses_.SS = s - get_molar_volume() * get_temperature() /
                            get_molar_isostress_heat_capacity() * left *
                            right.transpose();
  }
  return *responses_.SS;
}
const types::Matrix6d &
HyperelasticMineral::get_isentropic_stiffness_tensor() const {
  if (!responses_.CS)
    responses_.CS = inverse(get_isentropic_compliance_tensor());
  return *responses_.CS;
}

AnisotropicMineral::PsiState HyperelasticMineral::compute_psi_state() const {
  throw exceptions::NotImplementedError(
      "HyperelasticMineral",
      "Psi derivatives belong to the hydrostatic reference.");
}
double HyperelasticMineral::compute_molar_volume() const {
  return geometry().volume;
}
double HyperelasticMineral::compute_molar_volume_unmodified() const {
  return geometry().volume;
}
double HyperelasticMineral::compute_molar_mass() const {
  return hydrostatic_->get_molar_mass();
}
double HyperelasticMineral::compute_molar_helmholtz() const {
  return hydrostatic_->get_molar_helmholtz() + get_helmholtz_elastic();
}
double HyperelasticMineral::compute_molar_internal_energy() const {
  return get_molar_helmholtz() + get_temperature() * get_molar_entropy();
}
double HyperelasticMineral::compute_molar_gibbs() const {
  return get_molar_helmholtz() + get_pressure() * get_molar_volume();
}
double HyperelasticMineral::compute_molar_enthalpy() const {
  return get_molar_internal_energy() + get_pressure() * get_molar_volume();
}
double HyperelasticMineral::compute_isothermal_compressibility_reuss() const {
  return get_isothermal_compliance_tensor().topLeftCorner<3, 3>().sum();
}
double HyperelasticMineral::compute_isentropic_compressibility_reuss() const {
  return get_isentropic_compliance_tensor().topLeftCorner<3, 3>().sum();
}
double HyperelasticMineral::compute_isothermal_bulk_modulus_reuss() const {
  return 1. / get_isothermal_compressibility_reuss();
}
double HyperelasticMineral::compute_isentropic_bulk_modulus_reuss() const {
  return 1. / get_isentropic_compressibility_reuss();
}
double HyperelasticMineral::compute_thermal_expansivity() const {
  return hydrostatic_->get_thermal_expansivity();
}
double HyperelasticMineral::compute_molar_heat_capacity_p() const {
  return hydrostatic_->get_molar_heat_capacity_p();
}
double HyperelasticMineral::compute_molar_heat_capacity_v() const {
  return get_molar_heat_capacity_p() -
         get_molar_volume() * get_temperature() *
             std::pow(get_thermal_expansivity(), 2) *
             get_isothermal_bulk_modulus_reuss();
}
} // namespace burnman
