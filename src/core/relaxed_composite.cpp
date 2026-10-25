// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/relaxed_composite.hpp"
#include "burnman/core/mineral.hpp"
#include "burnman/core/solution.hpp"
#include "burnman/tools/equilibration/equality_constraint_helpers.hpp"
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/equilibration/equilibrate_impl.hpp"
#include <Eigen/SVD>
#include <cmath>

namespace burnman {
RelaxedComposite::RelaxedComposite(std::shared_ptr<Assemblage> composite,
                                   const Eigen::MatrixXd &vectors)
    : unrelaxed_(std::move(composite)), dndq_(vectors.transpose()) {
  if (!unrelaxed_ || unrelaxed_->get_n_phases() == 0)
    throw std::invalid_argument("RelaxedComposite requires an assemblage.");
  for (const auto &phase : unrelaxed_->get_phases())
    if (!phase || (!dynamic_cast<const Mineral *>(phase.get()) &&
                   !dynamic_cast<const Solution *>(phase.get())))
      throw std::invalid_argument("RelaxedComposite requires a flat assemblage "
                                  "of minerals and solutions.");
  if (dndq_.rows() != unrelaxed_->get_n_endmembers() || !dndq_.allFinite())
    throw std::invalid_argument(
        "Relaxation vectors must be finite rows with one entry per endmember.");
  const Eigen::MatrixXd delta =
      unrelaxed_->get_stoichiometric_matrix().transpose() * dndq_;
  if ((delta.array().abs() > 1.e-8).any())
    throw std::invalid_argument("Relaxation vectors must be isochemical.");
  add_phases(unrelaxed_->get_phases()); // Share phases, as in Python BurnMan.
  if (unrelaxed_->get_molar_fractions().size())
    set_fractions(unrelaxed_->get_molar_fractions());
  set_n_moles(unrelaxed_->has_n_moles() ? unrelaxed_->get_n_moles() : 1.);
  set_averaging_scheme(types::AveragingType::VRH);
  set_name(unrelaxed_->get_name());
}

std::shared_ptr<Material> RelaxedComposite::copy() const {
  // Copy the current assemblage independently and share those copied phases
  // between the new relaxed wrapper and its ordinary, unrelaxed assemblage.
  auto frozen = std::dynamic_pointer_cast<Assemblage>(Assemblage::copy());
  frozen->reset_cache();
  auto result = std::make_shared<RelaxedComposite>(frozen, dndq_.transpose());
  static_cast<Assemblage &>(*result) = *frozen;
  return result;
}

void RelaxedComposite::reset_cache() {
  Assemblage::reset_cache();
  derivatives_.reset();
}
void RelaxedComposite::set_state(double p, double t) {
  set_state(p, t, unrelaxed_->get_formula());
}
void RelaxedComposite::set_state(double p, double t,
                                 const types::FormulaMap &composition) {
  if (!std::isfinite(p) || !std::isfinite(t) || t < 0.)
    throw std::invalid_argument(
        "Pressure and temperature must be finite, T>=0.");
  const auto bulk = composition; // The input may refer to the formula cache.
  const auto &elements = unrelaxed_->get_elements();
  double total = 0.;
  for (const auto &[element, amount] : bulk) {
    if (!std::isfinite(amount) || amount < 0. ||
        std::find(elements.begin(), elements.end(), element) == elements.end())
      throw std::invalid_argument(
          "Invalid bulk composition for the assemblage.");
    total += amount;
  }
  if (!(total > 0.))
    throw std::invalid_argument(
        "Bulk composition must have a positive amount.");
  Eigen::VectorXd amounts(unrelaxed_->get_n_elements());
  for (Eigen::Index i = 0; i < amounts.size(); ++i) {
    const auto entry = bulk.find(elements[static_cast<std::size_t>(i)]);
    if (entry == bulk.end())
      throw std::invalid_argument(
          "Bulk composition must specify each assemblage element.");
    amounts[i] = entry->second;
  }
  if (((unrelaxed_->get_compositional_null_basis() * amounts).array().abs() >
       1.e-12)
          .any())
    throw std::invalid_argument(
        "Bulk composition is outside the assemblage's compositional space.");
  reset_cache();
  unrelaxed_->set_state(p, t);
  // Physical-unit equivalents of Python BurnMan's default step tolerances.
  Eigen::VectorXd tolerances =
      Eigen::VectorXd::Constant(unrelaxed_->get_n_endmembers() + 2, 1.e-9);
  tolerances[0] = 1.;
  tolerances[1] = 1.e-6;
  const auto constraints = equilibration::make_constraint_list(
      equilibration::make_constraint<equilibration::PressureConstraint>(p),
      equilibration::make_constraint<equilibration::TemperatureConstraint>(t));
  const auto result =
      equilibration::equilibrate(bulk, *unrelaxed_, constraints, {}, 1.e-3,
                                 false, false, 100, false, tolerances);
  const auto &solution = result.sol_array[0];
  if (!solution.success)
    throw std::runtime_error(
        "Equilibration failed in RelaxedComposite.set_state(): " +
        solution.message);
  copy_state_from_unrelaxed();
}
void RelaxedComposite::copy_state_from_unrelaxed() {
  Assemblage::set_state(unrelaxed_->get_pressure(),
                        unrelaxed_->get_temperature());
  set_fractions(unrelaxed_->get_molar_fractions());
  set_n_moles(unrelaxed_->get_n_moles());
}

const RelaxedComposite::Derivatives &RelaxedComposite::derivatives() const {
  if (derivatives_)
    return *derivatives_;
  if (!has_state() || !unrelaxed_->has_state())
    throw std::runtime_error("Set or copy the relaxed composite state first.");
  const double n = unrelaxed_->get_n_moles(),
               v = unrelaxed_->get_molar_volume() * n,
               t = unrelaxed_->get_temperature();
  if (!(t > 0.) || !(n > 0.))
    throw std::domain_error(
        "Relaxed derivatives require positive T and n_moles.");
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(dndq_.rows(), dndq_.rows());
  const auto counts = unrelaxed_->get_endmembers_per_phase();
  const auto amounts = (unrelaxed_->get_molar_fractions() * n).eval();
  Eigen::Index start = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (const auto solution = unrelaxed_->get_phase<Solution>(i)) {
      if (!(amounts[static_cast<Eigen::Index>(i)] > 0.))
        throw std::domain_error(
            "Relaxed derivatives require positive solution amounts.");
      h.block(start, start, counts[i], counts[i]) =
          solution->get_gibbs_hessian() / amounts[static_cast<Eigen::Index>(i)];
    }
    start += counts[i];
  }
  Derivatives d;
  d.structural = dndq_.transpose() * h * dndq_;
  d.coupling.resize(dndq_.cols(), 2);
  d.coupling.col(0) =
      dndq_.transpose() * unrelaxed_->get_endmember_partial_volumes().matrix();
  d.coupling.col(1) = -dndq_.transpose() *
                      unrelaxed_->get_endmember_partial_entropies().matrix();
  if (!d.structural.allFinite() || !d.coupling.allFinite())
    throw std::domain_error("Non-finite Gibbs relaxation derivatives.");
  d.susceptibility = Eigen::MatrixXd::Zero(dndq_.cols(), dndq_.cols());
  if (dndq_.cols()) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        d.structural, Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.setThreshold(
        1.e-22); // np.linalg.pinv(rcond=1.e-22) in RelaxedComposite.
    d.susceptibility =
        svd.solve(Eigen::MatrixXd::Identity(dndq_.cols(), dndq_.cols()));
  }
  d.dqdz = -d.susceptibility * d.coupling;
  const double alpha_v = v * unrelaxed_->get_thermal_expansivity();
  d.frozen << -v * unrelaxed_->get_isothermal_compressibility_reuss(), alpha_v,
      alpha_v, -n * unrelaxed_->get_molar_heat_capacity_p() / t;
  d.relaxed = d.frozen + d.coupling.transpose() * d.dqdz;
  derivatives_ = std::move(d);
  return *derivatives_;
}
const Eigen::MatrixXd &RelaxedComposite::get_dqdz_relaxed() const {
  return derivatives().dqdz;
}
const Eigen::MatrixXd &RelaxedComposite::get_structural_hessian() const {
  return derivatives().structural;
}
const Eigen::MatrixXd &RelaxedComposite::get_susceptibility() const {
  return derivatives().susceptibility;
}
const Eigen::MatrixXd &RelaxedComposite::get_coupling() const {
  return derivatives().coupling;
}
const Eigen::Matrix2d &RelaxedComposite::get_frozen_hessian() const {
  return derivatives().frozen;
}
const Eigen::Matrix2d &RelaxedComposite::get_relaxed_hessian() const {
  return derivatives().relaxed;
}
double RelaxedComposite::compute_isothermal_compressibility_reuss() const {
  return -get_relaxed_hessian()(0, 0) / (get_molar_volume() * get_n_moles());
}
double RelaxedComposite::compute_isothermal_bulk_modulus_reuss() const {
  return 1. / get_isothermal_compressibility_reuss();
}
double RelaxedComposite::compute_thermal_expansivity() const {
  return get_relaxed_hessian()(0, 1) / (get_molar_volume() * get_n_moles());
}
double RelaxedComposite::compute_molar_heat_capacity_p() const {
  return -get_relaxed_hessian()(1, 1) * get_temperature() / get_n_moles();
}
} // namespace burnman
