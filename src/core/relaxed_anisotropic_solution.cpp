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

#include "burnman/core/relaxed_anisotropic_solution.hpp"
#include "burnman/utils/index_utils.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <cmath>
#include <nlopt.hpp>
#include <stdexcept>

namespace burnman {
namespace {
const AnisotropicSolution &
require_solution(const std::shared_ptr<AnisotropicSolution> &solution) {
  if (!solution)
    throw std::invalid_argument(
        "An unrelaxed anisotropic solution is required.");
  return *solution;
}
Eigen::MatrixXd pseudoinverse(const Eigen::MatrixXd &matrix) {
  if (!matrix.allFinite())
    throw std::domain_error("Relaxation derivatives must be finite.");
  if (!matrix.rows() || !matrix.cols())
    return Eigen::MatrixXd::Zero(matrix.cols(), matrix.rows());
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU |
                                                    Eigen::ComputeThinV);
  svd.setThreshold(1.e-15); // numpy.linalg.pinv default
  return svd.solve(Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows()));
}
} // namespace

RelaxedAnisotropicSolution::RelaxedAnisotropicSolution(
    std::shared_ptr<AnisotropicSolution> solution,
    const Eigen::MatrixXd &relaxation_vectors,
    const Eigen::MatrixXd &unrelaxed_vectors)
    : Material(require_solution(solution)),
      AnisotropicSolution(*std::dynamic_pointer_cast<AnisotropicSolution>(
          solution->AnisotropicSolution::copy())),
      unrelaxed_(std::move(solution)), dndq_(relaxation_vectors.transpose()),
      dndx_(unrelaxed_vectors.transpose()) {
  const auto n = get_n_endmembers();
  if (dndq_.rows() != n || dndx_.rows() != n || !dndx_.cols() ||
      dndq_.cols() + dndx_.cols() != n || !dndq_.allFinite() ||
      !dndx_.allFinite() || dndq_.colwise().sum().norm() > 1.e-12 ||
      (dndx_.colwise().sum().array() - 1.).matrix().norm() > 1.e-12 ||
      (get_stoichiometric_matrix().transpose() * dndq_).norm() > 1.e-12)
    throw std::invalid_argument(
        "Invalid isochemical relaxation and bulk-composition basis.");
  Eigen::MatrixXd basis(n, n);
  basis << dndq_, dndx_;
  if (basis.fullPivLu().rank() != n)
    throw std::invalid_argument(
        "Relaxation and bulk vectors must span the endmembers.");
  inverse_basis_ = basis.inverse();
  q_initial_ = Eigen::VectorXd::Constant(dndq_.cols(), .001);
  if (get_molar_fractions().size()) {
    const Eigen::VectorXd coordinates =
        inverse_basis_ * get_molar_fractions().matrix();
    q_initial_ = coordinates.head(dndq_.cols());
    independent_ = coordinates.tail(dndx_.cols()).array();
    if (has_state())
      set_state(get_pressure(), get_temperature());
  }
}

std::shared_ptr<Material> RelaxedAnisotropicSolution::copy() const {
  auto result = std::make_shared<RelaxedAnisotropicSolution>(*this);
  const auto base = std::dynamic_pointer_cast<AnisotropicSolution>(
      AnisotropicSolution::copy());
  static_cast<AnisotropicSolution &>(*result) = *base;
  result->unrelaxed_ = std::dynamic_pointer_cast<AnisotropicSolution>(
      unrelaxed_->AnisotropicSolution::copy());
  return result;
}
void RelaxedAnisotropicSolution::reset_cache() {
  AnisotropicSolution::reset_cache();
  responses_.reset();
}
void RelaxedAnisotropicSolution::require_composition() const {
  if (independent_.size() != dndx_.cols())
    throw std::runtime_error(
        "Call set_composition() with the independent fractions first.");
}
void RelaxedAnisotropicSolution::set_q_initial(const Eigen::VectorXd &q) {
  if (q.size() != dndq_.cols() || !q.allFinite())
    throw std::invalid_argument(
        "q_initial must be finite and match the relaxation modes.");
  q_initial_ = q;
}
void RelaxedAnisotropicSolution::set_composition(
    const Eigen::ArrayXd &fractions) {
  set_composition(fractions, std::nullopt, true);
}
void RelaxedAnisotropicSolution::set_composition(
    const Eigen::ArrayXd &fractions, const std::optional<Eigen::VectorXd> &q,
    bool relaxed) {
  if (fractions.size() != dndx_.cols() || !fractions.isFinite().all() ||
      std::abs(fractions.sum() - 1.) > 1.e-12 || (fractions < 0.).any())
    throw std::invalid_argument(
        "Independent fractions must be finite, nonnegative and sum to one.");
  const Eigen::VectorXd initial = q ? *q : q_initial_;
  if (initial.size() != dndq_.cols() || !initial.allFinite())
    throw std::invalid_argument(
        "q_initial must be finite and match the relaxation modes.");
  const Eigen::ArrayXd n =
      (dndx_ * fractions.matrix() + dndq_ * initial).array();
  const auto sites = get_endmember_occupancies().matrix().transpose().eval();
  if ((sites * n.matrix()).minCoeff() < -1.e-10)
    throw std::invalid_argument(
        "Initial relaxation parameters give negative site occupancies.");
  independent_ = fractions;
  q_initial_ = initial;
  unrelaxed_->set_composition(n);
  AnisotropicSolution::set_composition(n);
  if (has_state() && relaxed)
    relax_at_PTX();
}
void RelaxedAnisotropicSolution::synchronize(const Eigen::ArrayXd &fractions,
                                             double P, double T) {
  unrelaxed_->set_composition(fractions);
  unrelaxed_->set_state(P, T);
  AnisotropicSolution::set_composition(fractions);
  AnisotropicSolution::set_state(P, T);
  const Eigen::VectorXd coordinates = inverse_basis_ * fractions.matrix();
  q_initial_ = coordinates.head(dndq_.cols());
  independent_ = coordinates.tail(dndx_.cols()).array();
}
void RelaxedAnisotropicSolution::set_state(double P, double T) {
  set_state(P, T, true);
}
void RelaxedAnisotropicSolution::set_state(double P, double T, bool relaxed) {
  require_composition();
  unrelaxed_->set_state(P, T);
  if (relaxed)
    minimize(false, 0., P, T, Eigen::Vector2d(0., 1.e10));
  else
    synchronize(unrelaxed_->get_molar_fractions(), P, T);
}
void RelaxedAnisotropicSolution::set_state_with_volume(
    double volume, double T, const Eigen::Vector2d &guesses) {
  require_composition();
  if (!std::isfinite(volume) || volume <= 0. || !std::isfinite(T) || T < 0. ||
      !guesses.allFinite() || guesses[0] >= guesses[1])
    throw std::invalid_argument(
        "Require V > 0, T >= 0 and increasing finite pressure guesses.");
  minimize(true, volume, 0., T, guesses);
}
void RelaxedAnisotropicSolution::relax_at_PTX() {
  require_composition();
  if (!has_state())
    throw std::runtime_error("Set pressure and temperature before relaxation.");
  set_state(get_pressure(), get_temperature());
}
void RelaxedAnisotropicSolution::set_method(types::EOSType method) {
  unrelaxed_->set_method(method);
  AnisotropicSolution::set_method(method);
}
void RelaxedAnisotropicSolution::set_method(
    std::shared_ptr<EquationOfState> method) {
  unrelaxed_->set_method(method);
  AnisotropicSolution::set_method(std::move(method));
}

void RelaxedAnisotropicSolution::minimize(bool fixed_volume, double volume,
                                          double P, double T,
                                          const Eigen::Vector2d &guesses) {
  struct Objective {
    std::shared_ptr<Solution> scalar;
    Eigen::VectorXd base;
    Eigen::MatrixXd Q;
    double volume, P, T, offset = 0., scale;
    bool fixed_volume;
    Eigen::Vector2d guesses;
    double evaluate(const Eigen::VectorXd &dq,
                    Eigen::VectorXd *gradient = nullptr) {
      scalar->set_composition((base + Q * dq).array());
      if (fixed_volume) {
        scalar->set_state_with_volume(volume, T, guesses);
        guesses << scalar->get_pressure() - 1.e9, scalar->get_pressure() + 1.e9;
      } else
        scalar->set_state(P, T);
      if (gradient)
        *gradient = Q.transpose() * scalar->get_partial_gibbs().matrix();
      return fixed_volume ? scalar->get_molar_helmholtz()
                          : scalar->get_molar_gibbs();
    }
    Eigen::MatrixXd hessian() const {
      Eigen::MatrixXd h = scalar->get_gibbs_hessian();
      if (fixed_volume) {
        const Eigen::VectorXd v = scalar->get_partial_volumes().matrix();
        h += scalar->get_isothermal_bulk_modulus_reuss() /
             scalar->get_molar_volume() * v * v.transpose();
      }
      return Q.transpose() * h * Q;
    }
  } data{std::dynamic_pointer_cast<Solution>(
             unrelaxed_->get_scalar_solution()->copy()),
         unrelaxed_->get_molar_fractions().matrix(),
         dndq_,
         volume,
         P,
         T,
         0.,
         std::max(1000., 8.314462618 * T),
         fixed_volume,
         guesses};
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(dndq_.cols());
  data.offset = data.evaluate(delta);
  const Eigen::MatrixXd sites =
      get_endmember_occupancies().matrix().transpose();
  if (dndq_.cols()) {
    nlopt::opt optimizer(
        nlopt::LD_SLSQP,
        static_cast<unsigned int>(utils::checked_int(dndq_.cols())));
    optimizer.set_min_objective(
        [](const std::vector<double> &q, std::vector<double> &gradient,
           void *context) {
          auto &d = *static_cast<Objective *>(context);
          Eigen::VectorXd g;
          const double energy =
              d.evaluate(Eigen::Map<const Eigen::VectorXd>(
                             q.data(), static_cast<Eigen::Index>(q.size())),
                         gradient.empty() ? nullptr : &g);
          if (!gradient.empty())
            Eigen::Map<Eigen::VectorXd>(gradient.data(), g.size()) =
                g / d.scale;
          return (energy - d.offset) / d.scale;
        },
        &data);
    struct Inequality {
      Eigen::VectorXd a;
      double b;
    };
    std::vector<Inequality> inequalities;
    for (Eigen::Index i = 0; i < sites.rows(); ++i) {
      const Eigen::VectorXd a = (-sites.row(i) * dndq_).transpose();
      if (a.norm() > 1.e-14)
        inequalities.push_back({a, -sites.row(i).dot(data.base)});
    }
    for (auto &constraint : inequalities)
      optimizer.add_inequality_constraint(
          [](const std::vector<double> &q, std::vector<double> &gradient,
             void *context) {
            const auto &c = *static_cast<const Inequality *>(context);
            if (!gradient.empty())
              std::copy(c.a.data(), c.a.data() + c.a.size(), gradient.begin());
            return c.a.dot(Eigen::Map<const Eigen::VectorXd>(
                       q.data(), static_cast<Eigen::Index>(q.size()))) +
                   c.b;
          },
          &constraint, 1.e-12);
    optimizer.set_xtol_abs(1.e-13);
    optimizer.set_ftol_abs(1.e-14);
    optimizer.set_maxeval(1000);
    std::vector<double> q(static_cast<std::size_t>(dndq_.cols()), 0.);
    // At a symmetry-breaking transition the previous minimum can become a
    // stationary maximum. Seed the unstable mode so a gradient method escapes.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> modes(data.hessian());
    if (modes.eigenvalues()[0] < -1.e-8) {
      Eigen::VectorXd seed = 1.e-3 * modes.eigenvectors().col(0);
      seed /= std::max(1., (dndq_ * seed).norm() / 1.e-3);
      for (int attempt = 0; attempt < 20; ++attempt, seed *= -.5) {
        if ((sites * (data.base + dndq_ * seed)).minCoeff() >= -1.e-12) {
          Eigen::Map<Eigen::VectorXd>(q.data(), dndq_.cols()) = seed;
          break;
        }
      }
    }
    double energy = 0.;
    try {
      optimizer.optimize(q, energy);
    } catch (const nlopt::roundoff_limited
                 &) { /* Refine and validate the returned minimum below. */
    }
    delta = Eigen::Map<const Eigen::VectorXd>(q.data(), dndq_.cols());
    // Analytic Newton refinement resolves composition changes in derivative
    // tests.
    for (int iteration = 0; iteration < 8; ++iteration) {
      Eigen::VectorXd gradient;
      const double before = data.evaluate(delta, &gradient);
      const Eigen::VectorXd step = -pseudoinverse(data.hessian()) * gradient;
      if (step.norm() < 1.e-13)
        break;
      double damping = 1.;
      bool accepted = false;
      for (int line = 0; line < 20; ++line, damping *= .5) {
        const Eigen::VectorXd candidate = delta + damping * step;
        if ((sites * (data.base + dndq_ * candidate)).minCoeff() < -1.e-12)
          continue;
        Eigen::VectorXd next_gradient;
        const double after = data.evaluate(candidate, &next_gradient);
        if (after <= before + 1.e-8 && next_gradient.norm() < gradient.norm()) {
          delta = candidate;
          accepted = true;
          break;
        }
      }
      if (!accepted)
        break;
    }
  }
  data.evaluate(delta);
  const Eigen::ArrayXd fractions = data.scalar->get_molar_fractions();
  if (!fractions.isFinite().all() ||
      (sites * fractions.matrix()).minCoeff() < -1.e-10)
    throw std::runtime_error("Relaxation returned an infeasible composition.");
  synchronize(fractions, data.scalar->get_pressure(), T);
}

const RelaxedAnisotropicSolution::Responses &
RelaxedAnisotropicSolution::responses() const {
  require_composition();
  if (!has_state() || !(get_temperature() > 0.))
    throw std::domain_error(
        "Relaxed response tensors require a state with positive temperature.");
  if (!responses_) {
    Responses d;
    const auto &frozen = *unrelaxed_;
    const auto &scalar = *frozen.get_scalar_solution();
    const double v = get_molar_volume(), t = get_temperature();
    const double kt = scalar.get_isothermal_bulk_modulus_reuss();
    const auto &ct = frozen.get_isothermal_stiffness_tensor();
    const Eigen::VectorXd partial_v = scalar.get_partial_volumes().matrix();
    d.dSdq =
        dndq_.transpose() * (scalar.get_partial_entropies().matrix() -
                             scalar.get_thermal_expansivity() * kt * partial_v);
    d.dPdq = kt / v * dndq_.transpose() * partial_v;
    Eigen::MatrixXd h =
        scalar.get_gibbs_hessian() + kt / v * partial_v * partial_v.transpose();
    d.structural_VT = dndq_.transpose() * h * dndq_;
    const auto depsdn = frozen.get_depsdn_fixed_VT();
    Eigen::MatrixXd e(6, dndq_.cols());
    for (Eigen::Index q = 0; q < dndq_.cols(); ++q) {
      Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
      for (Eigen::Index i = 0; i < dndq_.rows(); ++i)
        strain += depsdn[static_cast<std::size_t>(i)] * dndq_(i, q);
      d.depsdq.push_back(strain);
      e.col(q) = utils::contract_strains(strain);
    }
    const auto beta = frozen.get_isothermal_compressibility_tensor();
    const double beta_reuss = frozen.get_isothermal_compressibility_reuss();
    d.depsdT = frozen.get_thermal_expansivity_tensor() -
               frozen.get_thermal_expansivity() / beta_reuss * beta;
    types::Vector6d identity = types::Vector6d::Zero();
    identity.head<3>().setOnes();
    const types::Matrix6d projection =
        types::Matrix6d::Identity() -
        utils::contract_strains(beta) * identity.transpose() / beta_reuss;
    d.structural = d.structural_VT + v * e.transpose() * ct * e;
    d.coupling.resize(dndq_.cols(), 7);
    d.coupling.leftCols(6) =
        -v * (d.dPdq * identity.transpose() + e.transpose() * ct * projection);
    d.coupling.col(6) =
        -d.dSdq + v * e.transpose() * ct * utils::contract_strains(d.depsdT);
    d.susceptibility = pseudoinverse(d.structural);
    d.dqdz = -d.susceptibility * d.coupling;
    d.frozen.topLeftCorner<6, 6>() = v * ct;
    d.frozen.topRightCorner<6, 1>() =
        v * utils::contract_stresses(frozen.get_thermal_stress_tensor());
    d.frozen.bottomLeftCorner<1, 6>() =
        d.frozen.topRightCorner<6, 1>().transpose();
    d.frozen(6, 6) = -frozen.get_molar_isometric_heat_capacity() / t;
    d.relaxed = d.frozen + d.coupling.transpose() * d.dqdz;
    d.relaxed = (.5 * (d.relaxed + d.relaxed.transpose())).eval();
    d.stiffness = d.relaxed.topLeftCorner<6, 6>() / v;
    d.compliance = pseudoinverse(d.stiffness);
    const types::Vector6d pi =
        d.relaxed.bottomLeftCorner<1, 6>().transpose() / v;
    d.pi = utils::expand_stresses(pi);
    d.alpha = utils::expand_strains(-d.compliance * pi);
    d.isometric_heat_capacity = -t * d.relaxed(6, 6);
    if (!d.relaxed.allFinite() || !d.alpha.allFinite())
      throw std::domain_error("Non-finite relaxed thermodynamic response.");
    responses_ = std::move(d);
  }
  return *responses_;
}

const std::shared_ptr<AnisotropicSolution> &
RelaxedAnisotropicSolution::get_unrelaxed() const {
  return unrelaxed_;
}
const Eigen::MatrixXd &RelaxedAnisotropicSolution::get_dndq() const {
  return dndq_;
}
const Eigen::MatrixXd &RelaxedAnisotropicSolution::get_dndx() const {
  return dndx_;
}
const Eigen::ArrayXd &
RelaxedAnisotropicSolution::get_unrelaxed_vectors() const {
  return independent_;
}
const Eigen::VectorXd &RelaxedAnisotropicSolution::get_q_initial() const {
  return q_initial_;
}
const CompositionTensors &
RelaxedAnisotropicSolution::get_depsdq_fixed_VT() const {
  return responses().depsdq;
}
const Eigen::Matrix3d &
RelaxedAnisotropicSolution::get_depsdT_fixed_volume() const {
  return responses().depsdT;
}
types::ElasticTensor RelaxedAnisotropicSolution::get_dbarepsdeps() const {
  const Eigen::Matrix3d beta =
      unrelaxed_->get_isothermal_compressibility_tensor() /
      unrelaxed_->get_isothermal_compressibility_reuss();
  types::ElasticTensor result{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          result[i][j][k][l] = double(i == k && j == l) -
                               (k == l ? beta(static_cast<Eigen::Index>(i),
                                              static_cast<Eigen::Index>(j))
                                       : 0.);
  return result;
}
const Eigen::VectorXd &RelaxedAnisotropicSolution::get_dSdq() const {
  return responses().dSdq;
}
const Eigen::VectorXd &RelaxedAnisotropicSolution::get_dPdq() const {
  return responses().dPdq;
}
const Eigen::MatrixXd &
RelaxedAnisotropicSolution::get_structural_hessian_fixed_VT() const {
  return responses().structural_VT;
}
const Eigen::MatrixXd &
RelaxedAnisotropicSolution::get_structural_hessian_fixed_epsT() const {
  return responses().structural;
}
const Eigen::MatrixXd &RelaxedAnisotropicSolution::get_susceptibility() const {
  return responses().susceptibility;
}
const Eigen::MatrixXd &RelaxedAnisotropicSolution::get_coupling() const {
  return responses().coupling;
}
const Eigen::MatrixXd &RelaxedAnisotropicSolution::get_dqdz_relaxed() const {
  return responses().dqdz;
}
const types::Matrix7d &RelaxedAnisotropicSolution::get_frozen_hessian() const {
  return responses().frozen;
}
const types::Matrix7d &RelaxedAnisotropicSolution::get_relaxed_hessian() const {
  return responses().relaxed;
}
const types::Matrix6d &
RelaxedAnisotropicSolution::get_isothermal_stiffness_tensor() const {
  return responses().stiffness;
}
const types::Matrix6d &
RelaxedAnisotropicSolution::get_isothermal_compliance_tensor() const {
  return responses().compliance;
}
const Eigen::Matrix3d &
RelaxedAnisotropicSolution::get_thermal_expansivity_tensor() const {
  return responses().alpha;
}
Eigen::Matrix3d RelaxedAnisotropicSolution::get_thermal_stress_tensor() const {
  return responses().pi;
}
double RelaxedAnisotropicSolution::get_molar_isometric_heat_capacity() const {
  return responses().isometric_heat_capacity;
}
double
RelaxedAnisotropicSolution::compute_isothermal_compressibility_reuss() const {
  return get_isothermal_compressibility_tensor().trace();
}
double
RelaxedAnisotropicSolution::compute_isentropic_compressibility_reuss() const {
  return get_isentropic_compressibility_tensor().trace();
}
double
RelaxedAnisotropicSolution::compute_isothermal_bulk_modulus_reuss() const {
  return 1. / get_isothermal_compressibility_reuss();
}
double
RelaxedAnisotropicSolution::compute_isentropic_bulk_modulus_reuss() const {
  return 1. / get_isentropic_compressibility_reuss();
}
double RelaxedAnisotropicSolution::compute_thermal_expansivity() const {
  return get_thermal_expansivity_tensor().trace();
}
double RelaxedAnisotropicSolution::compute_molar_heat_capacity_p() const {
  return get_molar_isometric_heat_capacity() -
         get_molar_volume() * get_temperature() *
             (get_thermal_expansivity_tensor().array() *
              get_thermal_stress_tensor().array())
                 .sum();
}
} // namespace burnman
