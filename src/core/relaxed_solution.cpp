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

#include "burnman/core/relaxed_solution.hpp"
#include "burnman/utils/index_utils.hpp"
#include <nlopt.hpp>

namespace burnman {
RelaxedSolution::RelaxedSolution(Solution solution,
                                 const Eigen::MatrixXd &relaxation_vectors,
                                 const Eigen::MatrixXd &unrelaxed_vectors)
    : Solution(std::move(solution)), dndq_(relaxation_vectors.transpose()),
      dndx_(unrelaxed_vectors.transpose()) {
  set_solution_model(get_solution_model()->clone());
  const auto n = get_n_endmembers();
  if (dndq_.rows() != n || dndx_.rows() != n || !dndq_.cols() ||
      dndq_.cols() + dndx_.cols() != n || !dndq_.allFinite() ||
      !dndx_.allFinite() || dndq_.colwise().sum().norm() > 1.e-12 ||
      (dndx_.colwise().sum().array() - 1.).matrix().norm() > 1.e-12 ||
      (get_stoichiometric_matrix().transpose() * dndq_).norm() > 1.e-12)
    throw std::invalid_argument("Invalid relaxed-solution composition basis.");
  Eigen::MatrixXd basis(n, n);
  basis << dndq_, dndx_;
  if (basis.fullPivLu().rank() != n)
    throw std::invalid_argument(
        "Relaxation and bulk vectors must span the endmembers.");
  independent_ = basis.fullPivLu()
                     .solve(get_molar_fractions().matrix())
                     .tail(dndx_.cols())
                     .array();
  set_composition(independent_);
}

void RelaxedSolution::set_composition(const Eigen::ArrayXd &fractions) {
  set_composition(fractions, Eigen::VectorXd::Zero(dndq_.cols()), true);
}
void RelaxedSolution::set_composition(const Eigen::ArrayXd &fractions,
                                      const Eigen::VectorXd &q_initial,
                                      bool relaxed) {
  if (fractions.size() != dndx_.cols() || !fractions.allFinite() ||
      std::abs(fractions.sum() - 1.) > 1.e-12 || fractions.minCoeff() < 0. ||
      q_initial.size() != dndq_.cols() || !q_initial.allFinite())
    throw std::invalid_argument(
        "Invalid independent composition or relaxation parameters.");
  Eigen::ArrayXd n = (dndx_ * fractions.matrix() + dndq_ * q_initial).array();
  if ((get_endmember_occupancies().matrix().transpose() * n.matrix())
          .minCoeff() < -1.e-12)
    throw std::invalid_argument(
        "Initial relaxation has negative site occupancies.");
  independent_ = fractions;
  Solution::set_composition(n);
  if (state_set_ && relaxed)
    relax();
}
void RelaxedSolution::set_state(double new_pressure, double new_temperature) {
  set_state(new_pressure, new_temperature, true);
}
void RelaxedSolution::set_state(double new_pressure, double new_temperature,
                                bool relaxed) {
  Solution::set_state(new_pressure, new_temperature);
  state_set_ = true;
  if (relaxed)
    relax();
}

void RelaxedSolution::relax() {
  struct Objective {
    RelaxedSolution *self;
    Eigen::VectorXd base;
    double offset, scale;
  } data{this, dndx_ * independent_.matrix(), Solution::compute_molar_gibbs(),
         std::max(1000., 8.314462618 * get_temperature())};
  nlopt::opt opt(nlopt::LD_SLSQP,
                 static_cast<unsigned int>(utils::checked_int(dndq_.cols())));
  opt.set_min_objective(
      [](const std::vector<double> &q, std::vector<double> &gradient,
         void *context) {
        auto &d = *static_cast<Objective *>(context);
        const Eigen::Map<const Eigen::VectorXd> x(
            q.data(), static_cast<Eigen::Index>(q.size()));
        d.self->Solution::set_composition((d.base + d.self->dndq_ * x).array());
        if (!gradient.empty()) {
          Eigen::Map<Eigen::VectorXd> g(
              gradient.data(), static_cast<Eigen::Index>(gradient.size()));
          g = d.self->dndq_.transpose() * d.self->get_partial_gibbs().matrix() /
              d.scale;
        }
        return (d.self->Solution::compute_molar_gibbs() - d.offset) / d.scale;
      },
      &data);
  // Site inequalities allow signed endmember amounts, as in Solution.
  struct Inequality {
    Eigen::VectorXd a;
    double b;
  };
  std::vector<Inequality> inequalities;
  Eigen::MatrixXd sites = get_endmember_occupancies().matrix().transpose();
  for (Eigen::Index i = 0; i < sites.rows(); ++i) {
    auto a = (-sites.row(i) * dndq_).transpose().eval();
    if (a.norm() > 1.e-14)
      inequalities.push_back({a, -sites.row(i).dot(data.base)});
  }
  for (auto &inequality : inequalities)
    opt.add_inequality_constraint(
        [](const std::vector<double> &q, std::vector<double> &gradient,
           void *context) {
          const auto &c = *static_cast<const Inequality *>(context);
          if (!gradient.empty())
            std::copy(c.a.data(), c.a.data() + c.a.size(), gradient.begin());
          return c.a.dot(Eigen::Map<const Eigen::VectorXd>(
                     q.data(), static_cast<Eigen::Index>(q.size()))) +
                 c.b;
        },
        &inequality, 1.e-12);
  opt.set_xtol_abs(1.e-12);
  opt.set_ftol_abs(1.e-12);
  opt.set_maxeval(1000);
  std::vector<double> q(static_cast<std::size_t>(dndq_.cols()), 0.);
  double value = 0.;
  try {
    opt.optimize(q, value);
  } catch (const nlopt::roundoff_limited &) {
    // Validate the returned state rather than discarding a converged minimum.
  }
  auto n = (data.base +
            dndq_ * Eigen::Map<const Eigen::VectorXd>(q.data(), dndq_.cols()))
               .eval();
  if (!n.allFinite() || (sites * n).minCoeff() < -1.e-9)
    throw std::runtime_error(
        "Solution relaxation returned an infeasible state.");
  Solution::set_composition(n.array());
}

Eigen::Matrix2d RelaxedSolution::relaxed_hessian() const {
  const double v = get_molar_volume(), t = get_temperature();
  const double alpha_v = v * Solution::compute_thermal_expansivity();
  Eigen::Matrix2d h;
  h << -v / Solution::compute_isothermal_bulk_modulus_reuss(), alpha_v, alpha_v,
      -Solution::compute_molar_heat_capacity_p() / t;
  Eigen::MatrixXd coupling(dndq_.cols(), 2);
  coupling.col(0) = dndq_.transpose() * get_partial_volumes().matrix();
  coupling.col(1) = -dndq_.transpose() * get_partial_entropies().matrix();
  Eigen::MatrixXd structural = dndq_.transpose() * get_gibbs_hessian() * dndq_;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(structural, Eigen::ComputeFullU |
                                                        Eigen::ComputeFullV);
  svd.setThreshold(1.e-15); // numpy.linalg.pinv default in the reference
  return h - coupling.transpose() * svd.solve(coupling);
}
double RelaxedSolution::compute_isothermal_bulk_modulus_reuss() const {
  return -get_molar_volume() / relaxed_hessian()(0, 0);
}
double RelaxedSolution::compute_thermal_expansivity() const {
  return relaxed_hessian()(0, 1) / get_molar_volume();
}
double RelaxedSolution::compute_molar_heat_capacity_p() const {
  return -get_temperature() * relaxed_hessian()(1, 1);
}
} // namespace burnman
