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

#include "burnman/tools/pseudosection/critical.hpp"
#include "burnman/core/solution.hpp"
#include "burnman/utils/constants.hpp"
#include <algorithm>
#include <cmath>

namespace burnman::pseudosections::detail {
Eigen::VectorXd CriticalConstraint::fractions(const Eigen::VectorXd &x) const {
  Eigen::VectorXd p(model->get_n_endmembers());
  p.tail(p.size() - 1) = x.segment(start, p.size() - 1);
  p[0] = 1. - p.tail(p.size() - 1).sum();
  return p;
}

// The softest allowed composition mode is the lowest Gibbs-Hessian eigenvector.
// At an ordinary critical point its curvature vanishes; reference keeps the
// otherwise arbitrary eigenvector sign consistent during numerical derivatives.
std::pair<double, Eigen::VectorXd>
CriticalConstraint::curvature(double P, double T,
                              const Eigen::VectorXd &p) const {
  Eigen::MatrixXd h =
      basis.transpose() * model->compute_gibbs_hessian(P, T, p.array()) * basis;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig((h + h.transpose()) * .5);
  if (eig.info() != Eigen::Success)
    throw std::runtime_error("Critical curvature eigensolve failed.");
  Eigen::VectorXd mode = basis * eig.eigenvectors().col(0);
  if (mode.dot(reference) < 0.)
    mode = -mode;
  return {eig.eigenvalues()[0], mode};
}

std::pair<Eigen::VectorXd, double>
CriticalConstraint::higher_derivatives(double P, double T,
                                       const Eigen::VectorXd &p,
                                       const Eigen::VectorXd &mode) const {
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(p.size());
  double fourth = 0.;
  for (auto entry :
       {std::pair{model->get_endmember_n_occupancies().matrix().eval(), -1.},
        std::pair{model->get_site_multiplicities().matrix().eval(), 1.}}) {
    auto sites = (entry.first.transpose() * p).eval(),
         direction = (entry.first.transpose() * mode).eval();
    for (int i = 0; i < sites.size(); ++i) {
      double site = std::max(constants::precision::logish_eps, sites[i]);
      gradient +=
          entry.second * entry.first.col(i) * std::pow(direction[i] / site, 2);
      fourth -=
          2. * entry.second * std::pow(direction[i], 4) / std::pow(site, 3);
    }
  }
  gradient *= constants::physics::gas_constant * T;
  fourth *= constants::physics::gas_constant * T;
  if (auto regular =
          dynamic_cast<const solution_models::AsymmetricRegularSolution *>(
              model.get())) {
    // G_nonideal = A(p)/S(p), where A is quadratic and S linear. This
    // analytic derivative also covers symmetric regular solutions (S'=0).
    Eigen::MatrixXd W = regular->get_energy_interactions() -
                        T * regular->get_entropy_interactions() +
                        P * regular->get_volume_interactions();
    Eigen::VectorXd alpha = regular->get_alphas().matrix();
    Eigen::MatrixXd quadratic =
        alpha.asDiagonal() * (W + W.transpose()) * alpha.asDiagonal();
    Eigen::VectorXd dA = quadratic * p, mixed = quadratic * mode;
    double S = alpha.dot(p), dS = alpha.dot(mode), A = p.dot(dA) * .5,
           along = mode.dot(dA), ddA = mode.dot(mixed);
    gradient += -(ddA * alpha + 2. * mixed * dS) / (S * S) +
                (4. * along * dS * alpha + 2. * dA * dS * dS) / std::pow(S, 3) -
                6. * A * dS * dS * alpha / std::pow(S, 4);
    fourth += 12. * ddA * dS * dS / std::pow(S, 3) -
              24. * along * std::pow(dS, 3) / std::pow(S, 4) +
              24. * A * std::pow(dS, 4) / std::pow(S, 5);
  }
  return {gradient, fourth};
}

CriticalConstraint::CriticalConstraint(const Assemblage &a, int phase,
                                       const EquilibrationParameters &prm,
                                       const Eigen::VectorXd &direction,
                                       bool third_order)
    : model(a.get_phase<Solution>(static_cast<std::size_t>(phase))
                ->get_solution_model()),
      reference(direction), start(prm.phase_amount_indices[phase] + 1),
      third(third_order) {
  Eigen::VectorXd p = a.get_phase<Solution>(static_cast<std::size_t>(phase))
                          ->get_molar_fractions()
                          .matrix();
  Eigen::MatrixXd occupancies = model->get_endmember_occupancies().matrix();
  // Composition changes must preserve sum(p)=1 and keep empty sites empty
  // along the merging direction. Ordering models may have negative endmember
  // coefficients, so site contents supply the physical bounds.
  std::vector<Eigen::VectorXd> rows{Eigen::VectorXd::Ones(p.size())};
  for (int i = 0; i < occupancies.cols(); ++i)
    if (std::abs(p.dot(occupancies.col(i))) < 1.e-9 &&
        std::abs(reference.dot(occupancies.col(i))) < 1.e-9)
      rows.push_back(occupancies.col(i));
  Eigen::MatrixXd faces(rows.size(), p.size());
  for (std::size_t i = 0; i < rows.size(); ++i)
    faces.row(static_cast<Eigen::Index>(i)) = rows[i].transpose();
  Eigen::FullPivLU<Eigen::MatrixXd> lu(faces);
  lu.setThreshold(1.e-10);
  Eigen::MatrixXd kernel = lu.kernel();
  if (!kernel.cols() || kernel.isZero())
    throw std::runtime_error("No feasible critical composition direction.");
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(kernel);
  basis =
      qr.householderQ() * Eigen::MatrixXd::Identity(p.size(), kernel.cols());
}

std::unique_ptr<EqualityConstraint> CriticalConstraint::clone() const {
  return std::make_unique<CriticalConstraint>(*this);
}

double CriticalConstraint::evaluate(const Eigen::VectorXd &x,
                                    const Assemblage &) const {
  auto p = fractions(x);
  auto [eigenvalue, mode] = curvature(x[0], x[1], p);
  return (third ? mode.dot(higher_derivatives(x[0], x[1], p, mode).first)
                : eigenvalue) /
         1.e4;
}

// Use small changes to calculate derivatives of the critical-point
// conditions. The merging direction also changes with P,T and composition;
// keep its sign consistent with reference so it does not reverse between
// evaluations.
Eigen::VectorXd CriticalConstraint::derivative(const Eigen::VectorXd &x,
                                               const Assemblage &a,
                                               Eigen::Index size) const {
  Eigen::VectorXd result = Eigen::VectorXd::Zero(size);
  std::vector<int> indices{0, 1};
  for (int i = 0; i < model->get_n_endmembers() - 1; ++i)
    indices.push_back(start + i);
  for (int i : indices) {
    double step =
        i == 0   ? std::max(100., std::abs(x[0]) * 1.e-6)
        : i == 1 ? 1.e-3
                 : std::max(1.e-8, std::min(1.e-5, std::abs(x[i]) * 1.e-3));
    auto left = x.eval(), right = x.eval();
    left[i] -= step;
    right[i] += step;
    result[i] = (evaluate(right, a) - evaluate(left, a)) / (2. * step);
  }
  return result;
}

Eigen::VectorXd CriticalConstraint::mode(const Assemblage &a, int phase) const {
  return curvature(a.get_pressure(), a.get_temperature(),
                   a.get_phase<Solution>(static_cast<std::size_t>(phase))
                       ->get_molar_fractions()
                       .matrix())
      .second;
}

bool CriticalConstraint::stable(const Assemblage &a, int phase) const {
  if (typeid(*model) != typeid(solution_models::IdealSolution) &&
      !dynamic_cast<const solution_models::AsymmetricRegularSolution *>(
          model.get()))
    return false; // Higher derivatives of other nonideal models are unknown.
  auto p = a.get_phase<Solution>(static_cast<std::size_t>(phase))
               ->get_molar_fractions()
               .matrix()
               .eval();
  Eigen::MatrixXd h = basis.transpose() *
                      model->compute_gibbs_hessian(
                          a.get_pressure(), a.get_temperature(), p.array()) *
                      basis;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig((h + h.transpose()) * .5);
  if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite())
    return false;
  const double roundoff = 64. * std::numeric_limits<double>::epsilon();
  const auto hard = eig.eigenvalues().tail(h.rows() - 1).eval();
  if ((hard.array() <= roundoff * h.norm()).any())
    return false;
  auto [gradient, fourth] =
      higher_derivatives(a.get_pressure(), a.get_temperature(), p,
                         basis * eig.eigenvectors().col(0));
  Eigen::VectorXd coupling =
      eig.eigenvectors().rightCols(h.rows() - 1).transpose() *
      basis.transpose() * gradient;
  double relaxation = 3. * (coupling.array().square() / hard.array()).sum();
  return std::isfinite(fourth) && std::isfinite(relaxation) &&
         fourth - relaxation > roundoff * (std::abs(fourth) + relaxation);
}
} // namespace burnman::pseudosections::detail
