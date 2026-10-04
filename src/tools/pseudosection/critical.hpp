/* GPL v3 or later. Intrinsic critical-point constraints for native equilibrate.
 */
#pragma once
#include "burnman/utils/constants.hpp"
#include "internal.hpp"
#include <algorithm>
#include <cmath>

namespace burnman::pseudosections::detail {
// Replacing the coalescing copies by one solution removes the trivial
// identical-copy root. Its restricted Gibbs curvature and third directional
// derivative vanish at an ordinary critical point, supplying equilibrate's
// two P,T constraints without differencing nearly identical chemical
// potentials.
class CriticalConstraint : public EqualityConstraint {
  std::shared_ptr<const solution_models::SolutionModel> model;
  Eigen::MatrixXd basis;
  Eigen::VectorXd reference;
  int start;
  bool third;
  Eigen::VectorXd fractions(const Eigen::VectorXd &x) const {
    Eigen::VectorXd p(model->get_n_endmembers());
    p.tail(p.size() - 1) = x.segment(start, p.size() - 1);
    p[0] = 1. - p.tail(p.size() - 1).sum();
    return p;
  }
  std::pair<double, Eigen::VectorXd> curvature(double P, double T,
                                               const Eigen::VectorXd &p) const {
    Eigen::MatrixXd h = basis.transpose() *
                        model->compute_gibbs_hessian(P, T, p.array()) * basis;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig((h + h.transpose()) *
                                                       .5);
    if (eig.info() != Eigen::Success)
      throw std::runtime_error("Critical curvature eigensolve failed.");
    Eigen::VectorXd mode = basis * eig.eigenvectors().col(0);
    if (mode.dot(reference) < 0.)
      mode = -mode;
    return {eig.eigenvalues()[0], mode};
  }
  double third_derivative(double P, double T, const Eigen::VectorXd &p,
                          const Eigen::VectorXd &mode) const {
    double value = 0.;
    for (auto entry :
         {std::pair{model->get_endmember_n_occupancies().matrix().eval(), -1.},
          std::pair{model->get_site_multiplicities().matrix().eval(), 1.}}) {
      auto sites = (entry.first.transpose() * p).eval(),
           direction = (entry.first.transpose() * mode).eval();
      for (int i = 0; i < sites.size(); ++i) {
        double site = std::max(constants::precision::logish_eps, sites[i]);
        value += entry.second * direction[i] * direction[i] * direction[i] /
                 (site * site);
      }
    }
    value *= constants::physics::gas_constant * T;
    if (auto regular =
            dynamic_cast<const solution_models::AsymmetricRegularSolution *>(
                model.get())) {
      // G_nonideal = A(p)/S(p), where A is quadratic and S linear. This
      // analytic derivative also covers symmetric regular solutions (S'=0).
      Eigen::MatrixXd W = regular->get_energy_interactions() -
                          T * regular->get_entropy_interactions() +
                          P * regular->get_volume_interactions();
      Eigen::VectorXd y = (regular->get_alphas() * p.array()).matrix(),
                      v = (regular->get_alphas() * mode.array()).matrix();
      double S = y.sum(), dS = v.sum(), A = y.dot(W * y),
             dA = v.dot((W + W.transpose()) * y), ddA = 2. * v.dot(W * v);
      value += -6. * A * std::pow(dS, 3) / std::pow(S, 4) +
               6. * dA * dS * dS / std::pow(S, 3) - 3. * ddA * dS / (S * S);
    }
    return value;
  }

public:
  CriticalConstraint(const Assemblage &a, int phase,
                     const EquilibrationParameters &prm,
                     const Eigen::VectorXd &direction, bool third_order)
      : model(a.get_phase<Solution>(phase)->get_solution_model()),
        reference(direction), start(prm.phase_amount_indices[phase] + 1),
        third(third_order) {
    Eigen::VectorXd p =
        a.get_phase<Solution>(phase)->get_molar_fractions().matrix();
    Eigen::MatrixXd occupancies = model->get_endmember_occupancies().matrix();
    std::vector<Eigen::VectorXd> rows{Eigen::VectorXd::Ones(p.size())};
    for (int i = 0; i < occupancies.cols(); ++i)
      if (std::abs(p.dot(occupancies.col(i))) < 1.e-9 &&
          std::abs(reference.dot(occupancies.col(i))) < 1.e-9)
        rows.push_back(occupancies.col(i));
    Eigen::MatrixXd faces(rows.size(), p.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
      faces.row(i) = rows[i].transpose();
    Eigen::FullPivLU<Eigen::MatrixXd> lu(faces);
    lu.setThreshold(1.e-10);
    Eigen::MatrixXd kernel = lu.kernel();
    if (!kernel.cols() || kernel.isZero())
      throw std::runtime_error("No feasible critical composition direction.");
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(kernel);
    basis =
        qr.householderQ() * Eigen::MatrixXd::Identity(p.size(), kernel.cols());
  }
  std::unique_ptr<EqualityConstraint> clone() const override {
    return std::make_unique<CriticalConstraint>(*this);
  }
  double evaluate(const Eigen::VectorXd &x, const Assemblage &) const override {
    auto p = fractions(x);
    auto [eigenvalue, mode] = curvature(x[0], x[1], p);
    return (third ? third_derivative(x[0], x[1], p, mode) : eigenvalue) / 1.e4;
  }
  Eigen::VectorXd derivative(const Eigen::VectorXd &x, const Assemblage &a,
                             Eigen::Index size) const override {
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
  Eigen::VectorXd mode(const Assemblage &a, int phase) const {
    return curvature(
               a.get_pressure(), a.get_temperature(),
               a.get_phase<Solution>(phase)->get_molar_fractions().matrix())
        .second;
  }
};
} // namespace burnman::pseudosections::detail
