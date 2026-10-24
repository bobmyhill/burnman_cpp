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

#include "burnman/optim/linear_fitting.hpp"
#include "burnman/utils/index_utils.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <nlopt.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace burnman::optim {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using SVD = Eigen::JacobiSVD<Matrix>;
constexpr double feasibility_tolerance = 1.e-10;

double rank_tolerance(const Matrix &matrix) {
  return std::numeric_limits<double>::epsilon() *
         static_cast<double>(std::max(matrix.rows(), matrix.cols()));
}

Vector least_squares(const Matrix &A, const Vector &b) {
  if (!A.cols())
    return Vector(0);
  if (!A.rows())
    return Vector::Zero(A.cols());
  SVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(rank_tolerance(A));
  return svd.solve(b);
}

LinearConstraints constraints(const std::optional<LinearConstraints> &input,
                              Eigen::Index n, bool equality) {
  if (!input || (!input->first.rows() && !input->second.size()))
    return {Matrix(0, n), Vector(0)};
  auto [matrix, values] = *input;
  if (matrix.cols() != n || matrix.rows() != values.size() ||
      !matrix.allFinite() || !values.allFinite())
    throw std::invalid_argument(
        "Constraints require finite matrices with one column per parameter "
        "and one value per row.");
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    const double scale = matrix.row(i).stableNorm();
    if (!std::isfinite(scale))
      throw std::invalid_argument(
          "Constraint scaling exceeds numerical range.");
    if (scale > 0.) {
      matrix.row(i) /= scale;
      values[i] /= scale;
    } else if (equality ? values[i] != 0. : values[i] < 0.)
      throw std::invalid_argument("Linear constraints are infeasible.");
  }
  if (!values.allFinite())
    throw std::invalid_argument("Constraint scaling exceeds numerical range.");
  return {matrix, values};
}

double constraint_scale(const Matrix &D, const Vector &d, const Vector &x,
                        Eigen::Index i) {
  // Only parameters participating in this constraint set its tolerance.
  return 1. + std::abs(d[i]) + D.row(i).cwiseAbs().dot(x.cwiseAbs());
}

bool feasible(const Matrix &D, const Vector &d, const Vector &x) {
  if (!x.allFinite())
    return false;
  for (Eigen::Index i = 0; i < D.rows(); ++i) {
    const double value = D.row(i).dot(x);
    const double tolerance =
        feasibility_tolerance * constraint_scale(D, d, x, i);
    if (!std::isfinite(value) || !std::isfinite(tolerance) ||
        value - d[i] > tolerance)
      return false;
  }
  return true;
}

struct AffineSpace {
  Vector point;
  Matrix basis;
};

AffineSpace equality_space(const Matrix &C, const Vector &c) {
  const auto n = C.cols();
  if (!C.rows())
    return {Vector::Zero(n), Matrix::Identity(n, n)};
  SVD svd(C, Eigen::ComputeThinU | Eigen::ComputeFullV);
  svd.setThreshold(rank_tolerance(C));
  const Vector point = svd.solve(c);
  if (!feasible(C, c, point) || !feasible(-C, -c, point))
    throw std::invalid_argument("Equality constraints are infeasible.");
  return {point, svd.matrixV().rightCols(n - svd.rank())};
}

// Nonnegative multipliers certify stationarity at active inequalities,
// including redundant constraints. No bounds are imposed on fitted parameters.
bool stationary(const Matrix &A, const Vector &b, const Matrix &D,
                const Vector &d, const Vector &x) {
  const Vector residual = A * x - b;
  const Vector gradient = A.transpose() * residual;
  const double tolerance =
      1.e-9 * std::max(1., A.norm() * (b.norm() + (A * x).norm()));
  if (gradient.norm() <= tolerance)
    return true;
  std::vector<Eigen::Index> active;
  for (Eigen::Index i = 0; i < D.rows(); ++i)
    if (std::abs(D.row(i).dot(x) - d[i]) <=
        1.e-8 * constraint_scale(D, d, x, i))
      active.push_back(i);
  Matrix normals(x.size(), static_cast<Eigen::Index>(active.size()));
  for (std::size_t j = 0; j < active.size(); ++j)
    normals.col(static_cast<Eigen::Index>(j)) = D.row(active[j]).transpose();
  if (active.empty())
    return false;
  const Vector target = -gradient / gradient.norm();
  Vector multipliers = Vector::Zero(normals.cols());
  std::vector<Eigen::Index> passive;
  const Eigen::Index limit = 100 * std::max<Eigen::Index>(1, normals.cols());
  Eigen::Index iteration = 0;
  while (++iteration <= limit) {
    const Vector dual = normals.transpose() * (target - normals * multipliers);
    Eigen::Index entering = -1;
    double best = 1.e-12;
    for (Eigen::Index i = 0; i < dual.size(); ++i)
      if (std::find(passive.begin(), passive.end(), i) == passive.end() &&
          dual[i] > best) {
        entering = i;
        best = dual[i];
      }
    if (entering < 0)
      return (normals * multipliers - target).norm() <= 1.e-7;
    passive.push_back(entering);
    while (++iteration <= limit) {
      Matrix subset(normals.rows(), static_cast<Eigen::Index>(passive.size()));
      for (std::size_t j = 0; j < passive.size(); ++j)
        subset.col(static_cast<Eigen::Index>(j)) = normals.col(passive[j]);
      const Vector fit = least_squares(subset, target);
      Vector candidate = Vector::Zero(normals.cols());
      bool positive = true;
      for (std::size_t j = 0; j < passive.size(); ++j) {
        candidate[passive[j]] = fit[static_cast<Eigen::Index>(j)];
        positive = positive && candidate[passive[j]] > 0.;
      }
      if (positive) {
        multipliers = candidate;
        break;
      }
      double fraction = 1.;
      for (const auto i : passive)
        if (candidate[i] <= 0.) {
          const double difference = multipliers[i] - candidate[i];
          fraction = std::min(
              fraction, difference > 0. ? multipliers[i] / difference : 0.);
        }
      multipliers += fraction * (candidate - multipliers);
      passive.erase(std::remove_if(passive.begin(), passive.end(),
                                   [&](Eigen::Index i) {
                                     if (multipliers[i] > 1.e-12)
                                       return false;
                                     multipliers[i] = 0.;
                                     return true;
                                   }),
                    passive.end());
    }
  }
  return false;
}

Vector constrained_fit(const Matrix &A, const Vector &b, const Matrix &D,
                       const Vector &d) {
  Vector x = least_squares(A, b);
  if (feasible(D, d, x))
    return x;
  if (!x.size())
    throw std::invalid_argument("Linear constraints are infeasible.");
  struct Objective {
    Matrix matrix;
    Vector values;
  } objective{A, b};
  const double scale =
      std::max(A.cwiseAbs().maxCoeff(), b.cwiseAbs().maxCoeff());
  if (scale > 0.) {
    objective.matrix /= scale;
    objective.values /= scale;
  }
  nlopt::opt opt(nlopt::LD_SLSQP,
                 static_cast<unsigned int>(utils::checked_int(x.size())));
  opt.set_min_objective(
      [](const std::vector<double> &values, std::vector<double> &gradient,
         void *context) {
        const auto &data = *static_cast<const Objective *>(context);
        const Eigen::Map<const Vector> parameters(
            values.data(), static_cast<Eigen::Index>(values.size()));
        const Vector residual = data.matrix * parameters - data.values;
        if (!gradient.empty()) {
          Eigen::Map<Vector> result(gradient.data(), parameters.size());
          result = 2. * data.matrix.transpose() * residual;
        }
        return residual.squaredNorm();
      },
      &objective);
  struct Inequality {
    Vector row;
    double value;
  };
  std::vector<Inequality> inequalities;
  inequalities.reserve(static_cast<std::size_t>(D.rows()));
  for (Eigen::Index i = 0; i < D.rows(); ++i) {
    if (D.row(i).norm() <= rank_tolerance(D)) {
      if (d[i] < -feasibility_tolerance)
        throw std::invalid_argument("Linear constraints are infeasible.");
      continue;
    }
    inequalities.push_back({D.row(i).transpose(), d[i]});
  }
  for (auto &inequality : inequalities)
    opt.add_inequality_constraint(
        [](const std::vector<double> &values, std::vector<double> &gradient,
           void *context) {
          const auto &data = *static_cast<const Inequality *>(context);
          if (!gradient.empty())
            std::copy(data.row.data(), data.row.data() + data.row.size(),
                      gradient.begin());
          return data.row.dot(Eigen::Map<const Vector>(
                     values.data(), static_cast<Eigen::Index>(values.size()))) -
                 data.value;
        },
        &inequality, 1.e-12);
  opt.set_xtol_rel(1.e-13);
  opt.set_ftol_rel(1.e-14);
  opt.set_maxeval(2000);
  std::vector<double> parameters(x.data(), x.data() + x.size());
  double value = 0.;
  try {
    opt.optimize(parameters, value);
  } catch (const nlopt::roundoff_limited &) {
    // Accept a numerical stopping limit only after checking the optimum.
  }
  x = Eigen::Map<const Vector>(parameters.data(), x.size());
  if (!feasible(D, d, x))
    throw std::invalid_argument("Linear constraints are infeasible or could "
                                "not be satisfied to numerical precision.");
  if (!stationary(objective.matrix, objective.values, D, d, x))
    throw std::runtime_error("Constrained least squares did not converge.");
  return x;
}
} // namespace

LeastSquaresResult weighted_constrained_least_squares(
    const Matrix &A, const Vector &b, const std::optional<Matrix> &Cov_b,
    const std::optional<LinearConstraints> &equality_constraints,
    const std::optional<LinearConstraints> &inequality_constraints,
    bool allow_rank_deficient) {
  if (!A.rows() || !A.cols() || A.rows() != b.size() || !A.allFinite() ||
      !b.allFinite())
    throw std::invalid_argument("A must be a finite nonempty matrix with one "
                                "row per observation in b.");
  Matrix MA = A;
  Vector Mb = b;
  if (Cov_b) {
    if (Cov_b->rows() != b.size() || Cov_b->cols() != b.size() ||
        !Cov_b->allFinite())
      throw std::invalid_argument(
          "Cov_b must be finite, symmetric and match the observations.");
    const double scale = Cov_b->cwiseAbs().maxCoeff();
    Matrix covariance = scale > 0. ? (*Cov_b / scale).eval() : *Cov_b;
    if (!covariance.isApprox(covariance.transpose(), 1.e-12))
      throw std::invalid_argument("Cov_b must be symmetric.");
    covariance = (0.5 * covariance + 0.5 * covariance.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(covariance);
    if (eigen.info() != Eigen::Success ||
        eigen.eigenvalues().minCoeff() < -rank_tolerance(covariance))
      throw std::invalid_argument("Cov_b must be positive semidefinite.");
    Vector weights = Vector::Zero(b.size());
    const double largest = std::max(0., eigen.eigenvalues().maxCoeff());
    for (Eigen::Index i = 0; i < b.size(); ++i) {
      const double variance = eigen.eigenvalues()[i];
      if (variance > 1.e-15 * largest)
        weights[i] = (1. / std::sqrt(variance)) / std::sqrt(scale);
    }
    MA = weights.asDiagonal() * eigen.eigenvectors().transpose() * A;
    Mb = weights.asDiagonal() * eigen.eigenvectors().transpose() * b;
  }
  if (!MA.allFinite() || !Mb.allFinite())
    throw std::invalid_argument(
        "Weighted observations exceed numerical range.");
  SVD design(MA, Eigen::ComputeThinU | Eigen::ComputeThinV);
  design.setThreshold(rank_tolerance(MA));
  if (!allow_rank_deficient && design.rank() < A.cols())
    throw std::invalid_argument(
        "The weighted design matrix is rank-deficient. Supply independent "
        "observations or set allow_rank_deficient=True.");

  const auto [C, c] = constraints(equality_constraints, A.cols(), true);
  const auto [D, d] = constraints(inequality_constraints, A.cols(), false);
  const auto [point, basis] = equality_space(C, c);
  const Vector parameters =
      point + basis * constrained_fit(MA * basis, Mb - MA * point, D * basis,
                                      d - D * point);
  if (!parameters.allFinite())
    throw std::runtime_error(
        "Least-squares parameters exceed numerical range.");
  if (!feasible(C, c, parameters) || !feasible(-C, -c, parameters) ||
      !feasible(D, d, parameters))
    throw std::runtime_error(
        "The fitted parameters do not satisfy the constraints.");

  // The parameter covariance is the pseudoinverse of the unconstrained
  // information matrix. Singular values avoid forming its normal equations.
  Vector variances = Vector::Zero(design.singularValues().size());
  const double largest = design.singularValues()[0];
  for (Eigen::Index i = 0; i < variances.size(); ++i)
    if (design.singularValues()[i] > std::sqrt(1.e-15) * largest) {
      const double inverse = 1. / design.singularValues()[i];
      variances[i] = inverse * inverse;
    }
  const Matrix covariance =
      design.matrixV() * variances.asDiagonal() * design.matrixV().transpose();
  const double residual = (MA * parameters - Mb).squaredNorm();
  if (!covariance.allFinite() || !std::isfinite(residual))
    throw std::runtime_error("Least-squares result exceeds numerical range.");
  return {parameters, covariance, residual};
}
} // namespace burnman::optim
