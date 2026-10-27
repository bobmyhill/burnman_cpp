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

#include "burnman/tools/thermobarometry.hpp"
#include "burnman/core/combined_mineral.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/index_utils.hpp"
#include "burnman/utils/matrix_utils.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <nlopt.hpp>
#include <stdexcept>
#include <vector>

namespace burnman::thermobarometry {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Matrix covariance(const Matrix &input, Eigen::Index size, const char *name) {
  if (input.rows() != size || input.cols() != size || !input.allFinite() ||
      !input.isApprox(input.transpose(), 1.e-10))
    throw std::invalid_argument(
        std::string(name) +
        " must be finite, square and symmetric with matching dimensions.");
  Matrix result = 0.5 * (input + input.transpose());
  if (size) {
    Eigen::SelfAdjointEigenSolver<Matrix> eigen(result);
    const double scale = std::max(1., result.cwiseAbs().maxCoeff());
    if (eigen.info() != Eigen::Success ||
        eigen.eigenvalues().minCoeff() < -1.e-10 * scale)
      throw std::invalid_argument(std::string(name) +
                                  " must be positive semidefinite.");
  }
  return result;
}

Matrix inverse_covariance(const Matrix &input,
                          bool inverse_square_root = false) {
  const Matrix symmetric = covariance(input, input.rows(), "Covariance");
  if (!input.rows())
    return input;
  Eigen::SelfAdjointEigenSolver<Matrix> eigen(symmetric);
  Vector inverse = Vector::Zero(input.rows());
  const double largest = eigen.eigenvalues().cwiseAbs().maxCoeff();
  for (Eigen::Index i = 0; i < inverse.size(); ++i) {
    const double value = eigen.eigenvalues()[i];
    // pinv(sqrt(C)) and pinv(C) use their own relative singular-value cutoff.
    const double cutoff = inverse_square_root ? 1.e-30 : 1.e-15;
    if (value > cutoff * largest)
      inverse[i] = inverse_square_root ? 1. / std::sqrt(value) : 1. / value;
  }
  return eigen.eigenvectors() * inverse.asDiagonal() *
         eigen.eigenvectors().transpose();
}

void require_state(const Assemblage &assemblage) {
  if (!assemblage.has_state())
    throw std::invalid_argument(
        "Set the assemblage pressure and temperature first.");
}

void require_reactions(const Assemblage &assemblage, const Matrix &R) {
  require_state(assemblage);
  if (!R.rows() || R.cols() != assemblage.get_n_endmembers() || !R.allFinite())
    throw std::invalid_argument(
        "Reaction matrix must have finite rows and one column per endmember.");
}

Matrix endmember_covariance(const Assemblage &assemblage,
                            const minerals::CovarianceData &data) {
  const auto n = static_cast<Eigen::Index>(data.endmember_names.size());
  const Matrix C = covariance(data.covariance_matrix, n, "Dataset covariance");
  std::map<std::string, Eigen::Index> indices;
  for (Eigen::Index i = 0; i < n; ++i)
    if (!indices.emplace(data.endmember_names[static_cast<std::size_t>(i)], i)
             .second)
      throw std::invalid_argument(
          "Dataset covariance endmember names must be unique.");
  Matrix A = Matrix::Zero(assemblage.get_n_endmembers(), n);
  Eigen::Index row = 0;
  const auto add = [&](const std::map<std::string, double> &amounts) {
    for (const auto &[name, amount] : amounts) {
      const auto found = indices.find(name);
      if (found == indices.end())
        throw std::invalid_argument(
            "Endmember missing from dataset covariance: " + name);
      A(row, found->second) = amount;
    }
    ++row;
  };
  for (const auto &phase : assemblage.get_phases()) {
    if (const auto *solution = dynamic_cast<const Solution *>(phase.get())) {
      for (const auto &endmember : solution->get_solution_model()->endmembers)
        add(raw_endmember_amounts(endmember));
    } else {
      add({{phase->get_name(), 1.}});
    }
  }
  return A * C * A.transpose();
}

Matrix solution_covariance(const Assemblage &assemblage) {
  Matrix result = Matrix::Zero(assemblage.get_n_endmembers(),
                               assemblage.get_n_endmembers());
  Eigen::Index offset = 0;
  for (const auto &phase : assemblage.get_phases()) {
    if (const auto *solution = dynamic_cast<const Solution *>(phase.get())) {
      if (!solution->compositional_covariances)
        throw std::invalid_argument(
            "Each solution requires compositional_covariances.");
      const auto n = solution->get_n_endmembers();
      const Matrix C = covariance(*solution->compositional_covariances, n,
                                  "Compositional covariance");
      const Matrix H = solution->get_gibbs_hessian();
      result.block(offset, offset, n, n) = H * C * H.transpose();
      offset += n;
    } else {
      result(offset, offset) = 1.e-12;
      ++offset;
    }
  }
  return result;
}

Matrix affinity_covariance(const Assemblage &assemblage, const Matrix &R,
                           const std::optional<Matrix> &dataset,
                           bool include_state) {
  require_reactions(assemblage, R);
  Matrix C = solution_covariance(assemblage);
  if (dataset)
    C += *dataset;
  if (include_state) {
    if (!assemblage.state_covariances)
      throw std::invalid_argument(
          "State uncertainties require state_covariances.");
    Matrix jacobian(C.rows(), 2);
    jacobian.col(0) = assemblage.get_endmember_partial_volumes().matrix();
    jacobian.col(1) = -assemblage.get_endmember_partial_entropies().matrix();
    C += jacobian *
         covariance(*assemblage.state_covariances, 2, "State covariance") *
         jacobian.transpose();
  }
  return R * C * R.transpose();
}

Eigen::Index free_vectors(const Solution &solution) {
  if (!solution.free_compositional_vectors)
    return 0;
  const Matrix &vectors = *solution.free_compositional_vectors;
  if (vectors.cols() != solution.get_n_endmembers() || !vectors.allFinite())
    throw std::invalid_argument("Free compositional vectors must have one "
                                "finite column per endmember.");
  return vectors.rows();
}

std::vector<Eigen::Index> greedy_selection(const Matrix &occupancies,
                                           Vector residual, double tolerance) {
  std::vector<Eigen::Index> indices;
  for (Eigen::Index step = 0; step < occupancies.rows(); ++step) {
    double largest = 0.;
    Eigen::Index selected = -1;
    for (Eigen::Index i = 0; i < occupancies.rows(); ++i) {
      double amount = std::numeric_limits<double>::infinity();
      for (Eigen::Index j = 0; j < occupancies.cols(); ++j)
        if (occupancies(i, j) > 0.)
          amount = std::min(amount, residual[j] / occupancies(i, j));
      if (std::isfinite(amount) && amount > largest) {
        largest = amount;
        selected = i;
      }
    }
    if (selected < 0 || largest <= tolerance)
      break;
    indices.push_back(selected);
    residual = (residual - largest * occupancies.row(selected).transpose())
                   .cwiseMax(0.);
    if (residual.norm() <= 1.e-12)
      break;
  }
  return indices;
}

struct FitContext {
  Assemblage &assemblage;
  Matrix reactions;
  std::optional<Matrix> dataset;
  bool include_prior;
  Vector scales;
  Matrix occupancy_matrix;
  Vector occupancy_baseline;
  Vector fixed;
  std::vector<Eigen::Index> active;
  int evaluations = 0;

  Vector params(const std::vector<double> &values) const {
    Vector result = fixed;
    for (std::size_t i = 0; i < active.size(); ++i)
      result[active[i]] = values[i];
    return result.cwiseProduct(scales);
  }

  double value(const std::vector<double> &values) {
    ++evaluations;
    assemblage_set_state_from_params(assemblage, params(values));
    const Matrix C = affinity_covariance(assemblage, reactions, dataset, false);
    const Vector affinity = reactions * assemblage.get_partial_gibbs().matrix();
    const Eigen::LLT<Matrix> factor(C);
    if (factor.info() != Eigen::Success)
      return std::numeric_limits<double>::infinity();
    return factor.matrixL().solve(affinity).squaredNorm() +
           (include_prior ? assemblage_state_misfit(assemblage) : 0.);
  }

  double feasible_value(const std::vector<double> &values) {
    try {
      const double result = value(values);
      return std::isfinite(result) ? result : 1.e30;
    } catch (const std::invalid_argument &) {
      return 1.e30;
    }
  }
};

void occupancy_constraints(unsigned int m, double *result, unsigned int n,
                           const double *values, double *gradient,
                           void *context) {
  const auto &fit = *static_cast<FitContext *>(context);
  const Vector params = fit.params(std::vector<double>(values, values + n));
  const Vector occupancy =
      fit.occupancy_baseline + fit.occupancy_matrix * params;
  for (unsigned int i = 0; i < m; ++i) {
    result[i] = -occupancy[i];
    if (gradient)
      for (unsigned int j = 0; j < n; ++j)
        gradient[i * n + j] =
            -fit.occupancy_matrix(i, fit.active[j]) * fit.scales[fit.active[j]];
  }
}

double objective(const std::vector<double> &values,
                 std::vector<double> &gradient, void *context) {
  auto &fit = *static_cast<FitContext *>(context);
  const double value = fit.feasible_value(values);
  for (std::size_t i = 0; i < gradient.size(); ++i) {
    const double step = 1.e-5 * std::max(1., std::abs(values[i]));
    auto perturbed = values;
    perturbed[i] += step;
    const double plus = fit.feasible_value(perturbed);
    perturbed[i] -= 2. * step;
    const double minus = fit.feasible_value(perturbed);
    if (plus < 1.e30 && minus < 1.e30)
      gradient[i] = (plus - minus) / (2. * step);
    else if (plus < 1.e30)
      gradient[i] = (plus - value) / step;
    else if (minus < 1.e30)
      gradient[i] = (value - minus) / step;
    else
      gradient[i] = 0.;
  }
  return value;
}
} // namespace

Matrix get_reaction_matrix(const Assemblage &assemblage, double tolerance) {
  if (!std::isfinite(tolerance) || tolerance < 0.)
    throw std::invalid_argument(
        "small_fraction_tol must be finite and nonnegative.");
  Matrix transform = Matrix::Identity(assemblage.get_n_endmembers(),
                                      assemblage.get_n_endmembers());
  if (tolerance > 0.) {
    std::vector<Matrix> blocks;
    Eigen::Index rows = 0;
    for (const auto &phase : assemblage.get_phases()) {
      Matrix block = Matrix::Identity(1, 1);
      if (const auto *solution = dynamic_cast<const Solution *>(phase.get())) {
        const Matrix occupancies =
            solution->get_endmember_occupancies().matrix();
        const auto poly =
            polytope::solution_polytope_from_endmember_occupancies(occupancies);
        const auto selected = greedy_selection(
            poly.get_endmember_occupancies(),
            solution->get_site_occupancies().matrix(), tolerance);
        block.resize(static_cast<Eigen::Index>(selected.size()),
                     solution->get_n_endmembers());
        for (std::size_t i = 0; i < selected.size(); ++i)
          block.row(static_cast<Eigen::Index>(i)) =
              poly.get_endmembers_as_independent_endmember_amounts().row(
                  selected[i]);
      }
      rows += block.rows();
      blocks.push_back(block);
    }
    transform = Matrix::Zero(rows, assemblage.get_n_endmembers());
    Eigen::Index row = 0, column = 0;
    for (const auto &block : blocks) {
      transform.block(row, column, block.rows(), block.cols()) = block;
      row += block.rows();
      column += block.cols();
    }
  }
  const Matrix R = utils::nullspace(
      (transform * assemblage.get_stoichiometric_matrix()).transpose());
  if (!R.rows())
    throw std::invalid_argument(
        "No independent reactions found for the given assemblage.");
  return R * transform;
}

void assemblage_set_state_from_params(Assemblage &assemblage,
                                      const Vector &params) {
  if (params.size() < 2 || !params.allFinite() || params[1] < 0.)
    throw std::invalid_argument(
        "Parameters must contain finite pressure, nonnegative temperature and "
        "compositional coordinates.");
  Eigen::Index count = 2;
  for (const auto &phase : assemblage.get_phases())
    if (const auto *solution = dynamic_cast<const Solution *>(phase.get()))
      count += free_vectors(*solution);
  if (params.size() != count)
    throw std::invalid_argument("Parameter count must equal two plus the "
                                "number of free compositional vectors.");
  std::vector<std::pair<Solution *, Eigen::ArrayXd>> compositions;
  Eigen::Index offset = 2;
  for (const auto &phase : assemblage.get_phases()) {
    if (auto *solution = dynamic_cast<Solution *>(phase.get())) {
      const auto n = free_vectors(*solution);
      if (n) {
        if (!solution->baseline_composition ||
            solution->baseline_composition->size() !=
                solution->get_n_endmembers())
          throw std::invalid_argument(
              "Free compositional coordinates require baseline_composition.");
        const Eigen::ArrayXd composition =
            *solution->baseline_composition +
            (solution->free_compositional_vectors->transpose() *
             params.segment(offset, n))
                .array();
        if (!composition.isFinite().all() ||
            std::abs(composition.sum() - 1.) > 1.e-10 ||
            (solution->get_endmember_occupancies().matrix().transpose() *
             composition.matrix())
                    .minCoeff() < -1.e-10)
          throw std::invalid_argument(
              "Compositional coordinates must give unit-sum fractions and "
              "nonnegative site occupancies.");
        compositions.emplace_back(solution, composition);
        offset += n;
      }
    }
  }
  for (const auto &[solution, composition] : compositions)
    solution->set_composition(composition);
  assemblage.set_state(params[0], params[1]);
}

Matrix assemblage_affinity_covariance_matrix(
    const Assemblage &assemblage, const Matrix &R,
    const std::optional<minerals::CovarianceData> &dataset,
    bool include_state) {
  const std::optional<Matrix> transformed =
      dataset
          ? std::optional<Matrix>(endmember_covariance(assemblage, *dataset))
          : std::nullopt;
  return affinity_covariance(assemblage, R, transformed, include_state);
}

double assemblage_affinity_misfit(
    const Assemblage &assemblage, const Matrix &R,
    const std::optional<minerals::CovarianceData> &dataset,
    bool include_state) {
  const Matrix C = assemblage_affinity_covariance_matrix(assemblage, R, dataset,
                                                         include_state);
  const Vector affinity = R * assemblage.get_partial_gibbs().matrix();
  const Eigen::LLT<Matrix> factor(C);
  if (factor.info() != Eigen::Success)
    return std::numeric_limits<double>::infinity();
  return factor.matrixL().solve(affinity).squaredNorm();
}

double assemblage_state_misfit(const Assemblage &assemblage) {
  require_state(assemblage);
  if (!assemblage.state_priors || !assemblage.state_inverse_covariances ||
      !assemblage.state_priors->allFinite())
    throw std::invalid_argument("State misfit requires finite state_priors and "
                                "state_inverse_covariances.");
  const Eigen::Vector2d delta =
      Eigen::Vector2d(assemblage.get_pressure(), assemblage.get_temperature()) -
      *assemblage.state_priors;
  return delta.dot(covariance(*assemblage.state_inverse_covariances, 2,
                              "Inverse state covariance") *
                   delta);
}

EstimateConditionsResult estimate_conditions(
    Assemblage &assemblage,
    const std::optional<minerals::CovarianceData> &dataset, bool include_prior,
    const std::optional<Eigen::Vector2d> &guess,
    const Eigen::Vector2d &pressure_bounds,
    const Eigen::Vector2d &temperature_bounds, double tolerance, int max_it) {
  if (!pressure_bounds.allFinite() || !temperature_bounds.allFinite() ||
      pressure_bounds[0] > pressure_bounds[1] ||
      temperature_bounds[0] > temperature_bounds[1] ||
      temperature_bounds[0] < 0. || max_it <= 0)
    throw std::invalid_argument(
        "Bounds must be ordered and finite, temperatures nonnegative, "
        "and max_it positive.");
  if (!guess && !assemblage.has_state())
    throw std::invalid_argument(
        "Supply guessed_conditions or set the assemblage state.");
  const Eigen::Vector2d initial =
      guess ? *guess
            : Eigen::Vector2d(assemblage.get_pressure(),
                              assemblage.get_temperature());
  if (!initial.allFinite())
    throw std::invalid_argument("Guessed conditions must be finite.");
  Eigen::Index n_free = 0;
  for (const auto &phase : assemblage.get_phases())
    if (auto *solution = dynamic_cast<Solution *>(phase.get())) {
      const auto n = free_vectors(*solution);
      if (solution->free_compositional_vectors)
        solution->baseline_composition = solution->get_molar_fractions();
      n_free += n;
    }
  const auto close = [](const Eigen::Vector2d &bounds) {
    return bounds[0] == bounds[1];
  };
  if (close(pressure_bounds) && close(temperature_bounds) && !n_free)
    throw std::invalid_argument("Both pressure and temperature cannot be fixed "
                                "without free compositional vectors.");
  const Matrix R = get_reaction_matrix(assemblage, tolerance);
  assemblage.reaction_matrix_for_optimization = R;
  const int n_params = 2 + utils::checked_int(n_free) -
                       static_cast<int>(close(pressure_bounds)) -
                       static_cast<int>(close(temperature_bounds));
  const int n_constraints = include_prior ? 2 : 0;
  if (R.rows() + n_constraints < n_params)
    throw std::invalid_argument("Not enough independent reactions and state "
                                "priors to constrain the inversion.");
  // Follow Python's initial weighted-sensitivity preconditioning.
  assemblage.set_state(
      std::clamp(initial[0], pressure_bounds[0], pressure_bounds[1]),
      std::clamp(initial[1], temperature_bounds[0], temperature_bounds[1]));
  const std::optional<Matrix> transformed =
      dataset
          ? std::optional<Matrix>(endmember_covariance(assemblage, *dataset))
          : std::nullopt;
  const Matrix initial_covariance =
      affinity_covariance(assemblage, R, transformed, false);
  const Eigen::LLT<Matrix> initial_factor(initial_covariance);
  if (initial_factor.info() != Eigen::Success)
    throw std::invalid_argument(
        "Affinity covariance must be positive definite.");
  Matrix initial_derivatives(assemblage.get_n_endmembers(), 2);
  initial_derivatives.col(0) =
      assemblage.get_endmember_partial_volumes().matrix();
  initial_derivatives.col(1) =
      -assemblage.get_endmember_partial_entropies().matrix();
  const Matrix weighted =
      initial_factor.matrixL().solve((R * initial_derivatives).eval());
  Eigen::Vector2d curvature = weighted.colwise().squaredNorm().transpose();
  if (include_prior) {
    assemblage_state_misfit(assemblage); // Validate the state prior.
    curvature += assemblage.state_inverse_covariances->diagonal();
  }
  Vector scales = Vector::Ones(2 + n_free);
  for (Eigen::Index i = 0; i < 2; ++i)
    scales[i] = curvature[i] > 0. ? 1. / std::sqrt(curvature[i]) : 1.;
  Vector lower =
      Vector::Constant(2 + n_free, -std::numeric_limits<double>::infinity());
  Vector upper = -lower;
  lower.head<2>() = Eigen::Vector2d(pressure_bounds[0], temperature_bounds[0])
                        .cwiseQuotient(scales.head<2>());
  upper.head<2>() = Eigen::Vector2d(pressure_bounds[1], temperature_bounds[1])
                        .cwiseQuotient(scales.head<2>());
  Eigen::Index sites = 0;
  for (const auto &phase : assemblage.get_phases())
    if (const auto *solution = dynamic_cast<const Solution *>(phase.get()))
      if (free_vectors(*solution))
        sites += solution->get_endmember_occupancies().cols();
  Matrix occupancy_matrix = Matrix::Zero(sites, 2 + n_free);
  Vector occupancy_baseline(sites);
  Eigen::Index row = 0, column = 2;
  for (const auto &phase : assemblage.get_phases())
    if (const auto *solution = dynamic_cast<const Solution *>(phase.get())) {
      const auto count = free_vectors(*solution);
      if (count) {
        const Matrix occ = solution->get_endmember_occupancies().matrix();
        occupancy_matrix.block(row, column, occ.cols(), count) =
            occ.transpose() * solution->free_compositional_vectors->transpose();
        occupancy_baseline.segment(row, occ.cols()) =
            occ.transpose() * solution->baseline_composition->matrix();
        row += occ.cols();
        column += count;
      }
    }
  FitContext context{assemblage,
                     R,
                     transformed,
                     include_prior,
                     scales,
                     occupancy_matrix,
                     occupancy_baseline,
                     Vector::Zero(2 + n_free),
                     {},
                     0};
  context.fixed[0] = std::clamp(initial[0] / scales[0], lower[0], upper[0]);
  context.fixed[1] = std::clamp(initial[1] / scales[1], lower[1], upper[1]);
  std::vector<double> values, active_lower, active_upper;
  for (Eigen::Index i = 0; i < lower.size(); ++i)
    if (lower[i] != upper[i]) {
      context.active.push_back(i);
      values.push_back(context.fixed[i]);
      active_lower.push_back(lower[i]);
      active_upper.push_back(upper[i]);
    }
  assemblage_set_state_from_params(assemblage, context.params(values));
  // Validate observation data before entering the optimizer so input errors
  // are reported directly, rather than interpreted as infeasible trial states.
  context.value(values);

  const auto run = [&](nlopt::algorithm algorithm) {
    nlopt::opt optimizer(algorithm,
                         static_cast<unsigned int>(context.active.size()));
    optimizer.set_min_objective(objective, &context);
    optimizer.set_lower_bounds(active_lower);
    optimizer.set_upper_bounds(active_upper);
    if (algorithm == nlopt::LD_SLSQP && sites)
      optimizer.add_inequality_mconstraint(
          occupancy_constraints, &context,
          std::vector<double>(static_cast<std::size_t>(sites), 1.e-10));
    optimizer.set_maxeval(max_it);
    optimizer.set_xtol_rel(1.e-9);
    optimizer.set_ftol_abs(1.e-9);
    std::vector<double> steps(context.active.size());
    for (std::size_t i = 0; i < steps.size(); ++i)
      steps[i] = 0.05 * (values[i] != 0. ? std::abs(values[i]) : 0.005);
    optimizer.set_initial_step(steps);
    double minimum = 0.;
    try {
      return static_cast<int>(optimizer.optimize(values, minimum));
    } catch (const nlopt::roundoff_limited &) {
      return static_cast<int>(nlopt::ROUNDOFF_LIMITED);
    }
  };
  if (n_free)
    run(nlopt::LN_NELDERMEAD);
  const int status = run(nlopt::LD_SLSQP);
  EstimateConditionsResult result;
  result.x = context.params(values);
  result.fun = context.value(values);
  result.status = status;
  result.success =
      status > 0 && status < static_cast<int>(nlopt::MAXEVAL_REACHED);
  result.message = result.success ? "Optimization converged."
                                  : "Optimization stopped before convergence.";
  std::vector<double> gradient(values.size());
  objective(values, gradient, &context);
  result.jac = Vector::Zero(result.x.size());
  for (std::size_t i = 0; i < gradient.size(); ++i)
    result.jac[context.active[i]] = gradient[i] / scales[context.active[i]];
  assemblage_set_state_from_params(assemblage, result.x);
  result.acov = affinity_covariance(assemblage, R, context.dataset, false);
  Matrix dmudx(assemblage.get_n_endmembers(), 2);
  dmudx.col(0) = assemblage.get_endmember_partial_volumes().matrix();
  dmudx.col(1) = -assemblage.get_endmember_partial_entropies().matrix();
  result.dadx = R * dmudx;
  const Eigen::LLT<Matrix> factor(result.acov);
  if (factor.info() != Eigen::Success)
    throw std::invalid_argument(
        "Affinity covariance must be positive definite.");
  Matrix information = result.dadx.transpose() * factor.solve(result.dadx);
  if (include_prior)
    information += *assemblage.state_inverse_covariances;
  Eigen::Vector2i free;
  Eigen::Index free_count = 0;
  if (!close(pressure_bounds))
    free[free_count++] = 0;
  if (!close(temperature_bounds))
    free[free_count++] = 1;
  result.xcov = Matrix::Zero(2, 2);
  if (free_count) {
    Matrix block(free_count, free_count);
    for (Eigen::Index i = 0; i < free_count; ++i)
      for (Eigen::Index j = 0; j < free_count; ++j)
        block(i, j) =
            information(free[i], free[j]) * scales[free[i]] * scales[free[j]];
    const Matrix inverse = inverse_covariance(block);
    for (Eigen::Index i = 0; i < free_count; ++i)
      for (Eigen::Index j = 0; j < free_count; ++j)
        result.xcov(free[i], free[j]) =
            inverse(i, j) * scales[free[i]] * scales[free[j]];
  }
  result.var = result.xcov.diagonal().cwiseMax(0.).cwiseSqrt();
  result.xcorr =
      result.xcov.array() / (result.var * result.var.transpose()).array();
  result.affinities = R * assemblage.get_partial_gibbs().matrix();
  result.weighted_affinities =
      inverse_covariance(result.acov, true) * result.affinities;
  result.n_reactions = utils::checked_int(R.rows());
  result.n_constraints = n_constraints;
  result.n_params = n_params;
  result.degrees_of_freedom = result.n_reactions + n_constraints - n_params;
  if (result.degrees_of_freedom > 0) {
    result.reduced_chisqr =
        result.fun / static_cast<double>(result.degrees_of_freedom);
    result.fit = std::sqrt(result.reduced_chisqr);
  }
  result.nfev = context.evaluations;
  return result;
}
} // namespace burnman::thermobarometry
