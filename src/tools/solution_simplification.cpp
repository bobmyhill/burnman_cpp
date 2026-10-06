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

#include "burnman/core/combined_mineral.hpp"
#include "burnman/core/solution_models/asymmetric_regular_solution.hpp"
#include "burnman/core/solution_models/ideal_solution.hpp"
#include "burnman/core/solution_models/symmetric_regular_solution.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/math_utils.hpp"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace burnman::polytope {
namespace {
using namespace solution_models;
using Interactions = std::vector<std::vector<double>>;

Eigen::Index rank(const Eigen::MatrixXd &matrix, double tolerance) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  return (svd.singularValues().array() > tolerance).count();
}
std::string number(double value) {
  if (std::abs(value - std::round(value)) < 1.e-12)
    return std::to_string(static_cast<long>(std::round(value)));
  // Fraction notation is understood by the native site-formula parser.
  for (long denominator = 2; denominator <= 10000; ++denominator) {
    double numerator = std::round(value * static_cast<double>(denominator));
    if (std::abs(numerator / static_cast<double>(denominator) - value) <
        1.e-12) {
      return std::to_string(static_cast<long>(numerator)) + "/" +
             std::to_string(denominator);
    }
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(16) << value;
  std::string result = stream.str();
  while (result.back() == '0')
    result.pop_back();
  return result;
}
std::string site_formula(const SolutionModel &model,
                         const Eigen::VectorXd &occupancies,
                         const Eigen::VectorXd &multiplicities) {
  std::string formula;
  Eigen::Index offset = 0;
  for (const auto &site : model.get_sites()) {
    Eigen::Index count = static_cast<Eigen::Index>(site.size());
    double sum = occupancies.segment(offset, count).sum();
    if (count == 0 || (sum == 0.0 && multiplicities[offset] == 0.0)) {
      formula += "[]0";
      offset += count;
      continue;
    }
    if (sum <= 0.0 || multiplicities[offset] < 0.0) {
      throw std::invalid_argument("Transformed endmembers have invalid site "
                                  "occupancies or multiplicities.");
    }
    formula += "[";
    for (Eigen::Index j = 0; j < count; ++j) {
      double amount = occupancies[offset + j] / sum;
      if (amount > 1.e-12) {
        formula += site[static_cast<std::size_t>(j)];
        if (std::abs(amount - 1.0) > 1.e-12)
          formula += number(amount);
      }
    }
    formula += "]";
    if (std::abs(multiplicities[offset] - 1.0) > 1.e-12)
      formula += number(multiplicities[offset]);
    offset += count;
  }
  return formula;
}
Eigen::VectorXd configurational_entropy(const Eigen::ArrayXXd &occupancies,
                                        const Eigen::ArrayXXd &multiplicities) {
  return (-constants::physics::gas_constant *
          (occupancies *
           (utils::logish(occupancies) - utils::logish(multiplicities)))
              .rowwise()
              .sum())
      .matrix();
}
Interactions interactions(const Eigen::MatrixXd &q,
                          const Eigen::VectorXd &alphas) {
  Interactions result(static_cast<std::size_t>(q.rows() - 1));
  for (Eigen::Index i = 0; i < q.rows() - 1; ++i) {
    for (Eigen::Index j = i + 1; j < q.rows(); ++j) {
      result[static_cast<std::size_t>(i)].push_back(
          (q(i, j) + q(j, i) - q(i, i) - q(j, j)) * (alphas[i] + alphas[j]) /
          2.0);
    }
  }
  return result;
}
std::shared_ptr<SolutionModel> clone_model(const SolutionModel &model) {
  if (typeid(model) == typeid(SymmetricRegularSolution))
    return std::make_shared<SymmetricRegularSolution>(
        static_cast<const SymmetricRegularSolution &>(model));
  if (typeid(model) == typeid(AsymmetricRegularSolution))
    return std::make_shared<AsymmetricRegularSolution>(
        static_cast<const AsymmetricRegularSolution &>(model));
  if (typeid(model) == typeid(IdealSolution))
    return std::make_shared<IdealSolution>(
        static_cast<const IdealSolution &>(model));
  throw std::invalid_argument("Solution simplification supports ideal and "
                              "symmetric/asymmetric regular models.");
}
std::shared_ptr<Material> clone_phase(const std::shared_ptr<Material> &phase) {
  if (auto solution = std::dynamic_pointer_cast<Solution>(phase)) {
    auto result = std::make_shared<Solution>(*solution);
    result->set_solution_model(clone_model(*solution->get_solution_model()));
    result->reset_cache();
    return result;
  }
  return std::make_shared<Mineral>(*std::dynamic_pointer_cast<Mineral>(phase));
}

// Prefer original endmembers where possible; then use deterministic ordering
// for derived physical vertices. Do not impose nonnegative basis coordinates.
Eigen::MatrixXd independent_basis(const Eigen::MatrixXd &candidates,
                                  double tolerance) {
  std::vector<Eigen::Index> indices(
      static_cast<std::size_t>(candidates.rows()));
  std::iota(indices.begin(), indices.end(), 0);
  auto pivot = [&](Eigen::Index i) {
    for (Eigen::Index j = 0; j < candidates.cols(); ++j)
      if (std::abs(candidates(i, j)) > tolerance)
        return j;
    return candidates.cols();
  };
  auto nonzero = [&](Eigen::Index i) {
    return (candidates.row(i).array().abs() > tolerance).count();
  };
  std::sort(indices.begin(), indices.end(),
            [&](Eigen::Index i, Eigen::Index j) {
              bool pure_i = nonzero(i) == 1, pure_j = nonzero(j) == 1;
              if (pure_i != pure_j)
                return pure_i;
              if (pivot(i) != pivot(j))
                return pivot(i) < pivot(j);
              for (Eigen::Index k = 0; k < candidates.cols(); ++k) {
                if (candidates(i, k) != candidates(j, k))
                  return candidates(i, k) > candidates(j, k);
              }
              return i < j;
            });
  Eigen::MatrixXd basis(0, candidates.cols());
  for (auto index : indices) {
    Eigen::MatrixXd extended(basis.rows() + 1, basis.cols());
    extended.topRows(basis.rows()) = basis;
    extended.bottomRows(1) = candidates.row(index);
    if (rank(extended, tolerance) > basis.rows())
      basis = std::move(extended);
  }
  // Put the first nonzero coordinate in echelon order, preserving pure members.
  std::vector<Eigen::Index> order(static_cast<std::size_t>(basis.rows()));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](Eigen::Index i, Eigen::Index j) {
        Eigen::Index a = 0, b = 0;
        while (a < basis.cols() && std::abs(basis(i, a)) <= tolerance)
          ++a;
        while (b < basis.cols() && std::abs(basis(j, b)) <= tolerance)
          ++b;
        return a < b;
      });
  Eigen::MatrixXd result = basis;
  for (std::size_t i = 0; i < order.size(); ++i)
    result.row(static_cast<Eigen::Index>(i)) = basis.row(order[i]);
  return result;
}
} // namespace

std::shared_ptr<Material> transform_solution_to_new_basis(
    const Solution &solution, const Eigen::MatrixXd &basis,
    const Eigen::ArrayXd &fractions, const std::string &solution_name) {
  auto original = solution.get_solution_model();
  if (!original || !basis.rows() ||
      basis.cols() != original->get_n_endmembers() || !basis.allFinite() ||
      (basis.rowwise().sum().array() - 1.0).abs().maxCoeff() > 1.e-10 ||
      rank(basis, 1.e-10) != basis.rows()) {
    throw std::invalid_argument(
        "New basis must have independent finite rows summing to one, with one "
        "column per original endmember.");
  }
  const auto *regular =
      dynamic_cast<const AsymmetricRegularSolution *>(original.get());
  const auto &original_model = *original;
  bool symmetric = typeid(original_model) == typeid(SymmetricRegularSolution);
  if (typeid(original_model) != typeid(IdealSolution) && !symmetric &&
      typeid(original_model) != typeid(AsymmetricRegularSolution)) {
    throw std::invalid_argument("Basis transformations support ideal and "
                                "symmetric/asymmetric regular models.");
  }
  Eigen::MatrixXd occupancies =
      basis * original->get_endmember_occupancies().matrix();
  Eigen::ArrayXXd noccs =
      (basis * original->get_endmember_n_occupancies().matrix()).array();
  Eigen::ArrayXXd multiplicities =
      (basis * original->get_site_multiplicities().matrix()).array();
  if (occupancies.minCoeff() < -1.e-10 || noccs.minCoeff() < -1.e-10 ||
      multiplicities.minCoeff() < 0.0) {
    throw std::invalid_argument(
        "New basis must give nonnegative site occupancies and multiplicities.");
  }
  occupancies = occupancies.cwiseMax(0.0);
  noccs = noccs.max(0.0);
  Eigen::Index n = basis.rows();
  Eigen::VectorXd alphas = Eigen::VectorXd::Ones(n);
  Eigen::MatrixXd qe = Eigen::MatrixXd::Zero(n, n), qs = qe, qv = qe;
  Eigen::MatrixXd corrections = Eigen::MatrixXd::Zero(n, 3);
  if (regular) {
    alphas = basis * regular->get_alphas().matrix();
    if (alphas.minCoeff() <= 0.0 || !alphas.allFinite()) {
      throw std::invalid_argument(
          "Transformed van Laar parameters must be positive and finite.");
    }
    Eigen::MatrixXd b = regular->get_alphas().matrix().asDiagonal() *
                        basis.transpose() * alphas.cwiseInverse().asDiagonal();
    // Materialize the first product before applying the second basis factor.
    // This also avoids GCC's uninitialized temporary warning for lazy products.
    qe = (b.transpose() * regular->get_energy_interactions()).eval() * b;
    qs = (b.transpose() * regular->get_entropy_interactions()).eval() * b;
    qv = (b.transpose() * regular->get_volume_interactions()).eval() * b;
    corrections.col(0) = qe.diagonal().cwiseProduct(alphas);
    corrections.col(1) = qs.diagonal().cwiseProduct(alphas);
    corrections.col(2) = qv.diagonal().cwiseProduct(alphas);
  }
  // Account for both new and original standard-state configurational entropy.
  corrections.col(1) +=
      configurational_entropy(noccs, multiplicities) -
      basis * configurational_entropy(original->get_endmember_n_occupancies(),
                                      original->get_site_multiplicities());
  types::PairedEndmemberList members;
  for (Eigen::Index i = 0; i < n; ++i) {
    std::string sites = site_formula(*original, occupancies.row(i),
                                     multiplicities.row(i).matrix());
    Eigen::Index selector = -1;
    for (Eigen::Index j = 0; j < basis.cols(); ++j) {
      if ((basis.row(i).transpose() - Eigen::VectorXd::Unit(basis.cols(), j))
              .norm() < 1.e-12)
        selector = j;
    }
    if (selector >= 0 && corrections.row(i).norm() < 1.e-10) {
      members.emplace_back(
          original->endmembers[static_cast<std::size_t>(selector)], sites);
    } else {
      members.emplace_back(
          make_combined_mineral(original->endmembers,
                                basis.row(i).transpose().array(),
                                corrections.row(i),
                                "Derived member (occupancies: " + sites + ")"),
          sites);
    }
  }
  std::string name = solution_name.empty()
                         ? solution.get_name() + " (transformed)"
                         : solution_name;
  std::shared_ptr<Material> result;
  if (n == 1) {
    auto mineral = std::make_shared<Mineral>(members[0].first);
    mineral->set_name(name);
    result = mineral;
  } else {
    auto transformed = std::make_shared<Solution>();
    std::shared_ptr<SolutionModel> model;
    if (!regular)
      model = std::make_shared<IdealSolution>(members);
    else if (symmetric)
      model = std::make_shared<SymmetricRegularSolution>(
          members, interactions(qe, alphas), interactions(qv, alphas),
          interactions(qs, alphas));
    else
      model = std::make_shared<AsymmetricRegularSolution>(
          members, std::vector<double>(alphas.data(), alphas.data() + n),
          interactions(qe, alphas), interactions(qv, alphas),
          interactions(qs, alphas));
    transformed->set_solution_model(model);
    Eigen::ArrayXd composition = fractions;
    if (composition.size() == 0) {
      auto previous = solution.get_molar_fractions();
      composition = Eigen::ArrayXd::Constant(n, 1.0 / static_cast<double>(n));
      if (previous.size() == basis.cols() && previous.isFinite().all()) {
        Eigen::VectorXd x =
            basis.transpose().completeOrthogonalDecomposition().solve(
                previous.matrix());
        if ((basis.transpose() * x - previous.matrix()).norm() < 1.e-10)
          composition = x.array();
      }
    }
    if (composition.size() != n || !composition.isFinite().all() ||
        std::abs(composition.sum() - 1.0) > 1.e-10 ||
        (model->get_endmember_occupancies().matrix().transpose() *
         composition.matrix())
                .minCoeff() < -1.e-10) {
      throw std::invalid_argument("Transformed composition must sum to one and "
                                  "give nonnegative site occupancies.");
    }
    transformed->set_composition(composition);
    transformed->set_name(name);
    transformed->set_basis(basis * solution.get_basis());
    result = transformed;
  }
  if (solution.has_state())
    result->set_state(solution.get_pressure(), solution.get_temperature());
  return result;
}

std::shared_ptr<Assemblage> simplify_composite_with_composition(
    const Assemblage &composite, const types::FormulaMap &composition,
    double tolerance, double rational_tolerance) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance > 1.e-6) {
    throw std::invalid_argument(
        "Simplification tolerance must be finite, positive and at most 1e-6.");
  }
  auto poly = composite_polytope_at_constrained_composition(
      composite, composition, rational_tolerance);
  if (poly.is_empty() || poly.get_vertices().rows() == 0) {
    throw std::invalid_argument(
        "Bulk composition is infeasible for this assemblage.");
  }
  if (!poly.is_bounded())
    throw std::invalid_argument(
        "Assemblage amounts must form a bounded polytope.");
  const auto &vertices = poly.get_vertices();
  double total = vertices.rowwise().sum().maxCoeff();
  if (total <= 0.0)
    throw std::invalid_argument(
        "Bulk composition must specify a positive amount of material.");
  auto result = std::make_shared<Assemblage>();
  Eigen::Index offset = 0;
  for (Eigen::Index i = 0; i < composite.get_n_phases(); ++i) {
    auto phase = composite.get_phase(static_cast<std::size_t>(i));
    auto solution = std::dynamic_pointer_cast<Solution>(phase);
    Eigen::Index n = solution ? solution->get_n_endmembers() : 1;
    Eigen::MatrixXd amounts = vertices.middleCols(offset, n);
    offset += n;
    if (amounts.rowwise().sum().maxCoeff() <= tolerance * total)
      continue;
    if (!solution || rank(amounts / total, tolerance) == n) {
      result->add_phases({clone_phase(phase)});
      continue;
    }
    Eigen::VectorXd mean = amounts.colwise().mean();
    mean /= mean.sum();
    auto occupancies = solution->get_endmember_occupancies().matrix().eval();
    auto full = solution_polytope_from_endmember_occupancies(
        occupancies, rational_tolerance);
    Eigen::VectorXd mean_occupancies = occupancies.transpose() * mean;
    std::vector<Eigen::Index> allowed;
    for (Eigen::Index vertex = 0; vertex < full.get_vertices().rows();
         ++vertex) {
      bool valid = true;
      for (Eigen::Index site = 0; site < occupancies.cols(); ++site) {
        if (mean_occupancies[site] <= tolerance &&
            full.get_endmember_occupancies()(vertex, site) > tolerance) {
          valid = false;
          break;
        }
      }
      if (valid)
        allowed.push_back(vertex);
    }
    Eigen::MatrixXd candidates(static_cast<Eigen::Index>(allowed.size()), n);
    for (std::size_t j = 0; j < allowed.size(); ++j)
      candidates.row(static_cast<Eigen::Index>(j)) =
          full.get_vertices().row(allowed[j]);
    if (!candidates.rows())
      throw std::runtime_error(
          "No physical solution vertices span the feasible composition.");
    Eigen::MatrixXd basis = independent_basis(candidates, tolerance);
    if (basis.rows() < n)
      result->add_phases({transform_solution_to_new_basis(*solution, basis)});
    else
      result->add_phases({clone_phase(phase)});
  }
  if (!result->get_n_phases())
    throw std::runtime_error("Simplification removed all phases.");
  // A reduced assemblage is a new phase model, not an equilibrium phase split.
  result->set_fractions(Eigen::ArrayXd::Constant(
      result->get_n_phases(),
      1.0 / static_cast<double>(result->get_n_phases())));
  result->set_n_moles(composite.has_n_moles() ? composite.get_n_moles() : 1.0);
  result->set_averaging_scheme(types::AveragingType::VRH);
  if (composite.has_state())
    result->set_state(composite.get_pressure(), composite.get_temperature());
  return result;
}
} // namespace burnman::polytope
