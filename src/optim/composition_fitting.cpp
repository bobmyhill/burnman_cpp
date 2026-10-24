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

#include "burnman/optim/composition_fitting.hpp"
#include "burnman/utils/chemistry_utils.hpp"
#include "burnman/utils/exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burnman::optim {
namespace {
// The elemental formula defines composition and mass, but no equation of state.
class CompositionOnlyEOS : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &) override {}
};

class CompositionSolutionModel : public solution_models::SolutionModel {
public:
  explicit CompositionSolutionModel(
      const types::PairedEndmemberList &endmember_list)
      : SolutionModel(endmember_list) {}

  std::shared_ptr<SolutionModel> clone() const override {
    return std::make_shared<CompositionSolutionModel>(*this);
  }

#define UNAVAILABLE_PROPERTY(type, name)                                       \
  type name(double, double, const Eigen::ArrayXd &) const override {           \
    throw exceptions::NotImplementedError(                                     \
        "DummyCompositionSolution", #name,                                     \
        "No thermodynamic model is defined.");                                 \
  }
  UNAVAILABLE_PROPERTY(Eigen::ArrayXd,
                       compute_excess_partial_gibbs_free_energies)
  UNAVAILABLE_PROPERTY(Eigen::ArrayXd, compute_excess_partial_entropies)
  UNAVAILABLE_PROPERTY(Eigen::ArrayXd, compute_excess_partial_volumes)
  UNAVAILABLE_PROPERTY(Eigen::ArrayXd, compute_activities)
  UNAVAILABLE_PROPERTY(Eigen::ArrayXd, compute_activity_coefficients)
  UNAVAILABLE_PROPERTY(Eigen::MatrixXd, compute_gibbs_hessian)
  UNAVAILABLE_PROPERTY(Eigen::MatrixXd, compute_entropy_hessian)
  UNAVAILABLE_PROPERTY(Eigen::MatrixXd, compute_volume_hessian)
#undef UNAVAILABLE_PROPERTY
};

types::PairedEndmemberList
composition_endmembers(const std::vector<std::string> &element_formulae,
                       const std::vector<std::string> &site_formulae) {
  if (element_formulae.empty() ||
      element_formulae.size() != site_formulae.size())
    throw std::invalid_argument(
        "Elemental and site formulae must have the same nonzero length.");
  types::PairedEndmemberList endmembers;
  endmembers.reserve(element_formulae.size());
  for (std::size_t i = 0; i < element_formulae.size(); ++i) {
    bool in_site = false;
    std::size_t sites = 0;
    for (std::size_t j = 0; j < site_formulae[i].size(); ++j) {
      const char character = site_formulae[i][j];
      if (character == '[') {
        if (in_site)
          throw std::invalid_argument("Nested brackets in a site formula.");
        in_site = true;
        ++sites;
      } else if (character == ']') {
        if (!in_site || site_formulae[i][j - 1] == '[')
          throw std::invalid_argument("Unmatched or empty site brackets.");
        in_site = false;
      }
    }
    if (in_site || sites == 0)
      throw std::invalid_argument(
          "Site formulae require complete bracketed sites.");
    Mineral endmember;
    endmember.params.name = element_formulae[i];
    endmember.params.formula = utils::dictionarize_formula(element_formulae[i]);
    endmember.params.molar_mass =
        utils::formula_mass(*endmember.params.formula);
    endmember.set_method(std::make_shared<CompositionOnlyEOS>());
    endmembers.emplace_back(endmember, site_formulae[i]);
  }
  return endmembers;
}

Eigen::MatrixXd unique_nonzero_rows(const Eigen::MatrixXd &rows) {
  std::vector<Eigen::RowVectorXd> unique;
  for (Eigen::Index i = 0; i < rows.rows(); ++i) {
    if ((rows.row(i).array().abs() <= 1.e-10).all())
      continue;
    const auto duplicate = std::find_if(
        unique.begin(), unique.end(), [&](const Eigen::RowVectorXd &row) {
          return (row.array() == rows.row(i).array()).all();
        });
    if (duplicate == unique.end())
      unique.push_back(rows.row(i));
  }
  Eigen::MatrixXd result(static_cast<Eigen::Index>(unique.size()), rows.cols());
  for (std::size_t i = 0; i < unique.size(); ++i)
    result.row(static_cast<Eigen::Index>(i)) = unique[i];
  return result;
}
} // namespace

DummyCompositionSolution::DummyCompositionSolution(
    const std::vector<std::string> &endmember_element_formulae,
    const std::vector<std::string> &endmember_site_formulae) {
  auto model =
      std::make_shared<CompositionSolutionModel>(composition_endmembers(
          endmember_element_formulae, endmember_site_formulae));
  const auto &occupancies = model->get_endmember_occupancies();
  const auto &multiplicities = model->get_site_multiplicities();
  if (!occupancies.isFinite().all() || (occupancies < 0.).any() ||
      !multiplicities.isFinite().all() || (multiplicities <= 0.).any())
    throw std::invalid_argument(
        "Site occupancies must be finite and nonnegative, "
        "with positive finite multiplicities.");
  Eigen::Index column = 0;
  for (const auto &site : model->get_sites()) {
    const auto size = static_cast<Eigen::Index>(site.size());
    if (size == 0 || (occupancies.middleCols(column, size).rowwise().sum() - 1.)
                             .abs()
                             .maxCoeff() > 1.e-12)
      throw std::invalid_argument(
          "Species fractions must sum to one on each site.");
    column += size;
  }
  set_solution_model(model);
}

std::shared_ptr<Material> DummyCompositionSolution::copy() const {
  auto result = std::make_shared<DummyCompositionSolution>(*this);
  result->set_solution_model(get_solution_model()->clone());
  return result;
}

void DummyCompositionSolution::set_composition(
    const Eigen::ArrayXd &fractions) {
  if (fractions.size() != get_n_endmembers() || !fractions.isFinite().all() ||
      std::abs(fractions.sum() - 1.) > 1.e-12)
    throw std::invalid_argument("Molar fractions must be finite, match the "
                                "endmembers, and sum to one.");
  if ((get_endmember_occupancies().matrix().transpose() * fractions.matrix())
          .minCoeff() < -1.e-10)
    throw std::invalid_argument(
        "Molar fractions must give nonnegative site occupancies.");
  Solution::set_composition(fractions);
}

LeastSquaresResult fit_composition_to_solution(
    const Solution &solution, const std::vector<std::string> &fitted_variables,
    const Eigen::VectorXd &variable_values,
    const std::optional<Eigen::MatrixXd> &variable_covariances,
    const VariableConversions &variable_conversions, bool normalize) {
  if (!solution.get_solution_model())
    throw std::invalid_argument(
        "The solution must have an endmember site model.");
  if (fitted_variables.empty() ||
      static_cast<Eigen::Index>(fitted_variables.size()) !=
          variable_values.size() ||
      !variable_values.allFinite())
    throw std::invalid_argument(
        "Fitted variables require one finite observed amount per name.");

  auto solution_variables = solution.get_elements();
  const auto site_names = solution.get_site_names();
  solution_variables.insert(solution_variables.end(), site_names.begin(),
                            site_names.end());
  const auto n_endmembers = solution.get_n_endmembers();
  const auto n_variables = static_cast<Eigen::Index>(solution_variables.size());
  Eigen::MatrixXd solution_matrix = Eigen::MatrixXd::Zero(
      n_endmembers,
      n_variables + static_cast<Eigen::Index>(variable_conversions.size()));
  solution_matrix.leftCols(solution.get_n_elements()) =
      solution.get_stoichiometric_matrix();
  solution_matrix.middleCols(solution.get_n_elements(),
                             static_cast<Eigen::Index>(site_names.size())) =
      solution.get_endmember_n_occupancies().matrix();

  std::map<std::string, Eigen::Index> indices;
  for (Eigen::Index i = 0; i < n_variables; ++i)
    indices.emplace(solution_variables[static_cast<std::size_t>(i)], i);
  const auto variable_index = [&](const std::string &name) {
    const auto found = indices.find(name);
    if (found == indices.end())
      throw std::invalid_argument("Unknown compositional variable: " + name);
    return found->second;
  };
  Eigen::Index column = n_variables;
  for (const auto &[name, terms] : variable_conversions) {
    if (name.empty() || indices.count(name))
      throw std::invalid_argument(
          "New compositional variable names must be nonempty and unique: " +
          name);
    for (const auto &[variable, coefficient] : terms) {
      if (!std::isfinite(coefficient))
        throw std::invalid_argument(
            "Variable conversion coefficients must be finite.");
      solution_matrix.col(column) +=
          coefficient * solution_matrix.col(variable_index(variable));
    }
    indices.emplace(name, column++);
  }

  Eigen::MatrixXd A(variable_values.size(), n_endmembers);
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    A.row(i) =
        solution_matrix
            .col(variable_index(fitted_variables[static_cast<std::size_t>(i)]))
            .transpose();

  // Every explicitly absent component imposes an exact equality, including
  // repeated observations of the same compositional variable.
  Eigen::MatrixXd zero_amounts(A.rows(), n_endmembers);
  Eigen::Index n_zero = 0;
  for (Eigen::Index i = 0; i < A.rows(); ++i)
    if (std::abs(variable_values[i]) < 1.e-10)
      zero_amounts.row(n_zero++) = A.row(i);
  const Eigen::MatrixXd S = unique_nonzero_rows(zero_amounts.topRows(n_zero));
  const Eigen::MatrixXd T = unique_nonzero_rows(
      -solution.get_endmember_occupancies().matrix().transpose());
  const LinearConstraints equality_constraints{S,
                                               Eigen::VectorXd::Zero(S.rows())};
  const LinearConstraints inequality_constraints{
      T, Eigen::VectorXd::Zero(T.rows())};
  auto [popt, pcov, residual] = weighted_constrained_least_squares(
      A, variable_values, variable_covariances, equality_constraints,
      inequality_constraints);

  if (normalize) {
    const double sum = popt.sum();
    if (!std::isfinite(sum) || sum <= 0.)
      throw std::invalid_argument("Cannot normalize a composition with a "
                                  "nonpositive total endmember amount.");
    popt /= sum;
    pcov /= sum * sum;
    residual /= sum;
  }
  const Eigen::MatrixXd dpdx =
      Eigen::MatrixXd::Identity(n_endmembers, n_endmembers) -
      popt * Eigen::RowVectorXd::Ones(n_endmembers);
  pcov = (dpdx * pcov * dpdx.transpose()).eval();
  if (!popt.allFinite() || !pcov.allFinite() || !std::isfinite(residual))
    throw std::runtime_error("The fitted composition exceeds numerical range.");
  return {popt, pcov, residual};
}

LeastSquaresResult fit_phase_proportions_to_bulk_composition(
    const Eigen::MatrixXd &phase_compositions,
    const Eigen::VectorXd &bulk_composition) {
  const auto n_phases = phase_compositions.cols();
  const LinearConstraints inequality_constraints{
      -Eigen::MatrixXd::Identity(n_phases, n_phases),
      Eigen::VectorXd::Zero(n_phases)};
  return weighted_constrained_least_squares(
      phase_compositions, bulk_composition, std::nullopt, std::nullopt,
      inequality_constraints);
}
} // namespace burnman::optim
