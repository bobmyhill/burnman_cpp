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
