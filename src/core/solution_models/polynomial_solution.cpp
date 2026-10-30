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

#include "burnman/core/solution_models/polynomial_solution.hpp"
#include "burnman/utils/constants.hpp"
#include <cmath>

namespace burnman::solution_models {
PolynomialSolution::PolynomialSolution(
    const types::PairedEndmemberList &endmember_list, const Interactions &ESV,
    const std::vector<std::shared_ptr<Mineral>> &minerals,
    const Interactions &mineral_interactions,
    const std::optional<Eigen::MatrixXd> &transformation)
    : IdealSolution(endmember_list), interaction_endmembers_(minerals),
      dqdp_(transformation
                ? *transformation
                : Eigen::MatrixXd::Identity(n_endmembers, n_endmembers)),
      ESV_interactions_(ESV),
      endmember_coefficients_and_interactions_(mineral_interactions) {
  if (dqdp_.rows() != n_endmembers || dqdp_.cols() != n_endmembers ||
      !dqdp_.allFinite())
    throw std::invalid_argument("Polynomial transformation must be finite and "
                                "square, matching the endmembers.");
  for (const auto &mineral : minerals)
    if (!mineral)
      throw std::invalid_argument("Interaction endmembers cannot be null.");
  auto append = [&](const Interactions &rows, Eigen::Index count,
                    bool mineral) {
    for (const auto &row : rows) {
      if (static_cast<Eigen::Index>(row.size()) < count ||
          (row.size() - static_cast<std::size_t>(count)) % 2)
        throw std::invalid_argument(
            "Polynomial interactions require coefficients followed by "
            "index/exponent pairs.");
      Term term{Eigen::VectorXd(count), Eigen::VectorXd::Zero(n_endmembers),
                mineral};
      for (Eigen::Index i = 0; i < count; ++i) {
        term.coefficients[i] = row[static_cast<std::size_t>(i)];
        if (!std::isfinite(term.coefficients[i]))
          throw std::invalid_argument(
              "Polynomial coefficients must be finite.");
      }
      for (std::size_t i = static_cast<std::size_t>(count); i < row.size();
           i += 2) {
        const double index = row[i], exponent = row[i + 1];
        if (!std::isfinite(index) || index < 0. ||
            index >= static_cast<double>(n_endmembers) ||
            std::floor(index) != index || !std::isfinite(exponent) ||
            exponent < 0.)
          throw std::invalid_argument(
              "Polynomial indices must be valid integers and exponents finite "
              "and nonnegative.");
        term.exponents[static_cast<Eigen::Index>(index)] += exponent;
      }
      terms_.push_back(std::move(term));
    }
  };
  append(ESV, 3, false);
  if (minerals.empty() && !mineral_interactions.empty())
    throw std::invalid_argument(
        "Mineral interactions require interaction endmembers.");
  append(mineral_interactions, static_cast<Eigen::Index>(minerals.size()),
         true);
}

std::shared_ptr<SolutionModel> PolynomialSolution::clone() const {
  auto result = std::make_shared<PolynomialSolution>(*this);
  for (auto &mineral : result->interaction_endmembers_)
    mineral = std::dynamic_pointer_cast<Mineral>(mineral->copy());
  for (auto &mineral : result->polymorphic_endmembers)
    mineral = std::dynamic_pointer_cast<Mineral>(mineral->copy());
  return result;
}
const std::vector<std::shared_ptr<Mineral>> &
PolynomialSolution::get_interaction_endmembers() const {
  return interaction_endmembers_;
}
const Eigen::MatrixXd &PolynomialSolution::get_dqdp() const { return dqdp_; }
const PolynomialSolution::Interactions &
PolynomialSolution::get_ESV_interactions() const {
  return ESV_interactions_;
}
const PolynomialSolution::Interactions &
PolynomialSolution::get_endmember_coefficients_and_interactions() const {
  return endmember_coefficients_and_interactions_;
}

PolynomialSolution::Derivatives
PolynomialSolution::evaluate(double pressure, double temperature,
                             const Eigen::ArrayXd &fractions,
                             Property property) const {
  if (fractions.size() != n_endmembers || !fractions.allFinite())
    throw std::invalid_argument(
        "Polynomial fractions must match the endmembers and be finite.");
  const Eigen::VectorXd q = dqdp_ * fractions.matrix();
  // dq/dn at one mole. Differentiating n*g(n_i/n) gives the partial molar
  // gradient g + J^T*dg/dq and Hessian J^T*d2g/dq2*J.
  const Eigen::MatrixXd J =
      dqdp_ * (Eigen::MatrixXd::Identity(n_endmembers, n_endmembers) -
               fractions.matrix() * Eigen::RowVectorXd::Ones(n_endmembers));
  Eigen::VectorXd mineral_values(interaction_endmembers_.size());
  for (std::size_t i = 0; i < interaction_endmembers_.size(); ++i) {
    auto &m = *interaction_endmembers_[i];
    if (!m.has_state() || m.get_pressure() != pressure ||
        m.get_temperature() != temperature)
      m.set_state(pressure, temperature);
    double value;
    switch (property) {
    case Property::Gibbs:
      value = m.get_molar_gibbs();
      break;
    case Property::Entropy:
      value = m.get_molar_entropy();
      break;
    case Property::Volume:
      value = m.get_molar_volume();
      break;
    case Property::Cp:
      value = m.get_molar_heat_capacity_p();
      break;
    case Property::AlphaV:
      value = m.get_thermal_expansivity() * m.get_molar_volume();
      break;
    case Property::VoverKT:
      value = m.get_molar_volume() / m.get_isothermal_bulk_modulus_reuss();
      break;
    default:
      throw std::invalid_argument("Unknown polynomial interaction property.");
    }
    mineral_values[static_cast<Eigen::Index>(i)] = value;
  }
  double value = 0.;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n_endmembers);
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(n_endmembers, n_endmembers);
  for (const auto &term : terms_) {
    double coefficient = 0.;
    if (term.mineral)
      coefficient = term.coefficients.dot(mineral_values);
    else if (property == Property::Gibbs)
      coefficient = term.coefficients[0] - temperature * term.coefficients[1] +
                    pressure * term.coefficients[2];
    else if (property == Property::Entropy)
      coefficient = term.coefficients[1];
    else if (property == Property::Volume)
      coefficient = term.coefficients[2];
    if (coefficient == 0.)
      continue;
    auto product = [&](Eigen::Index i = -1, Eigen::Index j = -1) {
      double c = 1.;
      for (Eigen::Index k = 0; k < n_endmembers; ++k)
        c *=
            std::pow(q[k], term.exponents[k] - double(k == i) - double(k == j));
      return c;
    };
    value += coefficient * product();
    for (Eigen::Index i = 0; i < n_endmembers; ++i) {
      const double a = term.exponents[i];
      if (a == 0.)
        continue;
      gradient[i] += coefficient * a * product(i);
      for (Eigen::Index j = 0; j < n_endmembers; ++j) {
        const double b = term.exponents[j] - double(i == j);
        if (b != 0.)
          hessian(i, j) += coefficient * a * b * product(i, j);
      }
    }
  }
  if (!std::isfinite(value) || !gradient.allFinite() || !hessian.allFinite())
    throw std::domain_error(
        "Polynomial interactions are undefined at this composition.");
  return {value,
          (Eigen::VectorXd::Constant(n_endmembers, value) +
           J.transpose() * gradient)
              .array(),
          J.transpose() * hessian * J};
}

#define PARTIAL(name, prop)                                                    \
  Eigen::ArrayXd PolynomialSolution::compute_excess_partial_##name(            \
      double p, double t, const Eigen::ArrayXd &x) const {                     \
    return IdealSolution::compute_excess_partial_##name(p, t, x) +             \
           evaluate(p, t, x, Property::prop).gradient;                         \
  }
PARTIAL(gibbs_free_energies, Gibbs)
PARTIAL(entropies, Entropy)
PARTIAL(volumes, Volume)
#undef PARTIAL
#define HESSIAN(name, prop)                                                    \
  Eigen::MatrixXd PolynomialSolution::compute_##name##_hessian(                \
      double p, double t, const Eigen::ArrayXd &x) const {                     \
    return IdealSolution::compute_##name##_hessian(p, t, x) +                  \
           evaluate(p, t, x, Property::prop).hessian;                          \
  }
HESSIAN(gibbs, Gibbs)
HESSIAN(entropy, Entropy)
HESSIAN(volume, Volume)
#undef HESSIAN
#define THERMAL(name, prop)                                                    \
  double PolynomialSolution::compute_##name##_excess(                          \
      double p, double t, const Eigen::ArrayXd &x) const {                     \
    return evaluate(p, t, x, Property::prop).value;                            \
  }
THERMAL(Cp, Cp)
THERMAL(alphaV, AlphaV)
THERMAL(VoverKT, VoverKT)
#undef THERMAL
Eigen::ArrayXd PolynomialSolution::compute_activity_coefficients(
    double p, double t, const Eigen::ArrayXd &x) const {
  if (t <= 1.e-10)
    throw std::domain_error("Activity coefficients undefined at 0 K.");
  return (evaluate(p, t, x, Property::Gibbs).gradient /
          (constants::physics::gas_constant * t))
      .exp();
}
Eigen::ArrayXd
PolynomialSolution::compute_activities(double p, double t,
                                       const Eigen::ArrayXd &x) const {
  return IdealSolution::compute_activities(p, t, x) *
         compute_activity_coefficients(p, t, x);
}
} // namespace burnman::solution_models
