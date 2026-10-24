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

#include "burnman/tools/chemistry.hpp"
#include "burnman/core/equilibrated_composite.hpp"
#include "burnman/utils/chemistry_utils.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/matrix_utils.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace burnman::chemistry {
namespace {
void check_reaction(const Materials &minerals, const Eigen::VectorXd &amounts) {
  if (minerals.empty() ||
      amounts.size() != static_cast<Eigen::Index>(minerals.size()) ||
      !amounts.allFinite() || amounts.cwiseAbs().maxCoeff() == 0.)
    throw std::invalid_argument(
        "Supply minerals and a matching nonzero, finite reaction vector.");
  for (const auto &mineral : minerals)
    if (!mineral)
      throw std::invalid_argument("Reaction materials cannot be null.");
}
void check_state(double pressure, double temperature) {
  if (!std::isfinite(pressure) || !std::isfinite(temperature) ||
      temperature <= 0.)
    throw std::invalid_argument(
        "Pressure must be finite and temperature positive and finite.");
}
struct Reaction {
  double affinity = 0., volume = 0., entropy = 0.;
};
Reaction evaluate(const Materials &minerals, const Eigen::VectorXd &amounts,
                  double pressure, double temperature) {
  check_state(pressure, temperature);
  Reaction result;
  for (Eigen::Index i = 0; i < amounts.size(); ++i) {
    auto &mineral = *minerals[static_cast<std::size_t>(i)];
    mineral.set_state(pressure, temperature);
    if (amounts[i] == 0.)
      continue;
    result.affinity += amounts[i] * mineral.get_molar_gibbs();
    result.volume += amounts[i] * mineral.get_molar_volume();
    result.entropy += amounts[i] * mineral.get_molar_entropy();
  }
  if (!std::isfinite(result.affinity) || !std::isfinite(result.volume) ||
      !std::isfinite(result.entropy))
    throw std::runtime_error("Nonfinite reaction properties.");
  return result;
}
// Backtracking keeps Newton estimates in the domain of the material models.
// Derivatives use molar volumes, entropies and heat capacities in SI units.
double
scalar_root(const std::function<std::pair<double, double>(double)> &function,
            double initial, double step_limit, bool positive) {
  if (!std::isfinite(initial) || (positive && initial <= 0.))
    throw std::invalid_argument(
        "Supply a finite initial estimate in the material's domain.");
  double x = initial;
  for (int iteration = 0; iteration < 150; ++iteration) {
    const auto [value, derivative] = function(x);
    if (std::isfinite(value) && std::abs(value) < 1.e-6)
      return x;
    if (!std::isfinite(value) || !std::isfinite(derivative) || derivative == 0.)
      throw std::runtime_error("Cannot find equilibrium: the reaction "
                               "derivative is zero or nonfinite.");
    const double step =
        std::clamp(-value / derivative, -step_limit, step_limit);
    bool accepted = false;
    for (double fraction = 1.; fraction >= 1.e-12; fraction *= .5) {
      const double trial = x + fraction * step;
      if (!std::isfinite(trial) || (positive && trial <= 0.))
        continue;
      try {
        const double residual = function(trial).first;
        if (std::isfinite(residual) && std::abs(residual) < std::abs(value)) {
          x = trial;
          accepted = true;
          break;
        }
      } catch (const std::exception &) {
        // A shorter step can remain within the equation of state's domain.
      }
    }
    if (!accepted)
      break;
  }
  function(x);
  throw std::runtime_error(
      "Equilibrium calculation did not converge to zero residual.");
}
void check_same_state(const Material &first, const Material &second) {
  if (!first.has_state() || !second.has_state())
    throw std::runtime_error(
        "Set both material states before computing fugacity.");
  check_state(first.get_pressure(), first.get_temperature());
  check_state(second.get_pressure(), second.get_temperature());
  const double pressure_scale = std::max(
      {1., std::abs(first.get_pressure()), std::abs(second.get_pressure())});
  if (std::abs(first.get_pressure() - second.get_pressure()) >
          1.e-10 * pressure_scale ||
      std::abs(first.get_temperature() - second.get_temperature()) >
          1.e-10 * first.get_temperature())
    throw std::invalid_argument(
        "Fugacity references must have the same pressure and temperature.");
}
// Rational coefficients with bounded denominators make primitive integer
// reaction vectors independent of the arbitrary nullspace normalization.
long long denominator(double value) {
  const double original = std::abs(value);
  double remainder = original;
  long long p0 = 0, p1 = 1, q0 = 1, q1 = 0;
  for (int i = 0; i < 40; ++i) {
    const double integer = std::floor(remainder);
    if (integer > 1.e9)
      break;
    const auto a = static_cast<long long>(integer);
    if (q1 && a > (1000000 - q0) / q1)
      break;
    if (p1 && a > (std::numeric_limits<long long>::max() - p0) / p1)
      break;
    const long long p = a * p1 + p0, q = a * q1 + q0;
    p0 = p1;
    p1 = p;
    q0 = q1;
    q1 = q;
    if (std::abs(static_cast<double>(p1) / static_cast<double>(q1) - original) <
        1.e-10)
      return q1;
    const double fraction = remainder - integer;
    if (fraction == 0.)
      return q1;
    remainder = 1. / fraction;
  }
  if (!q1)
    throw std::invalid_argument(
        "Reaction coefficients exceed the supported rational range.");
  return q1;
}
Eigen::VectorXd integer_reaction(const Eigen::VectorXd &values) {
  long long scale = 1;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    const auto q = denominator(values[i]);
    const auto factor = q / std::gcd(scale, q);
    if (scale > 1000000000000LL / factor)
      throw std::invalid_argument(
          "Reaction coefficients require excessively large integers.");
    scale *= factor;
  }
  std::vector<long long> integers(static_cast<std::size_t>(values.size()));
  long long divisor = 0;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    const double value = values[i] * static_cast<double>(scale);
    if (!std::isfinite(value) || std::abs(value) > 1.e15)
      throw std::invalid_argument(
          "Reaction coefficients exceed integer precision.");
    const auto rounded = std::llround(value);
    integers[static_cast<std::size_t>(i)] = rounded;
    divisor = std::gcd(divisor, rounded);
  }
  if (!divisor)
    throw std::runtime_error("Unable to normalize a reaction.");
  Eigen::VectorXd result(values.size());
  for (Eigen::Index i = 0; i < values.size(); ++i)
    result[i] =
        static_cast<double>(integers[static_cast<std::size_t>(i)] / divisor);
  return result;
}
} // namespace

double fugacity(const Material &standard_material, Assemblage &assemblage) {
  check_same_state(assemblage, standard_material);
  const double mu =
      assemblage.chemical_potential({standard_material.get_formula()})[0];
  return std::exp(
      (mu - standard_material.get_molar_gibbs()) /
      (constants::physics::gas_constant * assemblage.get_temperature()));
}
double relative_fugacity(const types::FormulaMap &formula,
                         Assemblage &assemblage, Assemblage &reference) {
  check_same_state(assemblage, reference);
  const double difference = assemblage.chemical_potential({formula})[0] -
                            reference.chemical_potential({formula})[0];
  return std::exp(difference / (constants::physics::gas_constant *
                                assemblage.get_temperature()));
}
double equilibrium_pressure(const Materials &minerals,
                            const Eigen::VectorXd &amounts, double temperature,
                            double initial) {
  check_reaction(minerals, amounts);
  check_state(initial, temperature);
  const Eigen::VectorXd normalized = amounts / amounts.cwiseAbs().maxCoeff();
  const auto function = [&](double pressure) {
    const auto reaction = evaluate(minerals, normalized, pressure, temperature);
    return std::make_pair(reaction.affinity, reaction.volume);
  };
  const double pressure = scalar_root(function, initial, 1.e10, false);
  function(pressure);
  return pressure;
}
double equilibrium_temperature(const Materials &minerals,
                               const Eigen::VectorXd &amounts, double pressure,
                               double initial) {
  check_reaction(minerals, amounts);
  check_state(pressure, initial);
  const Eigen::VectorXd normalized = amounts / amounts.cwiseAbs().maxCoeff();
  const auto function = [&](double temperature) {
    const auto reaction = evaluate(minerals, normalized, pressure, temperature);
    return std::make_pair(reaction.affinity, -reaction.entropy);
  };
  const double temperature = scalar_root(function, initial, 1000., true);
  function(temperature);
  return temperature;
}
Eigen::Vector2d invariant_point(const Materials &first,
                                const Eigen::VectorXd &a,
                                const Materials &second,
                                const Eigen::VectorXd &b,
                                const Eigen::Vector2d &initial) {
  check_reaction(first, a);
  check_reaction(second, b);
  check_state(initial[0], initial[1]);
  const Eigen::VectorXd normalized_a = a / a.cwiseAbs().maxCoeff();
  const Eigen::VectorXd normalized_b = b / b.cwiseAbs().maxCoeff();
  Eigen::Vector2d state = initial;
  const Eigen::Vector2d scaling(1.e9, 1000.);
  auto residual = [&](const Eigen::Vector2d &point, Eigen::Matrix2d *jacobian) {
    const auto r1 = evaluate(first, normalized_a, point[0], point[1]);
    const auto r2 = evaluate(second, normalized_b, point[0], point[1]);
    if (jacobian) {
      *jacobian << r1.volume, -r1.entropy, r2.volume, -r2.entropy;
      *jacobian *= scaling.asDiagonal();
    }
    return Eigen::Vector2d(r1.affinity, r2.affinity);
  };
  for (int iteration = 0; iteration < 150; ++iteration) {
    Eigen::Matrix2d jacobian;
    const Eigen::Vector2d value = residual(state, &jacobian);
    const auto decomposition = jacobian.fullPivLu();
    if (decomposition.rank() != 2)
      throw std::runtime_error(
          "The two reactions do not constrain an isolated invariant point.");
    if (value.cwiseAbs().maxCoeff() < 1.e-6)
      return state;
    Eigen::Vector2d step = decomposition.solve(-value);
    if (step.cwiseAbs().maxCoeff() > 10.)
      step *= 10. / step.cwiseAbs().maxCoeff();
    step.array() *= scaling.array();
    bool accepted = false;
    for (double fraction = 1.; fraction >= 1.e-12; fraction *= .5) {
      const Eigen::Vector2d trial = state + fraction * step;
      if (!trial.allFinite() || trial[1] <= 0.)
        continue;
      try {
        if (residual(trial, nullptr).norm() < value.norm()) {
          state = trial;
          accepted = true;
          break;
        }
      } catch (const std::exception &) {
      }
    }
    if (!accepted)
      break;
  }
  residual(state, nullptr);
  throw std::runtime_error("Invariant-point calculation did not converge.");
}
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
hugoniot(Material &mineral, double P_ref, double T_ref,
         const Eigen::ArrayXd &pressures, Material *reference) {
  check_state(P_ref, T_ref);
  if (!pressures.isFinite().all())
    throw std::invalid_argument("Hugoniot pressures must be finite.");
  if (!reference)
    reference = &mineral;
  reference->set_state(P_ref, T_ref);
  const double U_ref = reference->get_molar_internal_energy();
  const double V_ref = reference->get_molar_volume();
  const double mass_ref = reference->get_molar_mass();
  if (!std::isfinite(mass_ref) || mass_ref <= 0.)
    throw std::invalid_argument(
        "The reference molar mass must be finite and positive.");
  Eigen::ArrayXd temperatures(pressures.size()), volumes(pressures.size());
  double initial = mineral.has_state() ? mineral.get_temperature() : T_ref;
  for (Eigen::Index i = 0; i < pressures.size(); ++i) {
    const double pressure = pressures[i];
    if (auto composite = dynamic_cast<EquilibratedComposite *>(&mineral)) {
      composite->set_state_with_hugoniot(pressure, P_ref, U_ref / mass_ref,
                                         V_ref / mass_ref, initial);
      temperatures[i] = composite->get_temperature();
      volumes[i] = composite->get_molar_volume();
      initial = temperatures[i];
      continue;
    }
    const auto function = [&](double temperature) {
      mineral.set_state(pressure, temperature);
      const double mass = mineral.get_molar_mass();
      if (!std::isfinite(mass) || mass <= 0.)
        throw std::runtime_error(
            "The material molar mass must be finite and positive.");
      const double scale = mass_ref / mass;
      const double V = scale * mineral.get_molar_volume();
      const double value = scale * mineral.get_molar_internal_energy() - U_ref -
                           .5 * (pressure - P_ref) * (V_ref - V);
      const double derivative =
          scale * mineral.get_molar_heat_capacity_p() -
          .5 * (pressure + P_ref) * V * mineral.get_thermal_expansivity();
      return std::make_pair(value, derivative);
    };
    temperatures[i] = scalar_root(function, initial, 2000., true);
    function(temperatures[i]);
    volumes[i] = mineral.get_molar_volume();
    initial = temperatures[i];
  }
  return {temperatures, volumes};
}
Eigen::MatrixXd
reactions_from_stoichiometric_matrix(const Eigen::MatrixXd &matrix) {
  if (matrix.rows() == 0 || matrix.cols() == 0 || !matrix.allFinite())
    throw std::invalid_argument(
        "Supply a finite, nonempty stoichiometric matrix.");
  for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    if (matrix.row(i).isZero(0.))
      throw std::invalid_argument("Compounds cannot have empty compositions.");
  Eigen::MatrixXd stoichiometry = matrix;
  for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
    const double scale = matrix.col(j).cwiseAbs().maxCoeff();
    if (scale > 0.)
      stoichiometry.col(j) /= scale;
  }
  const Eigen::Index rank = stoichiometry.fullPivLu().rank();
  const Eigen::Index n = matrix.rows();
  if (rank == n)
    return Eigen::MatrixXd(0, n);
  std::vector<std::vector<double>> reactions;
  // Circuits are exactly the support-minimal null vectors. Each has at most
  // rank+1 compounds, avoiding enumeration of all 2^n sign orthants.
  for (Eigen::Index size = 2; size <= std::min(n, rank + 1); ++size) {
    std::vector<Eigen::Index> indices;
    std::function<void(Eigen::Index)> select = [&](Eigen::Index start) {
      if (static_cast<Eigen::Index>(indices.size()) == size) {
        const Eigen::MatrixXd selected =
            stoichiometry(indices, Eigen::indexing::all);
        const Eigen::MatrixXd basis = utils::nullspace(selected.transpose());
        if (basis.rows() != 1 || (basis.array().abs() < 1.e-10).any())
          return;
        const Eigen::VectorXd coefficients =
            integer_reaction(basis.row(0).transpose());
        if ((selected.transpose() * coefficients).norm() >
            1.e-9 * std::max(1., selected.norm() * coefficients.norm()))
          throw std::runtime_error("Unable to represent a balanced reaction "
                                   "with rational coefficients.");
        std::vector<double> forward(static_cast<std::size_t>(n), 0.);
        std::vector<double> reverse = forward;
        for (Eigen::Index i = 0; i < size; ++i) {
          const auto j =
              static_cast<std::size_t>(indices[static_cast<std::size_t>(i)]);
          forward[j] = coefficients[i];
          reverse[j] = -coefficients[i];
        }
        reactions.push_back(std::move(forward));
        reactions.push_back(std::move(reverse));
        return;
      }
      const Eigen::Index remaining =
          size - static_cast<Eigen::Index>(indices.size());
      for (Eigen::Index i = start; i <= n - remaining; ++i) {
        indices.push_back(i);
        select(i + 1);
        indices.pop_back();
      }
    };
    select(0);
  }
  std::sort(reactions.begin(), reactions.end());
  reactions.erase(std::unique(reactions.begin(), reactions.end()),
                  reactions.end());
  Eigen::MatrixXd result(static_cast<Eigen::Index>(reactions.size()), n);
  for (std::size_t i = 0; i < reactions.size(); ++i)
    for (Eigen::Index j = 0; j < n; ++j)
      result(static_cast<Eigen::Index>(i), j) =
          reactions[i][static_cast<std::size_t>(j)];
  return result;
}
Eigen::MatrixXd
reactions_from_formulae(const std::vector<types::FormulaMap> &formulae) {
  if (formulae.empty())
    throw std::invalid_argument("Supply at least one compound formula.");
  std::unordered_set<std::string> elements;
  for (const auto &formula : formulae) {
    for (const auto &[element, amount] : formula) {
      if (!std::isfinite(amount) || amount < 0. ||
          constants::chemistry::atomic_masses.count(element) == 0)
        throw std::invalid_argument(
            "Supply known elements with finite, nonnegative amounts.");
      elements.insert(element);
    }
  }
  const auto ordered = utils::sort_element_list_to_IUPAC_order(elements);
  Eigen::MatrixXd matrix =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(formulae.size()),
                            static_cast<Eigen::Index>(ordered.size()));
  for (std::size_t i = 0; i < formulae.size(); ++i)
    for (std::size_t j = 0; j < ordered.size(); ++j) {
      const auto it = formulae[i].find(ordered[j]);
      if (it != formulae[i].end())
        matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            it->second;
    }
  return reactions_from_stoichiometric_matrix(matrix);
}
} // namespace burnman::chemistry
