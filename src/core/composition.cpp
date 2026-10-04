/*
 * This file is part of burnman_cpp, GPL v3.0 or later.
 * Based on BurnMan's Composition (GPL v2.0 or later).
 */
#include "burnman/core/composition.hpp"
#include "burnman/utils/chemistry_utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace burnman {
namespace {
using Amounts = Composition::ComponentAmounts;

bool is_mass(const std::string &unit) {
  return unit == "mass" || unit == "weight";
}

void check_input_unit(const std::string &unit) {
  if (!is_mass(unit) && unit != "molar") {
    throw std::invalid_argument(
        "Input unit_type must be 'mass', 'weight' or 'molar'.");
  }
}

double finite(double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("Composition amounts must be finite "
                                "(arithmetic overflow is not allowed).");
  return value;
}

double normalization_factor(const Amounts &amounts,
                            const std::string &component, double target) {
  finite(target);
  double denominator = 0.0;
  if (component == "total") {
    for (const auto &entry : amounts)
      denominator = finite(denominator + entry.second);
  } else {
    auto it = std::find_if(
        amounts.begin(), amounts.end(),
        [&component](const auto &entry) { return entry.first == component; });
    if (it == amounts.end())
      throw std::invalid_argument("Normalization component is absent: " +
                                  component);
    denominator = it->second;
  }
  if (denominator == 0.0)
    throw std::invalid_argument("Cannot normalize a zero amount.");
  return finite(target / denominator);
}

// Lawson-Hanson active-set NNLS. Column and inventory scaling make the
// tolerances independent of formula size and the absolute bulk amount.
// Rank-revealing QR also permits redundant/underdetermined component sets.
Eigen::VectorXd nonnegative_least_squares(const Eigen::MatrixXd &matrix,
                                          const Eigen::VectorXd &inventory) {
  const Eigen::Index n = matrix.cols();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  if (n == 0 || inventory.size() == 0)
    return x;
  const double bulk_scale = inventory.cwiseAbs().maxCoeff();
  if (bulk_scale == 0.0)
    return x;
  const Eigen::VectorXd b = inventory / bulk_scale;
  Eigen::MatrixXd a = matrix;
  Eigen::VectorXd column_scales(n);
  for (Eigen::Index j = 0; j < n; ++j) {
    column_scales[j] = finite(a.col(j).stableNorm());
    if (column_scales[j] != 0.0)
      a.col(j) /= column_scales[j];
    else
      column_scales[j] = 1.0;
  }

  std::vector<bool> passive(static_cast<std::size_t>(n), false);
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                           static_cast<double>(std::max(a.rows(), n));
  const Eigen::Index max_iterations = 100 * std::max<Eigen::Index>(1, n * n);
  Eigen::Index iterations = 0;
  while (true) {
    const Eigen::VectorXd gradient = a.transpose() * (b - a * x);
    Eigen::Index entering = -1;
    double best = tolerance;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (!passive[static_cast<std::size_t>(j)] && gradient[j] > best) {
        entering = j;
        best = gradient[j];
      }
    }
    if (entering < 0)
      break;
    passive[static_cast<std::size_t>(entering)] = true;
    while (true) {
      if (++iterations > max_iterations)
        throw std::runtime_error("Component basis NNLS did not converge.");
      std::vector<Eigen::Index> indices;
      for (Eigen::Index j = 0; j < n; ++j)
        if (passive[static_cast<std::size_t>(j)])
          indices.push_back(j);
      Eigen::MatrixXd subset(a.rows(),
                             static_cast<Eigen::Index>(indices.size()));
      for (std::size_t j = 0; j < indices.size(); ++j)
        subset.col(static_cast<Eigen::Index>(j)) = a.col(indices[j]);
      const Eigen::VectorXd solution =
          subset.completeOrthogonalDecomposition().solve(b);
      Eigen::VectorXd candidate = Eigen::VectorXd::Zero(n);
      bool positive = true;
      for (std::size_t j = 0; j < indices.size(); ++j) {
        candidate[indices[j]] = solution[static_cast<Eigen::Index>(j)];
        positive = positive && candidate[indices[j]] > 0.0;
      }
      if (positive) {
        x = std::move(candidate);
        break;
      }
      double alpha = 1.0;
      for (const auto j : indices) {
        if (candidate[j] <= 0.0) {
          const double denominator = x[j] - candidate[j];
          alpha =
              std::min(alpha, denominator == 0.0 ? 0.0 : x[j] / denominator);
        }
      }
      x += alpha * (candidate - x);
      for (const auto j : indices) {
        if (x[j] <= tolerance) {
          x[j] = 0.0;
          passive[static_cast<std::size_t>(j)] = false;
        }
      }
    }
  }
  const double residual = (a * x - b).stableNorm();
  if (!std::isfinite(residual) || residual > 1.0e-10 * b.stableNorm()) {
    throw std::invalid_argument(
        "Failed to change component set: no nonnegative representation of the "
        "elemental inventory (relative residual=" +
        std::to_string(residual / b.stableNorm()) + ").");
  }
  for (Eigen::Index j = 0; j < n; ++j)
    x[j] = finite((x[j] / column_scales[j]) * bulk_scale);
  return x;
}
} // namespace

Composition::Composition(const ComponentAmounts &input,
                         const std::string &unit_type, bool normalize) {
  check_input_unit(unit_type);
  std::unordered_set<std::string> components;
  std::unordered_set<std::string> elements;
  const double factor =
      normalize ? normalization_factor(input, "total", 1.0) : 1.0;
  for (const auto &[component, amount] : input) {
    finite(amount);
    if (!components.insert(component).second)
      throw std::invalid_argument("Duplicate composition component: " +
                                  component);
    std::vector<std::string> order;
    auto formula = utils::dictionarize_formula(component, &order);
    const double mass = utils::formula_mass(formula);
    if (mass <= 0.0)
      throw std::invalid_argument(
          "Component must have a positive molar mass: " + component);
    for (const auto &element : order) {
      if (elements.insert(element).second)
        element_list_.push_back(element);
    }
    mass_composition_.emplace_back(
        component,
        finite(finite(amount * factor) * (is_mass(unit_type) ? 1.0 : mass)));
    component_formulae_.push_back(std::move(formula));
    molar_masses_.push_back(mass);
  }
}

Composition::ComponentAmounts Composition::get_mass_composition() const {
  return mass_composition_;
}
Composition::ComponentAmounts Composition::get_weight_composition() const {
  return get_mass_composition();
}

Composition::ComponentAmounts Composition::get_molar_composition() const {
  auto result = mass_composition_;
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i].second = finite(result[i].second / molar_masses_[i]);
  return result;
}

Composition::ComponentAmounts Composition::get_atomic_composition() const {
  const auto moles = get_molar_composition();
  types::FormulaMap atoms;
  for (std::size_t i = 0; i < moles.size(); ++i) {
    for (const auto &[element, count] : component_formulae_[i])
      atoms[element] = finite(atoms[element] + count * moles[i].second);
  }
  ComponentAmounts result;
  for (const auto &element : element_list_)
    result.emplace_back(element, atoms.at(element));
  return result;
}

Composition::ComponentAmounts
Composition::composition(const std::string &unit_type) const {
  if (is_mass(unit_type))
    return get_mass_composition();
  if (unit_type == "molar")
    return get_molar_composition();
  if (unit_type == "atomic")
    return get_atomic_composition();
  throw std::invalid_argument(
      "unit_type must be 'mass', 'weight', 'molar' or 'atomic'.");
}

std::vector<std::pair<std::string, types::FormulaMap>>
Composition::get_component_formulae() const {
  std::vector<std::pair<std::string, types::FormulaMap>> result;
  for (std::size_t i = 0; i < mass_composition_.size(); ++i)
    result.emplace_back(mass_composition_[i].first, component_formulae_[i]);
  return result;
}

const std::vector<std::string> &Composition::get_element_list() const {
  return element_list_;
}

void Composition::renormalize(const std::string &unit_type,
                              const std::string &component, double amount) {
  *this *= normalization_factor(composition(unit_type), component, amount);
}

void Composition::add_components(const ComponentAmounts &input,
                                 const std::string &unit_type) {
  const Composition addition(input, unit_type);
  auto result = mass_composition_;
  for (const auto &entry : addition.mass_composition_) {
    const auto &component = entry.first;
    const double amount = entry.second;
    auto it = std::find_if(
        result.begin(), result.end(),
        [&component](const auto &entry) { return entry.first == component; });
    if (it == result.end())
      result.emplace_back(component, amount);
    else
      it->second = finite(it->second + amount);
  }
  *this = Composition(result);
}

void Composition::change_component_set(const std::vector<std::string> &names) {
  ComponentAmounts empty_basis;
  for (const auto &name : names)
    empty_basis.emplace_back(name, 0.0);
  const Composition basis(empty_basis, "molar");
  auto elements = element_list_;
  for (const auto &element : basis.element_list_) {
    if (std::find(elements.begin(), elements.end(), element) == elements.end())
      elements.push_back(element);
  }
  const auto amounts = get_atomic_composition();
  const types::FormulaMap atoms(amounts.begin(), amounts.end());
  Eigen::VectorXd inventory =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(elements.size()));
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(
      inventory.size(), static_cast<Eigen::Index>(names.size()));
  for (std::size_t i = 0; i < elements.size(); ++i) {
    auto it = atoms.find(elements[i]);
    if (it != atoms.end())
      inventory[static_cast<Eigen::Index>(i)] = it->second;
    for (std::size_t j = 0; j < names.size(); ++j) {
      auto entry = basis.component_formulae_[j].find(elements[i]);
      if (entry != basis.component_formulae_[j].end())
        matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            entry->second;
    }
  }
  if (names.empty() && inventory.cwiseAbs().sum() != 0.0) {
    throw std::invalid_argument("A nonzero composition cannot be represented "
                                "by an empty component set.");
  }
  const auto solution = nonnegative_least_squares(matrix, inventory);
  auto new_amounts = empty_basis;
  for (std::size_t j = 0; j < names.size(); ++j)
    new_amounts[j].second = solution[static_cast<Eigen::Index>(j)];
  *this = Composition(new_amounts, "molar");
}

void Composition::remove_null_components(double tol) {
  if (!std::isfinite(tol) || tol < 0.0)
    throw std::invalid_argument(
        "Null-component tolerance must be finite and nonnegative.");
  const auto moles = get_molar_composition();
  ComponentAmounts remaining;
  for (std::size_t i = 0; i < moles.size(); ++i)
    if (std::abs(moles[i].second) >= tol)
      remaining.push_back(mass_composition_[i]);
  *this = Composition(remaining);
}

std::string Composition::format(const std::string &unit_type, int digits,
                                const std::string &component,
                                std::optional<double> amount) const {
  if (digits < 0)
    throw std::invalid_argument("significant_figures must be nonnegative.");
  auto values = composition(unit_type);
  const double factor =
      amount ? normalization_factor(values, component, *amount) : 1.0;
  std::sort(values.begin(), values.end());
  std::string title = unit_type;
  title[0] =
      static_cast<char>(std::toupper(static_cast<unsigned char>(title[0])));
  std::ostringstream output;
  output << title << " composition\n"
         << std::fixed << std::setprecision(digits);
  for (const auto &[key, value] : values)
    output << key << ": " << finite(value * factor) << '\n';
  return output.str();
}

std::string Composition::repr() const {
  std::ostringstream output;
  output << "Composition (mass):\n" << std::setprecision(17);
  for (const auto &[component, amount] : mass_composition_)
    output << "  " << component << ": " << amount << '\n';
  return output.str();
}

Composition Composition::operator+(const Composition &other) const {
  auto result = *this;
  result += other;
  return result;
}
Composition Composition::operator-(const Composition &other) const {
  auto result = *this;
  result -= other;
  return result;
}
Composition Composition::operator*(double scalar) const {
  auto result = *this;
  result *= scalar;
  return result;
}
Composition Composition::operator/(double scalar) const {
  auto result = *this;
  result /= scalar;
  return result;
}
Composition &Composition::operator+=(const Composition &other) {
  add_components(other.mass_composition_, "mass");
  return *this;
}
Composition &Composition::operator-=(const Composition &other) {
  add_components((other * -1.0).mass_composition_, "mass");
  return *this;
}
Composition &Composition::operator*=(double scalar) {
  finite(scalar);
  auto scaled = mass_composition_;
  for (auto &entry : scaled)
    entry.second = finite(entry.second * scalar);
  mass_composition_ = std::move(scaled);
  return *this;
}
Composition &Composition::operator/=(double scalar) {
  finite(scalar);
  if (scalar == 0.0)
    throw std::invalid_argument("Cannot divide a composition by zero.");
  auto scaled = mass_composition_;
  for (auto &entry : scaled)
    entry.second = finite(entry.second / scalar);
  mass_composition_ = std::move(scaled);
  return *this;
}
Composition operator*(double scalar, const Composition &composition) {
  return composition * scalar;
}

std::pair<std::vector<Composition>, std::vector<std::vector<std::string>>>
file_to_composition_list(const std::string &fname, const std::string &unit_type,
                         bool normalize) {
  check_input_unit(unit_type);
  std::ifstream file(fname);
  if (!file)
    throw std::runtime_error("Cannot open composition file: " + fname);
  std::vector<std::string> components;
  std::vector<Composition> compositions;
  std::vector<std::vector<std::string>> comments;
  std::string line;
  std::size_t line_number = 0;
  bool have_header = false;
  while (std::getline(file, line)) {
    ++line_number;
    std::istringstream tokens(line);
    std::string word;
    if (!(tokens >> word) || word[0] == '#')
      continue;
    if (!have_header) {
      do {
        if (word == "Comment") {
          have_header = true;
          break;
        }
        components.push_back(word);
      } while (tokens >> word);
      if (!have_header || components.empty())
        throw std::invalid_argument("Composition file header must list "
                                    "components followed by 'Comment'.");
      Amounts empty;
      for (const auto &name : components)
        empty.emplace_back(name, 0.0);
      Composition validate_header(empty);
      continue;
    }
    std::istringstream values(line);
    Amounts amounts;
    for (const auto &name : components) {
      double amount = 0.0;
      std::string value;
      std::size_t consumed = 0;
      try {
        if (!(values >> value))
          throw std::invalid_argument("Missing amount.");
        amount = finite(std::stod(value, &consumed));
        if (consumed != value.size())
          throw std::invalid_argument("Invalid number.");
      } catch (const std::exception &) {
        throw std::invalid_argument("Invalid composition amounts at line " +
                                    std::to_string(line_number) + ".");
      }
      amounts.emplace_back(name, amount);
    }
    std::vector<std::string> comment;
    while (values >> word)
      comment.push_back(word);
    compositions.emplace_back(amounts, unit_type, normalize);
    comments.push_back(std::move(comment));
  }
  if (!have_header)
    throw std::invalid_argument("Composition file has no header.");
  return {std::move(compositions), std::move(comments)};
}

} // namespace burnman
