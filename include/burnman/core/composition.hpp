/*
 * This file is part of burnman_cpp, GPL v3.0 or later.
 * Based on BurnMan's Composition (GPL v2.0 or later).
 */
#pragma once

#include "burnman/utils/types/simple_types.hpp"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace burnman {

/**
 * Chemical composition expressed in formula components (e.g. MgO or MgSiO3).
 * Amounts use kg, mol of components, or mol of atoms; "weight" aliases "mass".
 * Signed amounts are supported for compositional differences. Input and output
 * retain component order. Returned compositions are independent snapshots.
 */
class Composition {
public:
  using ComponentAmounts = std::vector<std::pair<std::string, double>>;

  explicit Composition(const ComponentAmounts &composition_dictionary,
                       const std::string &unit_type = "mass",
                       bool normalize = false);

  ComponentAmounts get_mass_composition() const;
  ComponentAmounts get_weight_composition() const;
  ComponentAmounts get_molar_composition() const;
  ComponentAmounts get_atomic_composition() const;
  ComponentAmounts composition(const std::string &unit_type) const;
  std::vector<std::pair<std::string, types::FormulaMap>>
  get_component_formulae() const;
  const std::vector<std::string> &get_element_list() const;

  void renormalize(const std::string &unit_type,
                   const std::string &normalization_component,
                   double normalization_amount);
  void add_components(const ComponentAmounts &composition_dictionary,
                      const std::string &unit_type);
  /**
   * Express the same elemental inventory in a nonnegative new component basis.
   * Uses native NNLS with a relative elemental residual tolerance of 1e-10.
   * Invalid bases leave the original composition unchanged.
   */
  void change_component_set(const std::vector<std::string> &new_component_list);
  /// Remove components with absolute molar amounts strictly below tol (mol).
  void remove_null_components(double tol = 1.0e-12);

  /** Format without modifying this object. As in Python BurnMan,
   * significant_figures specifies the number of digits after the decimal point.
   */
  std::string
  format(const std::string &unit_type, int significant_figures = 1,
         const std::string &normalization_component = "total",
         std::optional<double> normalization_amount = std::nullopt) const;
  std::string repr() const;

  Composition operator+(const Composition &other) const;
  Composition operator-(const Composition &other) const;
  Composition operator*(double scalar) const;
  Composition operator/(double scalar) const;
  Composition &operator+=(const Composition &other);
  Composition &operator-=(const Composition &other);
  Composition &operator*=(double scalar);
  Composition &operator/=(double scalar);

private:
  ComponentAmounts mass_composition_;
  std::vector<types::FormulaMap> component_formulae_;
  std::vector<double> molar_masses_;
  std::vector<std::string> element_list_;
};

Composition operator*(double scalar, const Composition &composition);

/// Read a whitespace-delimited table: component header ending in "Comment",
/// then amounts and optional comment words. Ignore blank lines and # comments.
std::pair<std::vector<Composition>, std::vector<std::vector<std::string>>>
file_to_composition_list(const std::string &fname, const std::string &unit_type,
                         bool normalize);

} // namespace burnman
