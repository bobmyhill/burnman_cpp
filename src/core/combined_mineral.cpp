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
#include <cmath>
#include <stdexcept>
#include <utility>

namespace burnman {
namespace {
// Putting the combination in an EOS preserves it when SolutionModel copies
// Mineral values. Components are copied for evaluation so a shared EOS never
// shares mutable mineral state between solutions.
class CombinedEOS final : public EquationOfState {
public:
  CombinedEOS(std::vector<Mineral> minerals, Eigen::ArrayXd amounts)
      : minerals_(std::move(minerals)), amounts_(std::move(amounts)) {}
  void validate_parameters(types::MineralParams &) override {}

  template <typename Getter>
  double sum(double p, double t, Getter getter) const {
    double total = 0.0;
    for (Eigen::Index i = 0; i < amounts_.size(); ++i) {
      if (amounts_[i] == 0.0)
        continue;
      auto mineral = minerals_[static_cast<std::size_t>(i)];
      mineral.set_state(p, t);
      total += amounts_[i] * getter(mineral);
    }
    return total;
  }
  double compute_volume(double p, double t,
                        const types::MineralParams &) const override {
    return sum(p, t, [](const Mineral &m) { return m.get_molar_volume(); });
  }
  double
  compute_gibbs_free_energy(double p, double t, double,
                            const types::MineralParams &) const override {
    return sum(p, t, [](const Mineral &m) { return m.get_molar_gibbs(); });
  }
  double compute_entropy(double p, double t, double,
                         const types::MineralParams &) const override {
    return sum(p, t, [](const Mineral &m) { return m.get_molar_entropy(); });
  }
  double
  compute_molar_heat_capacity_p(double p, double t, double,
                                const types::MineralParams &) const override {
    return sum(p, t,
               [](const Mineral &m) { return m.get_molar_heat_capacity_p(); });
  }
  double compute_isothermal_bulk_modulus_reuss(
      double p, double t, double v,
      const types::MineralParams &) const override {
    return v / sum(p, t, [](const Mineral &m) {
             return m.get_molar_volume() /
                    m.get_isothermal_bulk_modulus_reuss();
           });
  }
  double
  compute_thermal_expansivity(double p, double t, double v,
                              const types::MineralParams &) const override {
    return sum(p, t,
               [](const Mineral &m) {
                 return m.get_molar_volume() * m.get_thermal_expansivity();
               }) /
           v;
  }
  double compute_shear_modulus(double p, double t, double,
                               const types::MineralParams &) const override {
    return amounts_.sum() / sum(p, t, [](const Mineral &m) {
             return 1.0 / m.get_shear_modulus();
           });
  }
  double compute_grueneisen_parameter(
      double p, double t, double v,
      const types::MineralParams &params) const override {
    double alpha = compute_thermal_expansivity(p, t, v, params);
    double kt = compute_isothermal_bulk_modulus_reuss(p, t, v, params);
    double cv = compute_molar_heat_capacity_p(p, t, v, params) -
                t * alpha * alpha * kt * v;
    return alpha * kt * v / cv;
  }

private:
  const std::vector<Mineral> minerals_;
  const Eigen::ArrayXd amounts_;
};
} // namespace

Mineral make_combined_mineral(const std::vector<Mineral> &minerals,
                              const Eigen::ArrayXd &amounts,
                              const Eigen::Vector3d &adjustment,
                              const std::string &name) {
  if (minerals.empty() ||
      amounts.size() != static_cast<Eigen::Index>(minerals.size()) ||
      !amounts.isFinite().all()) {
    throw std::invalid_argument("Combined minerals require matching nonempty "
                                "minerals and finite amounts.");
  }
  if (!std::isfinite(adjustment[0]) || !std::isfinite(adjustment[1]) ||
      !std::isfinite(adjustment[2])) {
    throw std::invalid_argument(
        "Energy adjustment must contain finite [delta_E, delta_S, delta_V].");
  }
  types::FormulaMap formula;
  double mass = 0.0;
  for (std::size_t i = 0; i < minerals.size(); ++i) {
    if (!minerals[i].params.formula)
      throw std::invalid_argument(
          "Combined minerals require component formulae.");
    for (const auto &[element, n] : minerals[i].get_formula())
      formula[element] += amounts[static_cast<Eigen::Index>(i)] * n;
    mass +=
        amounts[static_cast<Eigen::Index>(i)] * minerals[i].get_molar_mass();
  }
  for (auto it = formula.begin(); it != formula.end();) {
    if (std::abs(it->second) < 1.0e-12)
      it = formula.erase(it);
    else {
      if (it->second < 0.0 || !std::isfinite(it->second))
        throw std::invalid_argument(
            "Combined formula cannot have negative or nonfinite atom counts.");
      ++it;
    }
  }
  if (!std::isfinite(mass) || mass <= 0.0 || formula.empty()) {
    throw std::invalid_argument("Combined minerals require a positive molar "
                                "mass and a nonempty formula.");
  }
  Mineral mineral;
  mineral.params.formula = formula;
  mineral.params.molar_mass = mass;
  mineral.params.name = name;
  mineral.set_method(std::make_shared<CombinedEOS>(minerals, amounts));
  mineral.set_property_modifier_params({eos::excesses::LinearParams{
      adjustment[2], adjustment[1], adjustment[0]}});
  return mineral;
}
} // namespace burnman
