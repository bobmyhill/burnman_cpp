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

#pragma once

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/**
 * Liquid model with ideal-gas, electronic, bonding and magnetic terms.
 * Port of Python BurnMan dks_liquid.py (de Koker et al., 2013).
 * All quantities are SI. Evaluation requires T > 0 and V > 0.
 */
class DKS_L : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;
  double compute_volume(double pressure, double temperature,
                        const types::MineralParams &params) const override;
  double compute_pressure(double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_grueneisen_parameter(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_isothermal_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_isentropic_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_shear_modulus(double pressure, double temperature, double volume,
                        const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_v(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_p(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_thermal_expansivity(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_helmholtz_free_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;
  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_enthalpy(double pressure, double temperature, double volume,
                          const types::MineralParams &params) const override;

private:
  struct State {
    double helmholtz, entropy, pressure, bulk_modulus, heat_capacity_v,
        alpha_kt;
  };
  static State compute_state(double temperature, double volume,
                             const types::MineralParams &params);
};
} // namespace burnman::eos
