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
 * Finite-strain solid model with constant isochoric heat capacity.
 * Power-law Gruneisen thermal relation from de Koker's original Fortran.
 * All quantities are SI. Evaluation requires T > 0 and V > 0.
 * dks_s_power_law_grueneisen defaults to true. Set false to recover the
 * later Python finite-strain thermal law and its optional shear treatment.
 */
class DKS_S : public EquationOfState {
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
  static double integrated_grueneisen(double volume,
                                      const types::MineralParams &params);
  static double volume_dependent_q(double volume,
                                   const types::MineralParams &params);
};
} // namespace burnman::eos
