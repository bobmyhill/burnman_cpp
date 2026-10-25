// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/** Isothermal MACAW EOS (Lozano et al. 2022).
 * Requires V_0, K_0, Kprime_0 and Kprime_inf. F_0 defaults to zero, P_0 to
 * 1e5 Pa; E_0 is not accepted. The derived coefficients use the largest real
 * root of the parameter cubic, matching Python's branch where it is real.
 * Volume inversion follows the stable branch connected to V_0. Helmholtz
 * energy includes reference-pressure work and retains small strain energies.
 * Isothermal placeholders match Python: S=alpha=gamma=0, Cp=Cv=1e-99,
 * shear modulus=1e99 Pa. All quantities use molar SI units.
 */
class MACAW : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;
  double compute_volume(double pressure, double temperature,
                        const types::MineralParams &params) const override;
  double compute_pressure(double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_isothermal_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_isentropic_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_helmholtz_free_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_enthalpy(double pressure, double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_grueneisen_parameter(
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
  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;
};
} // namespace burnman::eos
