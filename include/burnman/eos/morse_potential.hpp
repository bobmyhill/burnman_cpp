// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/** Isothermal Morse-potential EOS (Stacey 1981).
 * Requires V_0, K_0 and Kprime_0; F_0 and P_0 default to zero. E_0 is not
 * accepted, following Python Morse and native SPOCK. Thermal properties are
 * placeholders: S=alpha=gamma=0, Cp=Cv=1e-99; shear modulus is zero.
 * The Kprime_0=1 limit is evaluated analytically. Helmholtz energy includes
 * P_0*(V_0-V), so -dF/dV equals the full pressure, including P_0.
 * Volume inversion follows the stable branch connected to V_0. All units SI.
 */
class Morse : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;

  /// Pressure [Pa] at relative volume compression = V/V_0.
  static double compute_morse(double compression,
                              const types::MineralParams &params);
  /// Bulk modulus [Pa] at molar volume [m^3/mol].
  static double compute_morse_bulk_modulus(double volume,
                                           const types::MineralParams &params);

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
