// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/** Pitzer-Sterner (1994/1995) fluid EOS, in molar SI units.
 * c_coeffs is the paper's 10x6 coefficient table (density in mol/cm^3).
 * Cp selects an ideal-gas reference with H_0, S_0, T_0 and P_0. Otherwise
 * F_0, Cv_0, Debye_0 and Debye_n specify Python BurnMan's Debye reference.
 * P_max/T_min/T_max optionally restrict the state domain. Volume inversion
 * selects the lowest-Gibbs root with positive isothermal bulk modulus.
 * Thermodynamic derivatives are evaluated at the supplied volume.
 */
class PitzerSterner : public EquationOfState {
public:
  PitzerSterner();
  /// An alternate gas constant supports the original PS water convention.
  explicit PitzerSterner(double gas_constant);
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
  double compute_thermal_expansivity(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_v(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_p(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_grueneisen_parameter(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_shear_modulus(double pressure, double temperature, double volume,
                        const types::MineralParams &params) const override;
  double compute_helmholtz_free_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;
  double compute_enthalpy(double pressure, double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;

protected:
  struct ThermalState {
    double helmholtz, entropy, cv;
  };
  virtual void validate_thermal_parameters(types::MineralParams &params);
  virtual ThermalState thermal_state(double temperature,
                                     const types::MineralParams &params) const;

private:
  double gas_constant_;
  struct State {
    double helmholtz, bulk, entropy, cv, dpdt;
  };
  State state(double temperature, double volume,
              const types::MineralParams &params) const;
};
} // namespace burnman::eos
