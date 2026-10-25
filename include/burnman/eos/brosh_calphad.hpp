// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/** Brosh, Makov & Shneck (2007), with the full Appendix A cold curve.
 * Uses Appendix C's continuation below 298.15 K and extrapolates the final
 * SGTE segment. Quantities are molar SI; theta_0 is an Einstein temperature.
 * As in Python BurnMan, the ambient SGTE isobar uses the P=0 approximation.
 * Magnetic contributions are separate Mineral property modifiers.
 */
class BroshCalphad : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;
  double compute_volume(double pressure, double temperature,
                        const types::MineralParams &params) const override;
  double compute_pressure(double temperature, double volume,
                          const types::MineralParams &params) const override;
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_p(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_thermal_expansivity(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_isothermal_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_v(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_isentropic_bulk_modulus_reuss(
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
  double compute_enthalpy(double pressure, double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
};
} // namespace burnman::eos
