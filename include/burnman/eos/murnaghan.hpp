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

#ifndef BURNMAN_EOS_MURNAGHAN_HPP_INCLUDED
#define BURNMAN_EOS_MURNAGHAN_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"
#include "burnman/utils/types/mineral_params.hpp"

namespace burnman {
namespace eos {

/**
 * @class Murnaghan
 * @brief Isothermal equation of state with constant dK/dP = Kprime_0.
 *
 * Requires V_0, K_0 and Kprime_0. As in BM3, E_0 and P_0 default to zero.
 * F_0 is also accepted as the reference energy, following BM4.
 * Temperature dependence and shear elasticity are not modelled. Thermal
 * properties use the same placeholders as BM3 and Vinet.
 *
 * @note All functions assume SI units for all properties.
 */
class Murnaghan : public EquationOfState {
public:
  // Helper functions
  void validate_parameters(types::MineralParams &params) override;

  /** @brief Pressure [Pa] at inverse compression V_0/V. */
  static double compute_murnaghan(double inv_compression,
                                  const types::MineralParams &params);

  // Specific EOS functions
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
};

} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_MURNAGHAN_HPP_INCLUDED
