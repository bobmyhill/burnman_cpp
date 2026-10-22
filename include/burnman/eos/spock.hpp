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

#ifndef BURNMAN_EOS_SPOCK_HPP_INCLUDED
#define BURNMAN_EOS_SPOCK_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"
#include "burnman/utils/types/mineral_params.hpp"

namespace burnman {
namespace eos {

/**
 * @class SPOCK
 * @brief Scaled Power Of Compression K-prime equation of state.
 *
 * The isothermal EOS has K' = a b (V/V_0)^a + c, where c = Kprime_inf.
 * Requires V_0, K_0, Kprime_0, Kdprime_0 and Kprime_inf. As in Python
 * BurnMan, F_0 defaults to zero and P_0 to 1.0e5 Pa; E_0 is not accepted.
 * Thermal and shear properties are placeholders, not physical predictions.
 *
 * @note All properties use SI units.
 */
class SPOCK : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;

  /** @brief Pressure [Pa] at relative volume compression = V/V_0. */
  static double compute_spock(double compression,
                              const types::MineralParams &params);

  /** @brief Bulk modulus [Pa] at molar volume [m^3/mol]. */
  static double compute_spock_bulk_modulus(double volume,
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
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_helmholtz_free_energy(
      double pressure, double temperature, double volume,
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

private:
  // GSL wrapper for P(V) - P, following the BM3 volume solver.
  static double spock_gsl_wrapper(double volume, void *p);
};

} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_SPOCK_HPP_INCLUDED
