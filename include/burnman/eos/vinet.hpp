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

#ifndef BURNMAN_EOS_VINET_HPP_INCLUDED
#define BURNMAN_EOS_VINET_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"
#include "burnman/utils/types/mineral_params.hpp"

namespace burnman {
namespace eos {

/**
 * @class Vinet
 * @brief Base class for the isothermal Vinet equation of state.
 *
 * References for this equation of state are Vinet (1986) and Vinet (1987).
 * This equation of state actually predates Vinet by 55 years (Rydberg, 1932)
 * and was investigated further by Stacey (1981).
 * Also called the Rose-Vinet, Vinet-Rydberg and Morse-Rydberg EOS.
 *
 * @note All functions assume SI units for all properties.
 */
class Vinet : public EquationOfState {
public:
  // Helper functions
  void validate_parameters(types::MineralParams &params) override;

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

  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;

  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;

private:
  /**
   * @brief Evaluate the Vinet EOS pressure.
   *
   * @param compression V/V_0.
   * @param params Mineral parameters object of type types::MineralParams
   *
   * @return Pressure in [Pa].
   */
  static double compute_vinet(double compression,
                              const types::MineralParams &params);

  /**
   * @brief GSL function wrapper to compute P(V) - P
   *
   * @param x Volume to test (passed by solver)
   * @param p Generic pointer for parameter object
   * @see `eos::gsl_params::SolverParams_P`
   */
  static double vinet_gsl_wrapper(double x, void *p);
};

} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_VINET_HPP_INCLUDED