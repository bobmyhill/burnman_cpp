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

#ifndef BURNMAN_EOS_CORK_HPP_INCLUDED
#define BURNMAN_EOS_CORK_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {

/**
 * Compensated Redlich-Kwong fluid equation of state of Holland and Powell
 * (1991). Parameters and returned properties use SI units. The critical
 * temperature and pressure are cork_T and cork_P; Cp holds the four
 * reference-pressure heat-capacity coefficients. The shear modulus is zero.
 */
class CORK : public EquationOfState {
public:
  void validate_parameters(types::MineralParams &params) override;
  double compute_volume(double pressure, double temperature,
                        const types::MineralParams &params) const override;
  double compute_isothermal_bulk_modulus_reuss(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_shear_modulus(double pressure, double temperature, double volume,
                        const types::MineralParams &params) const override;
  double compute_thermal_expansivity(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_heat_capacity_p(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double
  compute_gibbs_free_energy(double pressure, double temperature, double volume,
                            const types::MineralParams &params) const override;
  double compute_entropy(double pressure, double temperature, double volume,
                         const types::MineralParams &params) const override;

private:
  double
  compute_molar_heat_capacity_p0(double temperature,
                                 const types::MineralParams &params) const;
};

} // namespace burnman::eos

#endif // BURNMAN_EOS_CORK_HPP_INCLUDED
