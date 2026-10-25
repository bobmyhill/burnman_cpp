// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include "burnman/eos/hp.hpp"

namespace burnman::eos {
/** Holland & Powell (1998) thermal Murnaghan EOS.
 * V(T,0)=V_0*[1+a_0*(T-T_0)-20*a_0*(sqrt(T)-sqrt(T_0))],
 * K(T,0)=K_0+dKdT_0*(T-T_0). The reference-pressure Cp polynomial and its
 * integrals use the HP_TMT helpers; Cp is differentiated analytically.
 * T_0 defaults to 298.15 K. As in Python HP98, pressure integrals start at
 * zero and P_0 does not shift the model. H_0 and S_0 default to NaN.
 * a_0 is the volume-law coefficient: alpha(T,0) is V'(T,0)/V(T,0).
 * Requires T>0, positive reference V(T,0), K(T,0), and K(T,0)+Kprime_0*P>0.
 * Kprime_0=0 and 1 are supported. Shear modulus is zero. All units molar SI.
 */
class HP98 : public HP_TMT {
public:
  void validate_parameters(types::MineralParams &params) override;
  double compute_volume(double pressure, double temperature,
                        const types::MineralParams &params) const override;
  double compute_pressure(double temperature, double volume,
                          const types::MineralParams &params) const override;
  double compute_isothermal_bulk_modulus_reuss(
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
  double compute_molar_heat_capacity_p(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
  double compute_molar_internal_energy(
      double pressure, double temperature, double volume,
      const types::MineralParams &params) const override;
};
} // namespace burnman::eos
