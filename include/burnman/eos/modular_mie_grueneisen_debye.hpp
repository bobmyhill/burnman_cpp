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

#ifndef BURNMAN_EOS_MODULAR_MIE_GRUENEISEN_DEBYE_HPP_INCLUDED
#define BURNMAN_EOS_MODULAR_MIE_GRUENEISEN_DEBYE_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"

namespace burnman::eos {
/**
 * Modular Mie-Grueneisen-Debye EOS, matching Python BurnMan.
 *
 * MineralParams selects a reference_eos and debye_temperature_model.
 * Electronic contributions (bel_0, gel) and an additive anharmonic thermal
 * model with a volume prefactor are optional. T_0, V_0, molar_mass and napfu
 * are required, along with the parameters required by each component.
 * F_0 is the reference Helmholtz energy. Shear modulus is not modelled.
 * All units are SI.
 */
class ModularMGD : public EquationOfState {
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

protected:
  struct State {
    double helmholtz, pressure, bulk_modulus, entropy, heat_capacity_v,
        alpha_kt;
  };
  virtual State compute_state(double temperature, double volume,
                              const types::MineralParams &params) const;
};

/** Volume-rescaling anharmonic extension, controlled by c_anh. */
class ModularMGDWithAnharmonicity : public ModularMGD {
public:
  void validate_parameters(types::MineralParams &params) override;

protected:
  State compute_state(double temperature, double volume,
                      const types::MineralParams &params) const override;

private:
  double ln_volume_ratio(double temperature, double volume,
                         const types::MineralParams &params) const;
};

} // namespace burnman::eos
#endif // BURNMAN_EOS_MODULAR_MIE_GRUENEISEN_DEBYE_HPP_INCLUDED
