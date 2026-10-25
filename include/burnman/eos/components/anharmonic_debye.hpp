// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::eos {
/** Additive F=A(V/V_0)*[F_a(T,theta(V))-F_a(T_0,theta(V))].
 * The prefactor, characteristic-temperature and thermal models are supplied
 * through MineralParams. Derivatives use molar SI units and physical volume.
 */
class AnharmonicDebye {
public:
  struct State {
    double helmholtz, pressure, bulk_modulus, entropy, heat_capacity_v,
        alpha_kt;
  };
  static void validate_parameters(types::MineralParams &params);
  static State evaluate(double temperature, double volume,
                        const types::MineralParams &params);
  static double helmholtz_energy(double temperature, double volume,
                                 const types::MineralParams &params);
  static double pressure(double temperature, double volume,
                         const types::MineralParams &params);
  static double isothermal_bulk_modulus(double temperature, double volume,
                                        const types::MineralParams &params);
  static double entropy(double temperature, double volume,
                        const types::MineralParams &params);
  static double heat_capacity_v(double temperature, double volume,
                                const types::MineralParams &params);
  static double dSdV(double temperature, double volume,
                     const types::MineralParams &params);
};
} // namespace burnman::eos
