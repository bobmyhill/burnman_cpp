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

#ifndef BURNMAN_EOS_ANHARMONIC_THERMAL_MODELS_HPP_INCLUDED
#define BURNMAN_EOS_ANHARMONIC_THERMAL_MODELS_HPP_INCLUDED

#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::eos::anharmonic_thermal_models {
/** F_a = theta f(T/theta), with derivatives before multiplying by A(V/V_0). */
struct ThermalProperties {
  double helmholtz, entropy, heat_capacity, dhelmholtz_dTheta,
      d2helmholtz_dTheta2, dentropy_dTheta;
};
class AnharmonicThermalModelBase {
public:
  virtual ~AnharmonicThermalModelBase() = default;
  virtual void validate_parameters(types::MineralParams &params) const = 0;
  virtual ThermalProperties
  evaluate(double temperature, double theta,
           const types::MineralParams &params) const = 0;
};
class Pade : public AnharmonicThermalModelBase {
public:
  void validate_parameters(types::MineralParams &params) const override;
  ThermalProperties evaluate(double temperature, double theta,
                             const types::MineralParams &params) const override;
};
class LogNormal : public AnharmonicThermalModelBase {
public:
  void validate_parameters(types::MineralParams &params) const override;
  ThermalProperties evaluate(double temperature, double theta,
                             const types::MineralParams &params) const override;
};
} // namespace burnman::eos::anharmonic_thermal_models

#endif // BURNMAN_EOS_ANHARMONIC_THERMAL_MODELS_HPP_INCLUDED
