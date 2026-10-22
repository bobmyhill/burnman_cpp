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

#ifndef BURNMAN_EOS_DEBYE_TEMPERATURE_MODELS_HPP_INCLUDED
#define BURNMAN_EOS_DEBYE_TEMPERATURE_MODELS_HPP_INCLUDED

#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::eos::debye_temperature_models {
/** Debye temperature and its relative-volume derivatives. All units are SI. */
class DebyeTemperatureModelBase {
public:
  virtual ~DebyeTemperatureModelBase() = default;
  virtual void validate_parameters(types::MineralParams &params) const = 0;
  virtual double value(double Vrel,
                       const types::MineralParams &params) const = 0;
  virtual double gamma(double Vrel,
                       const types::MineralParams &params) const = 0;
  virtual double gamma_prime(double Vrel,
                             const types::MineralParams &params) const = 0;
  virtual double dVrel(double Vrel, const types::MineralParams &params) const;
  virtual double dVrel2(double Vrel, const types::MineralParams &params) const;
};
class SLB : public DebyeTemperatureModelBase {
public:
  void validate_parameters(types::MineralParams &params) const override;
  double value(double Vrel, const types::MineralParams &params) const override;
  double gamma(double Vrel, const types::MineralParams &params) const override;
  double gamma_prime(double Vrel,
                     const types::MineralParams &params) const override;
};
class PowerLawGammaSimple : public DebyeTemperatureModelBase {
public:
  void validate_parameters(types::MineralParams &params) const override;
  double value(double Vrel, const types::MineralParams &params) const override;
  double gamma(double Vrel, const types::MineralParams &params) const override;
  double gamma_prime(double Vrel,
                     const types::MineralParams &params) const override;
};
class PowerLawGamma : public DebyeTemperatureModelBase {
public:
  void validate_parameters(types::MineralParams &params) const override;
  double value(double Vrel, const types::MineralParams &params) const override;
  double gamma(double Vrel, const types::MineralParams &params) const override;
  double gamma_prime(double Vrel,
                     const types::MineralParams &params) const override;
};
} // namespace burnman::eos::debye_temperature_models

#endif // BURNMAN_EOS_DEBYE_TEMPERATURE_MODELS_HPP_INCLUDED
