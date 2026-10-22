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

#ifndef BURNMAN_EOS_ANHARMONIC_PREFACTOR_MODELS_HPP_INCLUDED
#define BURNMAN_EOS_ANHARMONIC_PREFACTOR_MODELS_HPP_INCLUDED

#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::eos::anharmonic_prefactor_models {
/** Volume prefactor A(V/V_0) for an additive anharmonic Helmholtz energy. */
class AnharmonicPrefactorModel {
public:
  virtual ~AnharmonicPrefactorModel() = default;
  virtual void validate_parameters(types::MineralParams &params) const = 0;
  virtual double value(double Vrel,
                       const types::MineralParams &params) const = 0;
  virtual double dVrel(double Vrel,
                       const types::MineralParams &params) const = 0;
  virtual double dVrel2(double Vrel,
                        const types::MineralParams &params) const = 0;
};
class PowerLaw : public AnharmonicPrefactorModel {
public:
  void validate_parameters(types::MineralParams &params) const override;
  double value(double Vrel, const types::MineralParams &params) const override;
  double dVrel(double Vrel, const types::MineralParams &params) const override;
  double dVrel2(double Vrel, const types::MineralParams &params) const override;
};
class Sigmoid : public AnharmonicPrefactorModel {
public:
  void validate_parameters(types::MineralParams &params) const override;
  double value(double Vrel, const types::MineralParams &params) const override;
  double dVrel(double Vrel, const types::MineralParams &params) const override;
  double dVrel2(double Vrel, const types::MineralParams &params) const override;
};
} // namespace burnman::eos::anharmonic_prefactor_models

#endif // BURNMAN_EOS_ANHARMONIC_PREFACTOR_MODELS_HPP_INCLUDED
