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

#pragma once
#include "burnman/eos/hp.hpp"
namespace burnman::eos {
class HP_TMTL : public HP_TMT {
public:
  void validate_parameters(types::MineralParams &) override;
  double compute_volume(double, double,
                        const types::MineralParams &) const override;
  double compute_pressure(double, double,
                          const types::MineralParams &) const override;
  double compute_gibbs_free_energy(double, double, double,
                                   const types::MineralParams &) const override;
  double compute_entropy(double, double, double,
                         const types::MineralParams &) const override;
  double
  compute_molar_heat_capacity_p(double, double, double,
                                const types::MineralParams &) const override;
  double compute_isothermal_bulk_modulus_reuss(
      double, double, double, const types::MineralParams &) const override;
  double
  compute_thermal_expansivity(double, double, double,
                              const types::MineralParams &) const override;

private:
  types::MineralParams thermal(double, const types::MineralParams &) const;
};
} // namespace burnman::eos
