// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#pragma once

#include "burnman/core/equation_of_state.hpp"
#include "burnman/optim/roots/brent.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burnman::eos::dks_detail {
inline void check_state(double temperature, double volume) {
  if (!(std::isfinite(temperature) && temperature > 0. &&
        std::isfinite(volume) && volume > 0.))
    throw std::domain_error("DKS requires finite T > 0 and V > 0.");
}
struct VolumeParams {
  const EquationOfState &eos;
  const types::MineralParams &params;
  double pressure, temperature;
};
inline double residual(double ratio, void *data) {
  const auto &p = *static_cast<const VolumeParams *>(data);
  return p.eos.compute_pressure(p.temperature, ratio * *p.params.V_0,
                                p.params) -
         p.pressure;
}
// As in Python, bracket about V_0 before using Brent. Work in relative volume
// to retain accuracy independently of the formula-unit size, and avoid
// stepping through V <= 0 or onto an unstable expanded branch.
inline double volume(const EquationOfState &eos, double pressure,
                     double temperature, const types::MineralParams &params) {
  check_state(temperature, *params.V_0);
  if (!std::isfinite(pressure))
    throw std::invalid_argument("DKS requires finite pressure.");
  VolumeParams p{eos, params, pressure, temperature};
  double x = 1.;
  double fx = residual(x, &p);
  if (fx == 0.)
    return *params.V_0;
  const bool expand = fx > 0.;
  for (int i = 0; i < 1000; ++i) {
    const double next = x * (expand ? 1.01 : 1. / 1.01);
    const double fn = residual(next, &p);
    const double kt = eos.compute_isothermal_bulk_modulus_reuss(
        pressure, temperature, next * *params.V_0, params);
    if (!(std::isfinite(fn) && std::isfinite(kt) && kt > 0.))
      break;
    if (fn == 0.)
      return next * *params.V_0;
    if (std::signbit(fn) != std::signbit(fx))
      return *params.V_0 * optim::roots::brent(&residual, p, std::min(x, next),
                                               std::max(x, next), 1.e-14);
    x = next;
    fx = fn;
  }
  throw std::domain_error(
      "Cannot bracket a DKS volume within the stable EOS domain.");
}
} // namespace burnman::eos::dks_detail
