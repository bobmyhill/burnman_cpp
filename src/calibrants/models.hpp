// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/calibrant.hpp"
#include "burnman/eos/birch_murnaghan.hpp"
#include "burnman/eos/components/debye.hpp"
#include "burnman/eos/vinet.hpp"
#include "burnman/utils/constants.hpp"
#include "differentiation.hpp"
#include <cmath>

namespace burnman::calibrants::detail {
using Params = Calibrant::Params;
inline types::MineralParams eos_params(const Params &p) {
  types::MineralParams m;
  m.V_0 = p.at("V_0");
  m.K_0 = p.at("K_0");
  m.Kprime_0 = p.at("Kprime_0");
  m.P_0 = p.at("P_0");
  if (p.count("Kprime_prime_0"))
    m.Kdprime_0 = p.at("Kprime_prime_0");
  return m;
}
inline double cold_bm3(double temperature, double volume, const Params &p) {
  return eos::BM3().compute_pressure(temperature, volume, eos_params(p));
}
inline double cold_bm4(double inverse_compression, const Params &p) {
  return eos::BM4().compute_pressure(0., p.at("V_0") / inverse_compression,
                                     eos_params(p));
}
inline double cold_vinet(double temperature, double volume, const Params &p) {
  return eos::Vinet().compute_pressure(temperature, volume, eos_params(p));
}
inline double thermal_energy(double temperature, double theta, double n) {
  return eos::debye::compute_thermal_energy(temperature, theta, n);
}
inline double thermal_pressure(double temperature, double volume,
                               const Params &p) {
  const double gamma =
      p.at("grueneisen_0") * std::pow(volume / p.at("V_0"), p.at("q_0"));
  const double theta =
      p.at("Debye_0") * std::exp((p.at("grueneisen_0") - gamma) / p.at("q_0"));
  return gamma * thermal_energy(temperature, theta, p.at("n")) / volume;
}
} // namespace burnman::calibrants::detail
