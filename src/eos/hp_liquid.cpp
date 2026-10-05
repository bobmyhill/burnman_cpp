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

/* HP2011: constant reference-pressure expansivity and linear K(T). */
#include "burnman/eos/hp_liquid.hpp"
#include "burnman/eos/modified_tait.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <stdexcept>
namespace burnman::eos {
void HP_TMTL::validate_parameters(types::MineralParams &m) {
  utils::require_set(m.H_0, "H_0");
  utils::require_set(m.S_0, "S_0");
  utils::require_set(m.Cp, "Cp");
  utils::require_set(m.a_0, "a_0");
  utils::require_set(m.dKdT_0, "dKdT_0");
  utils::require_set(m.napfu, "napfu");
  utils::require_set(m.molar_mass, "molar_mass");
  utils::fallback_to_default(m.T_0, 298.15);
  MT().validate_parameters(m);
}
types::MineralParams HP_TMTL::thermal(double t,
                                      const types::MineralParams &m) const {
  if (!std::isfinite(t) || t <= 0)
    throw std::invalid_argument(
        "Liquid EOS requires positive finite temperature.");
  auto out = m;
  out.V_0 = *m.V_0 * std::exp(*m.a_0 * (t - *m.T_0));
  out.K_0 = *m.K_0 + *m.dKdT_0 * (t - *m.T_0);
  if (*out.K_0 <= 0)
    throw std::invalid_argument(
        "Liquid bulk modulus is nonpositive at this temperature.");
  return out;
}
double HP_TMTL::compute_volume(double p, double t,
                               const types::MineralParams &m) const {
  return MT::compute_modified_tait_volume(p, thermal(t, m));
}
double HP_TMTL::compute_pressure(double t, double v,
                                 const types::MineralParams &m) const {
  auto tmp = thermal(t, m);
  return MT::compute_modified_tait_pressure(v / *tmp.V_0, tmp);
}
double HP_TMTL::compute_gibbs_free_energy(double p, double t, double,
                                          const types::MineralParams &m) const {
  auto tmp = thermal(t, m);
  auto [a, b, c] = MT::compute_tait_constants(tmp);
  double dp = p - *tmp.P_0;
  double integral =
      *tmp.V_0 * (a / (b * (1 - c)) * std::expm1((1 - c) * std::log1p(b * dp)) +
                  (1 - a) * dp);
  return *m.H_0 + compute_intCpdT(t, m) -
         t * (*m.S_0 + compute_intCpoverTdT(t, m)) + integral;
}
double HP_TMTL::compute_entropy(double p, double t, double v,
                                const types::MineralParams &m) const {
  return (compute_gibbs_free_energy(p, t - .05, v, m) -
          compute_gibbs_free_energy(p, t + .05, v, m)) /
         .1;
}
double
HP_TMTL::compute_molar_heat_capacity_p(double p, double t, double v,
                                       const types::MineralParams &m) const {
  return t *
         (compute_entropy(p, t + .05, v, m) -
          compute_entropy(p, t - .05, v, m)) /
         .1;
}
double HP_TMTL::compute_isothermal_bulk_modulus_reuss(
    double p, double t, double, const types::MineralParams &m) const {
  return MT::compute_modified_tait_bulk_modulus(p, thermal(t, m));
}
double
HP_TMTL::compute_thermal_expansivity(double p, double t, double v,
                                     const types::MineralParams &m) const {
  return (compute_volume(p, t + .05, m) - compute_volume(p, t - .05, m)) /
         (.1 * v);
}
} // namespace burnman::eos
