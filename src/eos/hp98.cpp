// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/hp98.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
struct Thermal {
  double v, first, second, k;
};
Thermal thermal(double t, const Params &p) {
  if (!std::isfinite(t) || t <= 0.)
    throw std::invalid_argument("HP98 requires positive finite temperature.");
  const double root = std::sqrt(t), root0 = std::sqrt(*p.T_0);
  const double dt = t - *p.T_0, va = *p.V_0 * *p.a_0;
  // Rationalise the square-root difference close to the reference state.
  const double v = *p.V_0 + va * dt * (1. - 20. / (root + root0));
  const double k = *p.K_0 + *p.dKdT_0 * dt;
  if (!std::isfinite(v) || v <= 0. || !std::isfinite(k) || k <= 0.)
    throw std::invalid_argument("HP98 requires positive reference volume and "
                                "bulk modulus at this temperature.");
  return {v, va * (1. - 10. / root), 5. * va / (t * root), k};
}
struct Compression {
  double z, log, factor, bulk;
};
Compression compression(double pressure, const Thermal &s, const Params &p) {
  if (!std::isfinite(pressure))
    throw std::invalid_argument("HP98 requires finite pressure.");
  const double q = *p.Kprime_0, z = pressure / s.k, bulk = s.k + q * pressure;
  if (!std::isfinite(bulk) || bulk <= 0.)
    throw std::invalid_argument(
        "HP98 pressure is outside the positive bulk modulus branch.");
  const double log = q == 0. ? z : std::log1p(q * z) / q;
  return {z, log, std::exp(-log), bulk};
}
struct Integral {
  double value, first, second;
};
Integral pressure_integral(const Thermal &s, const Compression &c,
                           const Params &p) {
  const double q = *p.Kprime_0, w = (q - 1.) * c.log;
  // Integral_0^P (1+Kprime*p/K)^(-1/Kprime) dp. This form retains
  // the removable Kprime=0 and 1 limits and small pressure differences.
  const double f = c.log * (w == 0. ? 1. : std::expm1(w) / w);
  double first;
  if (std::abs(c.z) < .001 && std::abs(q * c.z) < .001) {
    double term = 1.;
    first = 0.;
    for (int order = 1; order <= 8; ++order) {
      term *= -(1. + (order - 1.) * q) * c.z / order;
      first -= order / (order + 1.) * term * c.z;
    }
  } else {
    first = f - c.z * c.factor;
  }
  // The last two values are derivatives with respect to the reference K(T).
  return {s.k * f, first, -c.factor * c.z * c.z / c.bulk};
}
void require_finite(const std::optional<double> &value, const char *name) {
  utils::require_set(value, name);
  if (!std::isfinite(*value))
    throw std::invalid_argument(std::string("Nonfinite HP98 parameter: ") +
                                name);
}
} // namespace

void HP98::validate_parameters(Params &p) {
  require_finite(p.V_0, "V_0");
  require_finite(p.K_0, "K_0");
  require_finite(p.Kprime_0, "Kprime_0");
  require_finite(p.a_0, "a_0");
  require_finite(p.dKdT_0, "dKdT_0");
  require_finite(p.napfu, "napfu");
  require_finite(p.molar_mass, "molar_mass");
  utils::require_set(p.Cp, "Cp");
  utils::fallback_to_default(p.T_0, 298.15);
  utils::fallback_to_default(p.P_0, 0.);
  utils::fallback_to_default(p.F_0, 0.);
  utils::fallback_to_default(p.H_0, std::nan(""));
  utils::fallback_to_default(p.S_0, std::nan(""));
  utils::fallback_to_default(p.G_0, std::nan(""));
  utils::fallback_to_default(p.Gprime_0, std::nan(""));
  if (p.E_0)
    throw std::invalid_argument(
        "HP98 uses reference enthalpy H_0, not internal energy E_0.");
  require_finite(p.T_0, "T_0");
  if (*p.V_0 <= 0. || *p.K_0 <= 0. || *p.T_0 <= 0. || *p.napfu <= 0. ||
      *p.molar_mass <= 0.)
    throw std::invalid_argument(
        "HP98 requires positive reference volume, bulk modulus, temperature, "
        "atom count and molar mass.");
  for (double value : {p.Cp->a, p.Cp->b, p.Cp->c, p.Cp->d})
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "HP98 heat-capacity coefficients must be finite.");
  utils::check_in_range(p.V_0, 1.e-7, 1.e-2, "V_0");
  utils::check_in_range(p.K_0, 1.e9, 1.e13, "K_0");
  utils::check_in_range(p.Kprime_0, -5., 30., "Kprime_0");
  utils::check_in_range(p.a_0, 0., 1.e-3, "a_0");
  utils::check_in_range(p.napfu, 1., 1000., "napfu");
  utils::check_in_range(p.molar_mass, .001, 10., "molar_mass");
  if (compute_molar_heat_capacity_pref(*p.T_0, p) < 0.)
    utils::warn("HP98 heat capacity is negative at T_0.");
  if (compute_molar_heat_capacity_pref(2000., p) < 0.)
    utils::warn("HP98 heat capacity is negative at 2000 K.");
}
double HP98::compute_volume(double pressure, double t, const Params &p) const {
  const auto s = thermal(t, p);
  return s.v * compression(pressure, s, p).factor;
}
double HP98::compute_pressure(double t, double volume, const Params &p) const {
  const auto s = thermal(t, p);
  if (!std::isfinite(volume) || volume <= 0.)
    throw std::invalid_argument("HP98 requires positive finite volume.");
  const double q = *p.Kprime_0, log = std::log(s.v / volume);
  return s.k * (q == 0. ? log : std::expm1(q * log) / q);
}
double HP98::compute_isothermal_bulk_modulus_reuss(double pressure, double t,
                                                   double,
                                                   const Params &p) const {
  return compression(pressure, thermal(t, p), p).bulk;
}
double HP98::compute_thermal_expansivity(double pressure, double t, double,
                                         const Params &p) const {
  const auto s = thermal(t, p);
  const auto c = compression(pressure, s, p);
  return s.first / s.v + *p.dKdT_0 * c.z / c.bulk;
}
double HP98::compute_gibbs_free_energy(double pressure, double t, double,
                                       const Params &p) const {
  const auto s = thermal(t, p);
  const auto integral = pressure_integral(s, compression(pressure, s, p), p);
  return *p.H_0 + compute_intCpdT(t, p) -
         t * (*p.S_0 + compute_intCpoverTdT(t, p)) + s.v * integral.value;
}
double HP98::compute_entropy(double pressure, double t, double,
                             const Params &p) const {
  const auto s = thermal(t, p);
  const auto integral = pressure_integral(s, compression(pressure, s, p), p);
  return *p.S_0 + compute_intCpoverTdT(t, p) - s.first * integral.value -
         s.v * *p.dKdT_0 * integral.first;
}
double HP98::compute_molar_heat_capacity_p(double pressure, double t, double,
                                           const Params &p) const {
  const auto s = thermal(t, p);
  const auto integral = pressure_integral(s, compression(pressure, s, p), p);
  const double dk = *p.dKdT_0;
  return compute_molar_heat_capacity_pref(t, p) -
         t * (s.second * integral.value + 2. * s.first * dk * integral.first +
              s.v * dk * dk * integral.second);
}
double HP98::compute_molar_internal_energy(double pressure, double t,
                                           double volume,
                                           const Params &p) const {
  return compute_enthalpy(pressure, t, volume, p) - pressure * volume;
}
} // namespace burnman::eos
