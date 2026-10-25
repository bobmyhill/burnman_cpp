// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/macaw.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <algorithm>
#include <cmath>
#include <gsl/gsl_poly.h>
#include <limits>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
struct Shape {
  double a, b, c;
};
Shape make_params(const Params &p) {
  const double b = *p.Kprime_inf - 1., delta = *p.Kprime_0 - *p.Kprime_inf;
  double c0, c1, c2;
  // Equivalent to Python's Cardano formula, without a negative square root
  // in the three-real-root case or division by a vanishing cube root.
  const int roots = gsl_poly_solve_cubic(
      2. * b - 3.5 - delta, (b - .5) * (b - 1.5) - (2. * b - .5) * delta,
      -delta * b * (b + 1.), &c0, &c1, &c2);
  double c = roots == 3 ? c2 : c0;
  if (delta == 0. &&
      std::abs(c) < 32. * std::numeric_limits<double>::epsilon() * (1. + b))
    c = 0.; // Preserve the exact constant-Kprime limit when it is this branch.
  const double a = *p.K_0 / (b - .5 * c + (b + c) * (b + c));
  if (!std::isfinite(a) || a <= 0. || !std::isfinite(c))
    throw std::invalid_argument(
        "MACAW parameters do not define a positive reference stiffness.");
  return {a, b, c};
}
double log_volume(double volume, const Params &p) {
  if (!std::isfinite(volume) || volume <= 0.)
    throw std::invalid_argument("MACAW requires positive finite volume.");
  const double strain = (volume - *p.V_0) / *p.V_0;
  return std::abs(strain) < .5 ? std::log1p(strain) : std::log(volume / *p.V_0);
}
struct Strain {
  double w, exponent;
};
Strain strain(double u, const Shape &s) {
  const double w = std::expm1(1.5 * u);
  return {w, -(s.b + 1.) * u - (2. / 3.) * s.c * w};
}
double pressure_at(double u, const Shape &s, double p0) {
  const auto z = strain(u, s);
  return p0 + s.a * ((s.b + s.c) * std::expm1(z.exponent) +
                     s.c * z.w * std::exp(z.exponent));
}
double bulk_at(double u, const Shape &s) {
  const auto z = strain(u, s);
  const double c = s.c * (1. + z.w);
  return s.a * std::exp(z.exponent) * ((s.b + c) * (s.b + c) + s.b - .5 * c);
}
struct Root {
  Shape shape;
  double pressure, p0, k0;
};
double pressure_residual(double u, void *data) {
  const auto &state = *static_cast<Root *>(data);
  return (pressure_at(u, state.shape, state.p0) - state.pressure) / state.k0;
}
double bulk_residual(double u, void *data) {
  const auto &state = *static_cast<Root *>(data);
  return bulk_at(u, state.shape) / state.k0;
}
double exp_remainder(double x) {
  if (std::abs(x) > .01)
    return std::expm1(x) - x;
  double term = .5 * x * x, result = term;
  for (int order = 3; order <= 8; ++order) {
    term *= x / order;
    result += term;
  }
  return result;
}
} // namespace

void MACAW::validate_parameters(Params &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.K_0, "K_0");
  utils::require_set(p.Kprime_0, "Kprime_0");
  utils::require_set(p.Kprime_inf, "Kprime_inf");
  utils::fallback_to_default(p.F_0, 0.);
  utils::fallback_to_default(p.P_0, 1.e5);
  if (p.E_0)
    throw std::invalid_argument(
        "MACAW uses Helmholtz free energy F_0, not E_0.");
  if (!std::isfinite(*p.V_0) || *p.V_0 <= 0. || !std::isfinite(*p.K_0) ||
      *p.K_0 <= 0. || !std::isfinite(*p.Kprime_0) ||
      !std::isfinite(*p.Kprime_inf) || *p.Kprime_inf <= 1. ||
      !std::isfinite(*p.P_0))
    throw std::invalid_argument("MACAW requires finite reference parameters, "
                                "V_0,K_0 > 0 and Kprime_inf > 1.");
  (void)make_params(p);
  utils::check_in_range(p.P_0, 0., 1.e100, "P_0");
  utils::check_in_range(p.V_0, 1.e-7, 1.e-3, "V_0");
  utils::check_in_range(p.K_0, 1.e9, 1.e13, "K_0");
  utils::check_in_range(p.Kprime_0, 0., 10., "Kprime_0");
  utils::check_in_range(p.Kprime_inf, 1. + 45. / 29., *p.Kprime_0,
                        "Kprime_inf");
}
double MACAW::compute_volume(double pressure, double, const Params &p) const {
  if (!std::isfinite(pressure))
    throw std::invalid_argument("MACAW requires finite pressure.");
  if (pressure == *p.P_0)
    return *p.V_0;
  const auto s = make_params(p);
  if (s.c >= 0. && pressure <= *p.P_0 - s.a * (s.b + s.c))
    throw std::invalid_argument("MACAW pressure is below its tension limit.");
  // Prepare the cubic coefficients once for the entire volume inversion.
  Root state{s, pressure, *p.P_0, *p.K_0};
  const bool compressing = pressure > *p.P_0;
  double previous = 0.;
  for (int step = 0; step < 200; ++step) {
    double endpoint = previous + (compressing ? -.1 : .1);
    const double bulk = bulk_residual(endpoint, &state);
    if (!std::isfinite(bulk))
      break;
    if (bulk <= 0.)
      endpoint = optim::roots::brent(&bulk_residual, state,
                                     std::min(previous, endpoint),
                                     std::max(previous, endpoint), 0.);
    const double residual = pressure_residual(endpoint, &state);
    if (!std::isfinite(residual))
      break;
    if ((compressing && residual >= 0.) || (!compressing && residual <= 0.))
      return *p.V_0 * std::exp(optim::roots::brent(&pressure_residual, state,
                                                   std::min(0., endpoint),
                                                   std::max(0., endpoint), 0.));
    if (bulk <= 0.)
      break;
    previous = endpoint;
  }
  throw std::invalid_argument("Cannot bracket a stable MACAW volume.");
}
double MACAW::compute_pressure(double, double volume, const Params &p) const {
  return pressure_at(log_volume(volume, p), make_params(p), *p.P_0);
}
double MACAW::compute_isothermal_bulk_modulus_reuss(double, double,
                                                    double volume,
                                                    const Params &p) const {
  return bulk_at(log_volume(volume, p), make_params(p));
}
double MACAW::compute_isentropic_bulk_modulus_reuss(double pressure,
                                                    double temperature,
                                                    double volume,
                                                    const Params &p) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               p);
}
double MACAW::compute_helmholtz_free_energy(double, double, double volume,
                                            const Params &p) const {
  const auto s = make_params(p);
  const double u = log_volume(volume, p), w = std::expm1(1.5 * u);
  const double exponent = -s.b * u - (2. / 3.) * s.c * w;
  double energy;
  if (std::abs(u) < .01 && std::abs(exponent) < .05)
    energy = exp_remainder(exponent) + (s.b + s.c) * exp_remainder(u) -
             (2. / 3.) * s.c * exp_remainder(1.5 * u);
  else
    energy = std::expm1(exponent) + (s.b + s.c) * std::expm1(u);
  return *p.F_0 + s.a * *p.V_0 * energy + *p.P_0 * (*p.V_0 - volume);
}
double MACAW::compute_gibbs_free_energy(double pressure, double temperature,
                                        double volume, const Params &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p) +
         pressure * volume;
}
double MACAW::compute_molar_internal_energy(double pressure, double temperature,
                                            double volume,
                                            const Params &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p);
}
double MACAW::compute_enthalpy(double pressure, double temperature,
                               double volume, const Params &p) const {
  return compute_gibbs_free_energy(pressure, temperature, volume, p);
}
// Isothermal placeholders, matching Python MACAW and native SPOCK.
double MACAW::compute_grueneisen_parameter(double, double, double,
                                           const Params &) const {
  return 0.;
}
double MACAW::compute_shear_modulus(double, double, double,
                                    const Params &) const {
  return 1.e99;
}
double MACAW::compute_molar_heat_capacity_v(double, double, double,
                                            const Params &) const {
  return 1.e-99;
}
double MACAW::compute_molar_heat_capacity_p(double, double, double,
                                            const Params &) const {
  return 1.e-99;
}
double MACAW::compute_thermal_expansivity(double, double, double,
                                          const Params &) const {
  return 0.;
}
double MACAW::compute_entropy(double, double, double, const Params &) const {
  return 0.;
}
} // namespace burnman::eos
