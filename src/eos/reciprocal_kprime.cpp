// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/reciprocal_kprime.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gamma.h>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
struct Shape {
  double a, r, h;
  explicit Shape(const Params &p)
      : a(*p.Kprime_inf), r(*p.Kprime_0 / a), h((*p.Kprime_0 - a) / a) {}
};

// y = -log(1-Kprime_inf*xi), xi=(P-P_0)/K. Unlike xi, y has no finite
// compression pole. Equations 58 and 61 can then be inverted accurately
// even when xi would round to 1/Kprime_inf.
double log_volume(double y, const Shape &s) {
  return -(y + s.h * (y + std::expm1(-y))) / s.a;
}
double pressure_ratio(double y, const Shape &s) {
  return std::exp(s.h * y) * std::expm1(y) / s.a;
}
double lower_y(const Shape &s) { return std::log(s.h / s.r); }
struct Root {
  Shape shape;
  double target;
};
double pressure_residual(double y, void *data) {
  const auto &state = *static_cast<Root *>(data);
  const auto &s = state.shape;
  if (state.target < 0.)
    return pressure_ratio(y, s) - state.target;
  if (y == 0.)
    return -std::log1p(state.target);
  // log(1+(P-P_0)/K_0), without overflowing at a trial compression.
  const double z = s.r * y + std::log(-std::expm1(-y)) - std::log(s.a);
  const double value =
      z > 0. ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z));
  return value - std::log1p(state.target);
}
double volume_residual(double y, void *data) {
  const auto &state = *static_cast<Root *>(data);
  return log_volume(y, state.shape) - state.target;
}
double from_pressure(double pressure, const Params &p) {
  if (!std::isfinite(pressure))
    throw std::invalid_argument("RKprime requires finite pressure.");
  const Shape s(p);
  const double q = (pressure - *p.P_0) / *p.K_0;
  if (q == 0.)
    return 0.;
  if (s.h == 0.) {
    if (q <= -1. / s.a)
      throw std::invalid_argument(
          "RKprime pressure is below its tension limit.");
    return std::log1p(s.a * q);
  }
  Root state{s, q};
  if (q < 0.) {
    const double lower = lower_y(s);
    const double minimum = -std::exp(s.h * lower) / (s.a * s.r);
    if (q < minimum)
      throw std::invalid_argument(
          "RKprime pressure is below its tension limit.");
    if (q == minimum)
      return lower;
    return optim::roots::brent(&pressure_residual, state, lower, 0., 0.);
  }
  double upper = 1.;
  while (pressure_residual(upper, &state) < 0.)
    upper *= 2.;
  return optim::roots::brent(&pressure_residual, state, 0., upper, 0.);
}
double from_volume(double volume, const Params &p) {
  if (!std::isfinite(volume) || volume <= 0.)
    throw std::invalid_argument("RKprime requires positive finite volume.");
  const double ratio = volume / *p.V_0;
  const double target = std::isfinite(ratio) && ratio > 0.
                            ? std::log(ratio)
                            : std::log(volume) - std::log(*p.V_0);
  const Shape s(p);
  if (target == 0.)
    return 0.;
  if (s.h == 0.)
    return -s.a * target;
  Root state{s, target};
  if (target > 0.) {
    const double lower = lower_y(s);
    const double maximum = log_volume(lower, s);
    if (target > maximum)
      throw std::invalid_argument("RKprime volume exceeds its tension limit.");
    if (target == maximum)
      return lower;
    return optim::roots::brent(&volume_residual, state, lower, 0., 0.);
  }
  double upper = 1.;
  while (volume_residual(upper, &state) > 0.)
    upper *= 2.;
  return optim::roots::brent(&volume_residual, state, 0., upper, 0.);
}

struct Integral {
  Shape shape;
  double y;
};
double energy_integrand(double t, void *data) {
  const auto &state = *static_cast<Integral *>(data);
  const auto &s = state.shape;
  const double u = t * state.y, z = -std::expm1(-u);
  return std::exp(s.r * (1. - 1. / s.a) * u + s.h / s.a * z) * z *
         (1. + s.h * z);
}
double local_energy(double y, const Shape &s) {
  // Integrate -P_elastic*dV directly near V_0. This retains the quadratic
  // strain energy without subtracting nearly equal incomplete-gamma terms.
  Integral data{s, y};
  gsl_function function;
  function.function = &energy_integrand;
  function.params = &data;
  double result, error;
  std::size_t evaluations;
  auto *handler = gsl_set_error_handler_off();
  const int status = gsl_integration_qng(&function, 0., 1., 0., 1.e-12, &result,
                                         &error, &evaluations);
  gsl_set_error_handler(handler);
  if (status != GSL_SUCCESS)
    throw std::runtime_error("RKprime strain-energy integration failed.");
  return y * result / (s.a * s.a);
}
double strain_energy(double y, const Shape &s) {
  if (y == 0.)
    return 0.;
  if (std::abs(y) < .05)
    return local_energy(y, s);
  if (s.h == 0.) {
    // Constant Kprime: the Murnaghan compression integral.
    const double v = -y / s.a;
    const double integral =
        s.a == 1. ? v : std::expm1((1. - s.a) * v) / (1. - s.a);
    return (std::expm1(v) - integral) / s.a;
  }
  const double a = s.h / s.a, d = s.r * (1. / s.a - 1.);
  gsl_sf_result first, reference;
  auto *handler = gsl_set_error_handler_off();
  const int status_first = gsl_sf_gamma_inc_e(d, a * std::exp(-y), &first);
  const int status_reference = gsl_sf_gamma_inc_e(d, a, &reference);
  gsl_set_error_handler(handler);
  if (status_first != GSL_SUCCESS || status_reference != GSL_SUCCESS)
    return local_energy(y, s);
  // The non-regularised upper gamma is finite at negative integer orders;
  // no downward recurrence through Gamma(0) is needed.
  const double j = std::exp(a - d * std::log(a)) * (first.val - reference.val);
  const double z = -std::expm1(-y), exponent = -d * y + a * z;
  return s.r * j - std::expm1(exponent) - std::exp(exponent) * z / s.a;
}
} // namespace

void RKprime::validate_parameters(Params &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.K_0, "K_0");
  utils::require_set(p.Kprime_0, "Kprime_0");
  utils::require_set(p.Kprime_inf, "Kprime_inf");
  utils::fallback_to_default(p.F_0, 0.);
  utils::fallback_to_default(p.P_0, 0.);
  utils::fallback_to_default(p.G_0, std::nan(""));
  utils::fallback_to_default(p.Gprime_inf, std::nan(""));
  if (p.E_0)
    throw std::invalid_argument(
        "RKprime uses Helmholtz free energy F_0, not E_0.");
  if (!std::isfinite(*p.V_0) || *p.V_0 <= 0. || !std::isfinite(*p.K_0) ||
      *p.K_0 <= 0. || !std::isfinite(*p.Kprime_0) ||
      !std::isfinite(*p.Kprime_inf) || *p.Kprime_inf <= 0. ||
      *p.Kprime_0 < *p.Kprime_inf || !std::isfinite(*p.P_0))
    throw std::invalid_argument("RKprime requires V_0,K_0 > 0 and Kprime_0 >= "
                                "Kprime_inf > 0, all finite.");
  utils::check_in_range(p.P_0, 0., 1.e100, "P_0");
  utils::check_in_range(p.V_0, 1.e-7, 1.e-3, "V_0");
  utils::check_in_range(p.K_0, 1.e9, 1.e13, "K_0");
  utils::check_in_range(p.Kprime_0, 0., 10., "Kprime_0");
  utils::check_in_range(p.Kprime_inf, 5. / 3., *p.Kprime_0, "Kprime_inf");
}
double RKprime::compute_volume(double pressure, double, const Params &p) const {
  return *p.V_0 * std::exp(log_volume(from_pressure(pressure, p), Shape(p)));
}
double RKprime::compute_pressure(double, double volume, const Params &p) const {
  return *p.P_0 + *p.K_0 * pressure_ratio(from_volume(volume, p), Shape(p));
}
double RKprime::compute_isothermal_bulk_modulus_reuss(double pressure, double,
                                                      double,
                                                      const Params &p) const {
  return *p.K_0 * std::exp(Shape(p).r * from_pressure(pressure, p));
}
double RKprime::compute_isentropic_bulk_modulus_reuss(double pressure,
                                                      double temperature,
                                                      double volume,
                                                      const Params &p) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               p);
}
double RKprime::compute_shear_modulus(double pressure, double temperature,
                                      double volume, const Params &p) const {
  const double ratio = *p.G_0 / *p.K_0;
  return ratio * compute_isothermal_bulk_modulus_reuss(pressure, temperature,
                                                       volume, p) -
         (ratio * *p.Kprime_inf - *p.Gprime_inf) * (pressure - *p.P_0);
}
double RKprime::compute_helmholtz_free_energy(double, double, double volume,
                                              const Params &p) const {
  return *p.F_0 +
         *p.K_0 * *p.V_0 * strain_energy(from_volume(volume, p), Shape(p)) +
         *p.P_0 * (*p.V_0 - volume);
}
double RKprime::compute_gibbs_free_energy(double pressure, double temperature,
                                          double volume,
                                          const Params &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p) +
         pressure * volume;
}
double RKprime::compute_molar_internal_energy(double pressure,
                                              double temperature, double volume,
                                              const Params &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p);
}
double RKprime::compute_enthalpy(double pressure, double temperature,
                                 double volume, const Params &p) const {
  return compute_gibbs_free_energy(pressure, temperature, volume, p);
}
// Isothermal placeholders, as in Python RKprime and native Morse.
double RKprime::compute_grueneisen_parameter(double, double, double,
                                             const Params &) const {
  return 0.;
}
double RKprime::compute_molar_heat_capacity_v(double, double, double,
                                              const Params &) const {
  return 1.e-99;
}
double RKprime::compute_molar_heat_capacity_p(double, double, double,
                                              const Params &) const {
  return 1.e-99;
}
double RKprime::compute_thermal_expansivity(double, double, double,
                                            const Params &) const {
  return 0.;
}
double RKprime::compute_entropy(double, double, double, const Params &) const {
  return 0.;
}
} // namespace burnman::eos
