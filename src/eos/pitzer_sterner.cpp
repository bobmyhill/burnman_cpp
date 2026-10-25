// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/pitzer_sterner.hpp"
#include "burnman/eos/components/debye.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
// Value and first two temperature derivatives, at fixed molar density.
struct ThermalDerivative {
  double v, d = 0., dd = 0.;
  ThermalDerivative(double value = 0., double first = 0., double second = 0.)
      : v(value), d(first), dd(second) {}
};
ThermalDerivative operator+(ThermalDerivative a, ThermalDerivative b) {
  return {a.v + b.v, a.d + b.d, a.dd + b.dd};
}
ThermalDerivative operator-(ThermalDerivative a, ThermalDerivative b) {
  return {a.v - b.v, a.d - b.d, a.dd - b.dd};
}
ThermalDerivative operator*(ThermalDerivative a, ThermalDerivative b) {
  return {a.v * b.v, a.d * b.v + a.v * b.d,
          a.dd * b.v + 2. * a.d * b.d + a.v * b.dd};
}
ThermalDerivative operator/(ThermalDerivative a, ThermalDerivative b) {
  return a * ThermalDerivative{1. / b.v, -b.d / (b.v * b.v),
                               (2. * b.d * b.d / b.v - b.dd) / (b.v * b.v)};
}
ThermalDerivative expm1(ThermalDerivative a) {
  const double e = std::exp(a.v);
  return {std::expm1(a.v), e * a.d, e * (a.dd + a.d * a.d)};
}
ThermalDerivative exp(ThermalDerivative a) {
  const double e = std::exp(a.v);
  return {e, e * a.d, e * (a.dd + a.d * a.d)};
}
// Integral_0^rho exp(-b*x) dx, including b=0 and small b*rho.
ThermalDerivative exponential_integral(ThermalDerivative b, double rho) {
  const auto x = b * rho;
  if (std::abs(x.v) < 1.e-4)
    return rho *
           (1. + x * (-.5 +
                      x * (1. / 6. + x * (-1. / 24. +
                                          x * (1. / 120. + x * (-1. / 720. +
                                                                x / 5040.))))));
  return (0. - expm1(0. - x)) / b;
}
using Coefficients = std::array<ThermalDerivative, 10>;
Coefficients coefficients(double t, const Params &p) {
  constexpr std::array<int, 6> powers{-4, -2, -1, 0, 1, 2};
  Coefficients out{};
  for (std::size_t j = 0; j < powers.size(); ++j) {
    const double power = powers[j], basis = std::pow(t, power);
    for (std::size_t i = 0; i < out.size(); ++i) {
      const double value = (*p.c_coeffs)[i][j] * basis;
      out[i] =
          out[i] + ThermalDerivative{value, power * value / t,
                                     power * (power - 1.) * value / (t * t)};
    }
  }
  return out;
}
void check_temperature(double t, const Params &p) {
  if (!std::isfinite(t) || t <= 0. || t < p.T_min.value_or(0.) ||
      t > p.T_max.value_or(std::numeric_limits<double>::infinity()))
    throw std::invalid_argument(
        "Temperature outside the PitzerSterner domain.");
}
void check_volume(double t, double v, const Params &p) {
  check_temperature(t, p);
  if (!std::isfinite(v) || v <= 0.)
    throw std::invalid_argument(
        "PitzerSterner requires finite positive volume.");
}
struct Mechanical {
  double pressure, bulk;
};
Mechanical mechanical(double rho, double t, const Coefficients &c, double r) {
  const double d = c[1].v +
                   rho * (c[2].v +
                          rho * (c[3].v + rho * (c[4].v + rho * c[5].v))),
               dr = c[2].v + rho * (2. * c[3].v +
                                    rho * (3. * c[4].v + 4. * rho * c[5].v)),
               drr = 2. * c[3].v + rho * (6. * c[4].v + 12. * rho * c[5].v),
               e1 = c[6].v * std::exp(-c[7].v * rho),
               e2 = c[8].v * std::exp(-c[9].v * rho),
               ar = c[0].v - dr / (d * d) + e1 + e2,
               arr = 2. * dr * dr / (d * d * d) - drr / (d * d) - c[7].v * e1 -
                     c[9].v * e2;
  return {r * t * 1.e6 * rho * (1. + rho * ar),
          r * t * 1.e6 * rho * (1. + 2. * rho * ar + rho * rho * arr)};
}
struct DensitySolve {
  const Coefficients &c;
  double t, p, r;
};
double density_residual(double log_rho, void *data) {
  const auto &s = *static_cast<DensitySolve *>(data);
  return mechanical(std::exp(log_rho), s.t, s.c, s.r).pressure / s.p - 1.;
}
ThermalDerivative residual_helmholtz(double rho, const Coefficients &c) {
  const auto delta = rho * (c[2] + rho * (c[3] + rho * (c[4] + rho * c[5])));
  const auto d = c[1] + delta;
  if (c[1].v == 0. || d.v == 0.)
    throw std::domain_error("Singular PitzerSterner Helmholtz denominator.");
  return c[0] * rho - delta / (c[1] * d) +
         c[6] * exponential_integral(c[7], rho) +
         c[8] * exponential_integral(c[9], rho);
}
void require_finite(const std::optional<double> &value, const char *name) {
  utils::require_set(value, name);
  if (!std::isfinite(*value))
    throw std::invalid_argument(
        std::string("Nonfinite PitzerSterner parameter: ") + name);
}
} // namespace

PitzerSterner::PitzerSterner()
    : PitzerSterner(constants::physics::gas_constant) {}
PitzerSterner::PitzerSterner(double r) : gas_constant_(r) {
  if (!std::isfinite(r) || r <= 0.)
    throw std::invalid_argument(
        "PitzerSterner gas constant must be positive and finite.");
}
void PitzerSterner::validate_parameters(Params &p) {
  utils::require_set(p.c_coeffs, "c_coeffs");
  for (const auto &row : *p.c_coeffs)
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("PitzerSterner c_coeffs must be finite.");
  require_finite(p.napfu, "n");
  require_finite(p.molar_mass, "molar_mass");
  utils::require_set(p.formula, "formula");
  if (*p.napfu <= 0. || *p.molar_mass <= 0. ||
      (p.T_min && (!std::isfinite(*p.T_min) || *p.T_min < 0.)) ||
      (p.T_max && (!std::isfinite(*p.T_max) || *p.T_max <= 0. ||
                   *p.T_max < p.T_min.value_or(0.))) ||
      (p.P_max && (!std::isfinite(*p.P_max) || *p.P_max <= 0.)))
    throw std::invalid_argument(
        "Invalid PitzerSterner material or domain parameters.");
  validate_thermal_parameters(p);
}
void PitzerSterner::validate_thermal_parameters(Params &p) {
  if (p.Cp) {
    utils::fallback_to_default(p.T_0, 298.15);
    utils::fallback_to_default(p.P_0, 1.e5);
    require_finite(p.T_0, "T_0");
    require_finite(p.P_0, "P_0");
    require_finite(p.H_0, "H_0");
    require_finite(p.S_0, "S_0");
    if (*p.T_0 <= 0. || *p.P_0 <= 0.)
      throw std::invalid_argument(
          "PitzerSterner reference T_0 and P_0 must be positive.");
    for (double x : {p.Cp->a, p.Cp->b, p.Cp->c, p.Cp->d})
      if (!std::isfinite(x))
        throw std::invalid_argument(
            "PitzerSterner Cp coefficients must be finite.");
  } else {
    require_finite(p.F_0, "F_0");
    require_finite(p.Cv_0, "Cv_0");
    require_finite(p.debye_0, "Debye_0");
    require_finite(p.Debye_n, "Debye_n");
    if (*p.debye_0 <= 0. || *p.Debye_n < 0. || *p.Cv_0 < 0.)
      throw std::invalid_argument(
          "Invalid PitzerSterner Debye thermal reference.");
  }
}
PitzerSterner::ThermalState
PitzerSterner::thermal_state(double t, const Params &p) const {
  if (!p.Cp)
    return {
        *p.Cv_0 * t * (1. - std::log(t)) +
            debye::compute_helmholtz_free_energy(t, *p.debye_0, *p.Debye_n) +
            *p.F_0,
        *p.Cv_0 * std::log(t) +
            debye::compute_entropy(t, *p.debye_0, *p.Debye_n),
        *p.Cv_0 +
            debye::compute_molar_heat_capacity_v(t, *p.debye_0, *p.Debye_n)};
  const auto cp = *p.Cp;
  const double t0 = *p.T_0,
               h = *p.H_0 + cp.a * (t - t0) + cp.b * (t * t - t0 * t0) / 2. -
                   cp.c * (1. / t - 1. / t0) +
                   2. * cp.d * (std::sqrt(t) - std::sqrt(t0)),
               s = *p.S_0 + cp.a * std::log(t / t0) + cp.b * (t - t0) -
                   cp.c * (1. / (t * t) - 1. / (t0 * t0)) / 2. -
                   2. * cp.d * (1. / std::sqrt(t) - 1. / std::sqrt(t0)),
               c = cp.a + cp.b * t + cp.c / (t * t) + cp.d / std::sqrt(t),
               log_ref = std::log(gas_constant_ * t * 1.e6 / *p.P_0);
  return {h - t * s + gas_constant_ * t * (log_ref - 1.),
          s - gas_constant_ * log_ref, c - gas_constant_};
}
PitzerSterner::State PitzerSterner::state(double t, double v,
                                          const Params &p) const {
  check_volume(t, v, p);
  const double rho = 1.e-6 / v;
  const auto c = coefficients(t, p);
  const auto a = residual_helmholtz(rho, c);
  const auto delta = rho * (c[2] + rho * (c[3] + rho * (c[4] + rho * c[5])));
  const auto d = c[1] + delta;
  const auto dr =
      c[2] + rho * (2. * c[3] + rho * (3. * c[4] + 4. * rho * c[5]));
  const auto ar = c[0] - dr / (d * d) + c[6] * exp((0. - c[7]) * rho) +
                  c[8] * exp((0. - c[9]) * rho);
  const auto m = mechanical(rho, t, c, gas_constant_);
  const auto thermal = thermal_state(t, p);
  return {gas_constant_ * t * (std::log(rho) + a.v) + thermal.helmholtz, m.bulk,
          -gas_constant_ * (std::log(rho) + a.v + t * a.d) + thermal.entropy,
          -gas_constant_ * t * (2. * a.d + t * a.dd) + thermal.cv,
          gas_constant_ * 1.e6 * rho * (1. + rho * (ar.v + t * ar.d))};
}
double PitzerSterner::compute_volume(double pressure, double t,
                                     const Params &p) const {
  check_temperature(t, p);
  if (!std::isfinite(pressure) || pressure <= 0. ||
      pressure > p.P_max.value_or(std::numeric_limits<double>::infinity()))
    throw std::invalid_argument("Pressure outside the PitzerSterner domain.");
  const auto c = coefficients(t, p);
  DensitySolve solve{c, t, pressure, gas_constant_};
  const double low =
      std::max(std::numeric_limits<double>::min(),
               std::min(.01 * pressure / (gas_constant_ * t * 1.e6), 1.e-6));
  const double log_low = std::log(low), log_high = std::log(.16);
  double previous = log_low, residual = density_residual(previous, &solve),
         best_v = 0., best_g = std::numeric_limits<double>::infinity();
  for (int i = 1; i <= 320; ++i) {
    const double current = log_low + (log_high - log_low) * i / 320.,
                 next = density_residual(current, &solve);
    if (std::isfinite(residual) && std::isfinite(next) && residual <= 0. &&
        next >= 0.) {
      const double rho = std::exp(optim::roots::brent(
                       &density_residual, solve, previous, current, 1.e-13)),
                   v = 1.e-6 / rho;
      if (mechanical(rho, t, c, gas_constant_).bulk > 0.) {
        const double g = std::log(rho) + residual_helmholtz(rho, c).v +
                         pressure * v / (gas_constant_ * t);
        if (std::isfinite(g) && g < best_g) {
          best_v = v;
          best_g = g;
        }
      }
    }
    previous = current;
    residual = next;
  }
  if (best_v == 0.)
    throw std::runtime_error("No stable PitzerSterner density root found.");
  return best_v;
}
double PitzerSterner::compute_pressure(double t, double v,
                                       const Params &p) const {
  check_volume(t, v, p);
  return mechanical(1.e-6 / v, t, coefficients(t, p), gas_constant_).pressure;
}
double
PitzerSterner::compute_isothermal_bulk_modulus_reuss(double, double t, double v,
                                                     const Params &p) const {
  check_volume(t, v, p);
  return mechanical(1.e-6 / v, t, coefficients(t, p), gas_constant_).bulk;
}
double PitzerSterner::compute_helmholtz_free_energy(double, double t, double v,
                                                    const Params &p) const {
  return state(t, v, p).helmholtz;
}
double PitzerSterner::compute_entropy(double, double t, double v,
                                      const Params &p) const {
  return state(t, v, p).entropy;
}
double PitzerSterner::compute_molar_heat_capacity_v(double, double t, double v,
                                                    const Params &p) const {
  return state(t, v, p).cv;
}
double PitzerSterner::compute_thermal_expansivity(double, double t, double v,
                                                  const Params &p) const {
  const auto s = state(t, v, p);
  return s.dpdt / s.bulk;
}
double PitzerSterner::compute_molar_heat_capacity_p(double, double t, double v,
                                                    const Params &p) const {
  const auto s = state(t, v, p);
  return s.cv + t * v * s.dpdt * s.dpdt / s.bulk;
}
double
PitzerSterner::compute_isentropic_bulk_modulus_reuss(double, double t, double v,
                                                     const Params &p) const {
  const auto s = state(t, v, p);
  return s.bulk + t * v * s.dpdt * s.dpdt / s.cv;
}
double PitzerSterner::compute_grueneisen_parameter(double, double t, double v,
                                                   const Params &p) const {
  const auto s = state(t, v, p);
  return v * s.dpdt / s.cv;
}
double PitzerSterner::compute_shear_modulus(double, double, double,
                                            const Params &) const {
  return 0.;
}
double PitzerSterner::compute_gibbs_free_energy(double pressure, double t,
                                                double v,
                                                const Params &p) const {
  return state(t, v, p).helmholtz + pressure * v;
}
double PitzerSterner::compute_molar_internal_energy(double, double t, double v,
                                                    const Params &p) const {
  const auto s = state(t, v, p);
  return s.helmholtz + t * s.entropy;
}
double PitzerSterner::compute_enthalpy(double pressure, double t, double v,
                                       const Params &p) const {
  const auto s = state(t, v, p);
  return s.helmholtz + t * s.entropy + pressure * v;
}
} // namespace burnman::eos
