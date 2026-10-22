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

#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace burnman::eos {
namespace {
void require_finite(const std::optional<double> &value, const char *name) {
  utils::require_set(value, name);
  if (!std::isfinite(*value))
    throw std::invalid_argument(std::string(name) + " must be finite.");
}
void validate_debye(types::MineralParams &p) {
  require_finite(p.debye_0, "debye_0");
  require_finite(p.grueneisen_0, "grueneisen_0");
  if (*p.debye_0 <= 0.)
    throw std::invalid_argument("debye_0 must be positive.");
}
double log_volume(double x) {
  if (!(std::isfinite(x) && x > 0.))
    throw std::domain_error("Relative volume must be positive and finite.");
  return std::log(x);
}
double integrated_power(double q, double log_x) {
  return q == 0. ? log_x : std::expm1(q * log_x) / q;
}
std::array<double, 3> slb_coefficients(double x,
                                       const types::MineralParams &p) {
  const double g = *p.grueneisen_0;
  return {6. * g, -12. * g + 36. * g * g - 18. * *p.q_0 * g,
          .5 * std::expm1(-2. / 3. * log_volume(x))};
}
template <std::size_t N>
std::array<double, 3> polynomial(double x, const std::array<double, N> &coeff) {
  double value = 0., first = 0., second = 0.;
  for (auto i = coeff.rbegin(); i != coeff.rend(); ++i) {
    second = second * x + 2. * first;
    first = first * x + value;
    value = value * x + *i;
  }
  return {value, first, second};
}
anharmonic_thermal_models::ThermalProperties
thermal_properties(double temperature, double theta, double f, double first,
                   double second) {
  const double t = temperature / theta;
  return {theta * f,
          -first,
          -t * second,
          f - t * first,
          t * t * second / theta,
          t * second / theta};
}
} // namespace

namespace debye_temperature_models {
double DebyeTemperatureModelBase::dVrel(double x,
                                        const types::MineralParams &p) const {
  return -value(x, p) * gamma(x, p) / x;
}
double DebyeTemperatureModelBase::dVrel2(double x,
                                         const types::MineralParams &p) const {
  const double g = gamma(x, p);
  return value(x, p) / (x * x) * (g * (g + 1.) - x * gamma_prime(x, p));
}
void SLB::validate_parameters(types::MineralParams &p) const {
  validate_debye(p);
  require_finite(p.q_0, "q_0");
}
double SLB::value(double x, const types::MineralParams &p) const {
  const auto [a1, a2, f] = slb_coefficients(x, p);
  const double squared = 1. + a1 * f + .5 * a2 * f * f;
  if (!(std::isfinite(squared) && squared > 0.))
    throw std::domain_error(
        "SLB Debye temperature is outside its real, positive domain.");
  return *p.debye_0 * std::sqrt(squared);
}
double SLB::gamma(double x, const types::MineralParams &p) const {
  const auto [a1, a2, f] = slb_coefficients(x, p);
  return (2. * f + 1.) * (a1 + a2 * f) / (6. * (1. + a1 * f + .5 * a2 * f * f));
}
double SLB::gamma_prime(double x, const types::MineralParams &p) const {
  const auto [a1, a2, f] = slb_coefficients(x, p);
  const double n = (2. * f + 1.) * (a1 + a2 * f);
  const double d = 6. * (1. + a1 * f + .5 * a2 * f * f);
  const double dn = 2. * (a1 + a2 * f) + (2. * f + 1.) * a2;
  const double dd = 6. * (a1 + a2 * f);
  return (dn * d - n * dd) / (d * d) * (-std::pow(x, -5. / 3.) / 3.);
}
void PowerLawGammaSimple::validate_parameters(types::MineralParams &p) const {
  validate_debye(p);
  require_finite(p.q_0, "q_0");
}
double PowerLawGammaSimple::value(double x,
                                  const types::MineralParams &p) const {
  return *p.debye_0 *
         std::exp(-*p.grueneisen_0 * integrated_power(*p.q_0, log_volume(x)));
}
double PowerLawGammaSimple::gamma(double x,
                                  const types::MineralParams &p) const {
  return *p.grueneisen_0 * std::pow(x, *p.q_0);
}
double PowerLawGammaSimple::gamma_prime(double x,
                                        const types::MineralParams &p) const {
  return *p.q_0 * gamma(x, p) / x;
}
void PowerLawGamma::validate_parameters(types::MineralParams &p) const {
  validate_debye(p);
  require_finite(p.c_1, "c_1");
  require_finite(p.c_2, "c_2");
  require_finite(p.q_1, "q_1");
  require_finite(p.q_2, "q_2");
}
double PowerLawGamma::value(double x, const types::MineralParams &p) const {
  const double l = log_volume(x);
  return *p.debye_0 * std::exp(-*p.grueneisen_0 * l -
                               *p.c_1 * (integrated_power(*p.q_1, l) - l) -
                               *p.c_2 * (integrated_power(*p.q_2, l) - l));
}
double PowerLawGamma::gamma(double x, const types::MineralParams &p) const {
  const double l = log_volume(x);
  return *p.grueneisen_0 + *p.c_1 * std::expm1(*p.q_1 * l) +
         *p.c_2 * std::expm1(*p.q_2 * l);
}
double PowerLawGamma::gamma_prime(double x,
                                  const types::MineralParams &p) const {
  return (*p.c_1 * *p.q_1 * std::pow(x, *p.q_1) +
          *p.c_2 * *p.q_2 * std::pow(x, *p.q_2)) /
         x;
}
} // namespace debye_temperature_models

namespace anharmonic_prefactor_models {
void PowerLaw::validate_parameters(types::MineralParams &p) const {
  require_finite(p.a_anh, "a_anh");
  require_finite(p.m_anh, "m_anh");
}
double PowerLaw::value(double x, const types::MineralParams &p) const {
  return *p.a_anh * std::pow(x, *p.m_anh);
}
double PowerLaw::dVrel(double x, const types::MineralParams &p) const {
  return *p.m_anh * value(x, p) / x;
}
double PowerLaw::dVrel2(double x, const types::MineralParams &p) const {
  return *p.m_anh * (*p.m_anh - 1.) * value(x, p) / (x * x);
}
void Sigmoid::validate_parameters(types::MineralParams &p) const {
  require_finite(p.a_anh, "a_anh");
  require_finite(p.b_anh, "b_anh");
  require_finite(p.c_anh, "c_anh");
}
double Sigmoid::value(double x, const types::MineralParams &p) const {
  const double y = std::pow(x + *p.b_anh, *p.c_anh);
  return *p.a_anh * y / (1. + y);
}
double Sigmoid::dVrel(double x, const types::MineralParams &p) const {
  const double u = x + *p.b_anh;
  const double d = 1. + std::pow(u, *p.c_anh);
  return *p.a_anh * *p.c_anh * std::pow(u, *p.c_anh - 1.) / (d * d);
}
double Sigmoid::dVrel2(double x, const types::MineralParams &p) const {
  const double u = x + *p.b_anh, c = *p.c_anh;
  const double y = std::pow(u, c), d = 1. + y;
  return *p.a_anh * c * std::pow(u, c - 2.) * ((c - 1.) - (c + 1.) * y) /
         (d * d * d);
}
} // namespace anharmonic_prefactor_models

namespace anharmonic_thermal_models {
void Pade::validate_parameters(types::MineralParams &) const {}
ThermalProperties Pade::evaluate(double temperature, double theta,
                                 const types::MineralParams &) const {
  const double t = temperature / theta;
  const auto [p, dp, d2p] = polynomial(
      t, std::array<double, 8>{0., 0., 0., 0., .235256039, -.00744162069,
                               23.0722431, 366.827130});
  const auto [q, dq, d2q] = polynomial(
      t, std::array<double, 6>{1., 4.1784544, 46.00154792, 168.5910059,
                               568.90929887, 737.13224108});
  const double f = p / q;
  const double first = (dp - f * dq) / q;
  const double second = (d2p - f * d2q - 2. * first * dq) / q;
  return thermal_properties(temperature, theta, f, first, second);
}
void LogNormal::validate_parameters(types::MineralParams &p) const {
  require_finite(p.mu_anh, "mu_anh");
  require_finite(p.sigma_anh, "sigma_anh");
  if (*p.sigma_anh <= 0.)
    throw std::invalid_argument("sigma_anh must be positive.");
}
ThermalProperties LogNormal::evaluate(double temperature, double theta,
                                      const types::MineralParams &p) const {
  const double mu = *p.mu_anh, sigma = *p.sigma_anh;
  const double constant = .5 * std::exp(mu + .5 * sigma * sigma);
  if (temperature == 0.)
    return {-constant * theta, -.5 * mu, 0., -constant, 0., 0.};
  const double t = temperature / theta, u = std::log(t), diff = u - mu;
  const double z = diff / (sigma * std::sqrt(2.));
  const double cdf = .5 * std::erfc(-z);
  const double first =
      .5 * mu + diff * cdf +
      sigma / std::sqrt(2. * std::acos(-1.)) * std::exp(-z * z);
  const double f =
      t * (first - cdf) + constant * std::erf(z - sigma / std::sqrt(2.));
  return thermal_properties(temperature, theta, f, first, cdf / t);
}
} // namespace anharmonic_thermal_models
} // namespace burnman::eos
