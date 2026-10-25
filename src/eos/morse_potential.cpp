// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/morse_potential.hpp"
#include "burnman/eos/components/gsl_params.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <algorithm>
#include <cmath>

namespace burnman::eos {
namespace {
struct Strain {
  double s, x, exponential, integral;
};
Strain strain(double compression, const types::MineralParams &p) {
  if (!std::isfinite(compression) || compression <= 0.)
    throw std::invalid_argument("Morse requires positive finite volume.");
  const double s = std::cbrt(compression);
  // This equivalent difference retains small strains close to V_0.
  const double mu = (1. - compression) / (1. + s + s * s);
  const double x = (*p.Kprime_0 - 1.) * mu;
  return {s, x, std::exp(x), mu * (x == 0. ? 1. : std::expm1(x) / x)};
}
double pressure_residual(double compression, void *data) {
  const auto &state = *static_cast<gsl_params::SolverParams_P *>(data);
  return (Morse::compute_morse(compression, state.params) - state.pressure) /
         *state.params.K_0;
}
double bulk_residual(double compression, void *data) {
  const auto &state = *static_cast<gsl_params::SolverParams_P *>(data);
  return Morse::compute_morse_bulk_modulus(compression * *state.params.V_0,
                                           state.params) /
         *state.params.K_0;
}
} // namespace

void Morse::validate_parameters(types::MineralParams &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.K_0, "K_0");
  utils::require_set(p.Kprime_0, "Kprime_0");
  utils::fallback_to_default(p.F_0, 0.);
  utils::fallback_to_default(p.P_0, 0.);
  utils::fallback_to_default(p.G_0, std::nan(""));
  utils::fallback_to_default(p.Gprime_0, std::nan(""));
  if (p.E_0)
    throw std::invalid_argument(
        "Morse uses Helmholtz free energy F_0, not internal energy E_0.");
  if (!std::isfinite(*p.V_0) || *p.V_0 <= 0. || !std::isfinite(*p.K_0) ||
      *p.K_0 <= 0. || !std::isfinite(*p.Kprime_0) || !std::isfinite(*p.P_0))
    throw std::invalid_argument("Invalid Morse reference parameters.");
  utils::check_in_range(p.P_0, 0., 1.e100, "P_0");
  utils::check_in_range(p.V_0, 1.e-7, 1.e-3, "V_0");
  utils::check_in_range(p.K_0, 1.e9, 1.e13, "K_0");
  utils::check_in_range(p.Kprime_0, 0., 10., "Kprime_0");
}

double Morse::compute_morse(double compression, const types::MineralParams &p) {
  const auto z = strain(compression, p);
  return *p.P_0 + 3. * *p.K_0 * z.exponential * z.integral / (z.s * z.s);
}
double Morse::compute_morse_bulk_modulus(double volume,
                                         const types::MineralParams &p) {
  const auto z = strain(volume / *p.V_0, p);
  return *p.K_0 * z.exponential / (z.s * z.s) *
         (2. * z.integral + z.s * (1. + 2. * std::expm1(z.x)));
}
double Morse::compute_pressure(double, double volume,
                               const types::MineralParams &p) const {
  return compute_morse(volume / *p.V_0, p);
}
double Morse::compute_volume(double pressure, double,
                             const types::MineralParams &p) const {
  if (!std::isfinite(pressure))
    throw std::invalid_argument("Morse requires finite pressure.");
  if (pressure == *p.P_0)
    return *p.V_0;
  gsl_params::SolverParams_P state{p, pressure};
  const bool compressing = pressure > *p.P_0;
  double previous = 1.;
  for (int step = 0; step < 200; ++step) {
    double endpoint = previous * (compressing ? .9 : 1.05);
    const double bulk = bulk_residual(endpoint, &state);
    if (!std::isfinite(bulk))
      break;
    if (bulk <= 0.)
      endpoint = optim::roots::brent(&bulk_residual, state,
                                     std::min(previous, endpoint),
                                     std::max(previous, endpoint), 1.e-14);
    const double residual = pressure_residual(endpoint, &state);
    if (!std::isfinite(residual))
      break;
    if ((compressing && residual >= 0.) || (!compressing && residual <= 0.))
      return *p.V_0 * optim::roots::brent(&pressure_residual, state,
                                          std::min(1., endpoint),
                                          std::max(1., endpoint), 1.e-14);
    // A target beyond the first K=0 point has no root on this stable branch.
    if (bulk <= 0.)
      break;
    previous = endpoint;
  }
  throw std::invalid_argument("Cannot bracket a stable Morse volume.");
}
double Morse::compute_isothermal_bulk_modulus_reuss(
    double, double, double volume, const types::MineralParams &p) const {
  return compute_morse_bulk_modulus(volume, p);
}
double Morse::compute_isentropic_bulk_modulus_reuss(
    double pressure, double temperature, double volume,
    const types::MineralParams &p) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               p);
}
double
Morse::compute_helmholtz_free_energy(double, double, double volume,
                                     const types::MineralParams &p) const {
  const auto z = strain(volume / *p.V_0, p);
  return *p.F_0 + 4.5 * *p.K_0 * *p.V_0 * z.integral * z.integral +
         *p.P_0 * (*p.V_0 - volume);
}
double Morse::compute_gibbs_free_energy(double pressure, double temperature,
                                        double volume,
                                        const types::MineralParams &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p) +
         pressure * volume;
}
double
Morse::compute_molar_internal_energy(double pressure, double temperature,
                                     double volume,
                                     const types::MineralParams &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p);
}
double Morse::compute_enthalpy(double pressure, double temperature,
                               double volume,
                               const types::MineralParams &p) const {
  return compute_gibbs_free_energy(pressure, temperature, volume, p);
}

// Isothermal placeholders, matching Python Morse and native SPOCK's thermal
// terms.
double Morse::compute_grueneisen_parameter(double, double, double,
                                           const types::MineralParams &) const {
  return 0.;
}
double Morse::compute_shear_modulus(double, double, double,
                                    const types::MineralParams &) const {
  return 0.;
}
double
Morse::compute_molar_heat_capacity_v(double, double, double,
                                     const types::MineralParams &) const {
  return 1.e-99;
}
double
Morse::compute_molar_heat_capacity_p(double, double, double,
                                     const types::MineralParams &) const {
  return 1.e-99;
}
double Morse::compute_thermal_expansivity(double, double, double,
                                          const types::MineralParams &) const {
  return 0.;
}
double Morse::compute_entropy(double, double, double,
                              const types::MineralParams &) const {
  return 0.;
}
} // namespace burnman::eos
