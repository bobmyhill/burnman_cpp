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

#include "burnman/eos/spock.hpp"
#include "burnman/eos/components/gsl_params.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_gamma.h>
#include <stdexcept>

namespace burnman::eos {
namespace {

struct SpockParams {
  double a, b, c;
};

SpockParams make_params(const types::MineralParams &params) {
  const double delta = *params.Kprime_0 - *params.Kprime_inf;
  const double a = -*params.Kdprime_0 * *params.K_0 / delta;
  const double b = delta / a;
  if (!std::isfinite(a) || !std::isfinite(b) || a <= 0.0 || b <= 0.0) {
    throw std::invalid_argument(
        "SPOCK requires K_0 > 0, Kdprime_0 < 0 and Kprime_0 > Kprime_inf.");
  }
  return {a, b, *params.Kprime_inf};
}

double gamma_interval(double a, double x1, double x2) {
  // Shift to nonnegative order, then recur downward as in Python BurnMan.
  // Direct GSL upper incomplete gamma can lose accuracy for negative orders.
  // Snap roundoff near integers so recurrence never divides by almost zero.
  if (std::abs(a - std::round(a)) < 1.0e-12) {
    a = std::round(a);
  }
  const int shifts = a < 0.0 ? static_cast<int>(-std::floor(a)) : 0;
  a += shifts;
  double value = std::abs(a) < 1.0e-12
                     ? gsl_sf_expint_E1(x1) - gsl_sf_expint_E1(x2)
                     : (gsl_sf_gamma_inc_P(a, x2) - gsl_sf_gamma_inc_P(a, x1)) *
                           gsl_sf_gamma(a);
  const double e1 = std::exp(-x1);
  const double e2 = std::exp(-x2);
  for (int i = 0; i < shifts; ++i) {
    a -= 1.0;
    value = (value + std::pow(x2, a) * e2 - std::pow(x1, a) * e1) / a;
  }
  return value;
}

struct IntegralParams {
  SpockParams shape;
  double ln_volume;
  bool energy;
};

double integrand(double t, void *p) {
  const auto &data = *static_cast<const IntegralParams *>(p);
  const auto &[a, b, c] = data.shape;
  const double u = data.ln_volume * (1.0 - t);
  const double k = std::exp(-b * std::expm1(a * u) - c * u);
  // exp(u) - V/V_0, evaluated without cancellation near V_0.
  return data.energy
             ? k * std::exp(data.ln_volume) * std::expm1(u - data.ln_volume)
             : k;
}

double local_integral(const SpockParams &shape, double ln_volume, bool energy) {
  // Near V_0 the incomplete-gamma differences (and especially the energy's
  // difference of differences) lose precision. Integrate the equivalent
  // smooth, dimensionless bulk-modulus expression over a short interval.
  IntegralParams data{shape, ln_volume, energy};
  gsl_function function;
  function.function = &integrand;
  function.params = &data;
  double result, error;
  std::size_t evaluations;
  auto *old_handler = gsl_set_error_handler_off();
  const int status = gsl_integration_qng(&function, 0.0, 1.0, 0.0, 1.0e-12,
                                         &result, &error, &evaluations);
  gsl_set_error_handler(old_handler);
  if (status != GSL_SUCCESS) {
    throw std::runtime_error("SPOCK local integration failed to converge.");
  }
  return -ln_volume * result;
}

bool use_local_integral(const SpockParams &shape, double ln_volume) {
  return std::abs(ln_volume) < 0.05 && std::abs(shape.a * ln_volume) < 0.05;
}

} // namespace

void SPOCK::validate_parameters(types::MineralParams &params) {
  utils::require_set(params.V_0, "V_0");
  utils::require_set(params.K_0, "K_0");
  utils::require_set(params.Kprime_0, "Kprime_0");
  utils::require_set(params.Kdprime_0, "Kdprime_0");
  utils::require_set(params.Kprime_inf, "Kprime_inf");
  utils::fallback_to_default(params.F_0, 0.0);
  utils::fallback_to_default(params.P_0, 1.0e5);
  if (params.E_0.has_value()) {
    throw std::invalid_argument(
        "Isothermal equations of state should be defined in terms of "
        "Helmholtz free energy F_0, not internal energy E_0.");
  }
  utils::check_in_range(params.P_0, 0.0, 1.0e100, "P_0");
  utils::check_in_range(params.V_0, 1.0e-7, 1.0e-3, "V_0");
  utils::check_in_range(params.K_0, 1.0e9, 1.0e13, "K_0");
  utils::check_in_range(params.Kprime_0, 0.0, 10.0, "Kprime_0");
  if (*params.Kdprime_0 > 0.0) {
    utils::warn("Kdprime_0 should be negative");
  }
  if (-*params.Kdprime_0 * *params.K_0 <
      *params.Kprime_0 - *params.Kprime_inf) {
    utils::warn("Kdprime_0*K_0 is expected to be more negative than "
                "(Kprime_0 - Kprime_inf)");
  }
  utils::check_in_range(params.Kprime_inf, 5.0 / 3.0, *params.Kprime_0,
                        "Kprime_inf");
}

double SPOCK::compute_spock(double compression,
                            const types::MineralParams &params) {
  if (compression == 1.0) {
    return *params.P_0;
  }
  const auto shape = make_params(params);
  const auto &[a, b, c] = shape;
  const double ln_volume = std::log(compression);
  if (use_local_integral(shape, ln_volume)) {
    return *params.P_0 + *params.K_0 * local_integral(shape, ln_volume, false);
  }
  return *params.P_0 +
         *params.K_0 * std::exp(b) / a * std::pow(b, c / a) *
             gamma_interval(-c / a, b * std::exp(a * ln_volume), b);
}

double SPOCK::spock_gsl_wrapper(double volume, void *p) {
  const auto &data = *static_cast<const gsl_params::SolverParams_P *>(p);
  return compute_spock(volume / *data.params.V_0, data.params) - data.pressure;
}

double SPOCK::compute_volume(double pressure,
                             double temperature [[maybe_unused]],
                             const types::MineralParams &params) const {
  gsl_params::SolverParams_P data{params, pressure};
  return optim::roots::brent(&spock_gsl_wrapper, data, 0.1 * *params.V_0,
                             1.5 * *params.V_0, 1.0e-18 * *params.V_0);
}

double SPOCK::compute_pressure(double temperature [[maybe_unused]],
                               double volume,
                               const types::MineralParams &params) const {
  return compute_spock(volume / *params.V_0, params);
}

double SPOCK::compute_spock_bulk_modulus(double volume,
                                         const types::MineralParams &params) {
  const auto [a, b, c] = make_params(params);
  const double ln_volume = std::log(volume / *params.V_0);
  return *params.K_0 * std::exp(-b * std::expm1(a * ln_volume) - c * ln_volume);
}

double SPOCK::compute_isothermal_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  return compute_spock_bulk_modulus(volume, params);
}

double SPOCK::compute_isentropic_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  return compute_spock_bulk_modulus(volume, params);
}

double SPOCK::compute_helmholtz_free_energy(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  const double compression = volume / *params.V_0;
  if (compression == 1.0) {
    return *params.F_0;
  }
  const auto shape = make_params(params);
  const auto &[a, b, c] = shape;
  const double ln_volume = std::log(compression);
  double strain_energy;
  if (use_local_integral(shape, ln_volume)) {
    strain_energy =
        *params.V_0 * *params.K_0 * local_integral(shape, ln_volume, true);
  } else {
    const double x = b * std::exp(a * ln_volume);
    const double i1 =
        std::pow(b, 1.0 / a) * compression * gamma_interval(-c / a, x, b);
    const double i2 = gamma_interval((1.0 - c) / a, x, b);
    strain_energy = -*params.V_0 * *params.K_0 * std::exp(b) / a *
                    std::pow(b, (c - 1.0) / a) * (i1 - i2);
  }
  // Preserve the reference-pressure energy convention in Python SPOCK.
  return *params.F_0 + *params.P_0 * (volume - *params.V_0) + strain_energy;
}

double
SPOCK::compute_gibbs_free_energy(double pressure, double temperature,
                                 double volume,
                                 const types::MineralParams &params) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, params) +
         pressure * volume;
}

double
SPOCK::compute_molar_internal_energy(double pressure, double temperature,
                                     double volume,
                                     const types::MineralParams &params) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, params);
}

double SPOCK::compute_enthalpy(double pressure, double temperature,
                               double volume,
                               const types::MineralParams &params) const {
  return compute_gibbs_free_energy(pressure, temperature, volume, params);
}

// Isothermal placeholders follow Python SPOCK / IsothermalEquationOfState.
double SPOCK::compute_grueneisen_parameter(double pressure [[maybe_unused]],
                                           double temperature [[maybe_unused]],
                                           double volume [[maybe_unused]],
                                           const types::MineralParams &params
                                           [[maybe_unused]]) const {
  return 0.0;
}

double SPOCK::compute_shear_modulus(double pressure [[maybe_unused]],
                                    double temperature [[maybe_unused]],
                                    double volume [[maybe_unused]],
                                    const types::MineralParams &params
                                    [[maybe_unused]]) const {
  return 1.0e99;
}

double SPOCK::compute_molar_heat_capacity_v(double pressure [[maybe_unused]],
                                            double temperature [[maybe_unused]],
                                            double volume [[maybe_unused]],
                                            const types::MineralParams &params
                                            [[maybe_unused]]) const {
  return 1.0e-99;
}

double SPOCK::compute_molar_heat_capacity_p(double pressure [[maybe_unused]],
                                            double temperature [[maybe_unused]],
                                            double volume [[maybe_unused]],
                                            const types::MineralParams &params
                                            [[maybe_unused]]) const {
  return 1.0e-99;
}

double SPOCK::compute_thermal_expansivity(double pressure [[maybe_unused]],
                                          double temperature [[maybe_unused]],
                                          double volume [[maybe_unused]],
                                          const types::MineralParams &params
                                          [[maybe_unused]]) const {
  return 0.0;
}

double SPOCK::compute_entropy(double pressure [[maybe_unused]],
                              double temperature [[maybe_unused]],
                              double volume [[maybe_unused]],
                              const types::MineralParams &params
                              [[maybe_unused]]) const {
  return 0.0;
}

} // namespace burnman::eos
