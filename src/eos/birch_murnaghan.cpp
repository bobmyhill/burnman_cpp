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

#include "burnman/eos/birch_murnaghan.hpp"
#include "burnman/eos/components/gsl_params.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace burnman::eos {

namespace {
double fourth_order_pressure(double volume,
                             const types::MineralParams &params) {
  const double f = .5 * (std::pow(*params.V_0 / volume, 2. / 3.) - 1.);
  const double xi = .75 * (4. - *params.Kprime_0);
  const double zeta =
      .375 * (*params.K_0 * *params.Kdprime_0 +
              *params.Kprime_0 * (*params.Kprime_0 - 7.) + 143. / 9.);
  return 3. * *params.K_0 * f * std::pow(1. + 2. * f, 2.5) *
             (1. - 2. * xi * f + 4. * zeta * f * f) +
         *params.P_0;
}
double fourth_order_residual(double volume, void *data) {
  const auto &state = *static_cast<gsl_params::SolverParams_P *>(data);
  return fourth_order_pressure(volume, state.params) - state.pressure;
}
} // namespace
void BM4::validate_parameters(types::MineralParams &params) {
  utils::fallback_to_default(params.G_0, 0.0);
  utils::fallback_to_default(params.Gprime_0, 0.0);
  BM3::validate_parameters(params);
  utils::require_set(params.Kdprime_0, "Kdprime_0");
  if (!std::isfinite(*params.Kdprime_0))
    throw std::invalid_argument("Kdprime_0 must be finite.");
  if (params.F_0)
    params.E_0 = params.F_0;
}
double BM4::compute_pressure(double, double volume,
                             const types::MineralParams &params) const {
  return fourth_order_pressure(volume, params);
}
double BM4::compute_volume(double pressure, double,
                           const types::MineralParams &params) const {
  gsl_params::SolverParams_P state{params, pressure};
  // Expand from V_0 along the stable compression or expansion branch.
  double lower = *params.V_0, upper = *params.V_0;
  if (pressure == *params.P_0)
    return *params.V_0;
  for (int i = 0; i < 200; ++i) {
    if (pressure > *params.P_0)
      lower *= .95;
    else
      upper *= 1.01;
    if (fourth_order_residual(lower, &state) *
            fourth_order_residual(upper, &state) <=
        0.)
      return optim::roots::brent(&fourth_order_residual, state, lower, upper,
                                 std::numeric_limits<double>::min());
  }
  throw std::runtime_error(
      "Cannot bracket a stable fourth-order Birch-Murnaghan volume.");
}
double BM4::compute_isothermal_bulk_modulus_reuss(
    double, double, double volume, const types::MineralParams &params) const {
  const double f = .5 * (std::pow(*params.V_0 / volume, 2. / 3.) - 1.);
  const double xi = .75 * (4. - *params.Kprime_0);
  const double zeta =
      .375 * (*params.K_0 * *params.Kdprime_0 +
              *params.Kprime_0 * (*params.Kprime_0 - 7.) + 143. / 9.);
  return *params.K_0 * std::pow(1. + 2. * f, 2.5) *
         (5. * f * (1. - 2. * xi * f + 4. * zeta * f * f) +
          (1. + 2. * f) * (1. - 4. * xi * f + 12. * zeta * f * f));
}
double BM4::compute_isentropic_bulk_modulus_reuss(
    double pressure, double temperature, double volume,
    const types::MineralParams &params) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               params);
}
double BM4::compute_shear_modulus(double, double, double,
                                  const types::MineralParams &) const {
  return 0.;
}
double
BM4::compute_molar_internal_energy(double, double, double volume,
                                   const types::MineralParams &params) const {
  const double f = .5 * (std::pow(*params.V_0 / volume, 2. / 3.) - 1.);
  const double xi = .75 * (4. - *params.Kprime_0);
  const double zeta =
      .375 * (*params.K_0 * *params.Kdprime_0 +
              *params.Kprime_0 * (*params.Kprime_0 - 7.) + 143. / 9.);
  return *params.E_0 +
         4.5 * *params.K_0 * *params.V_0 * f * f *
             (1. - (4. / 3.) * xi * f + 2. * zeta * f * f) +
         *params.P_0 * (*params.V_0 - volume);
}
double
BM4::compute_helmholtz_free_energy(double pressure, double temperature,
                                   double volume,
                                   const types::MineralParams &params) const {
  return compute_molar_internal_energy(pressure, temperature, volume, params);
}

void BM3::validate_parameters(types::MineralParams &params) {
  // Check for required keys
  utils::require_set(params.V_0, "V_0");
  utils::require_set(params.K_0, "K_0");
  utils::require_set(params.Kprime_0, "Kprime_0");
  // Set defaults for missing values
  utils::fallback_to_default(params.E_0, 0.0);
  utils::fallback_to_default(params.P_0, 0.0);
  utils::fallback_to_default(params.G_0, std::nan(""));
  utils::fallback_to_default(params.Gprime_0, std::nan(""));
  // Check values are reasonable
  utils::check_in_range(params.P_0, 0.0, 1.0e100, "P_0");
  utils::check_in_range(params.V_0, 1.0e-7, 1.0e-3, "V_0");
  utils::check_in_range(params.K_0, 1.0e9, 1.0e13, "K_0");
  utils::check_in_range(params.Kprime_0, 0.0, 20.0, "Kprime_0");
  utils::check_in_range(params.G_0, 0.0, 1.0e13, "G_0");
  utils::check_in_range(params.Gprime_0, -5.0, 10.0, "Gprime_0");
}

// Compute P(V) - P as function to root find
double BM3::bm_gsl_wrapper(double x, void *p) {
  auto *bm_params = static_cast<const gsl_params::SolverParams_P *>(p);
  return compute_birch_murnaghan(*bm_params->params.V_0 / x,
                                 bm_params->params) -
         bm_params->pressure;
}

double BM3::compute_birch_murnaghan(double inv_compression,
                                    const types::MineralParams &params) {
  double cbrt_x = std::cbrt(inv_compression);
  double x23 = cbrt_x * cbrt_x;
  double x53 = inv_compression * x23;
  double x73 = inv_compression * inv_compression * cbrt_x;
  return 1.5 * (*params.K_0) * (x73 - x53) *
         (1 + (0.75 * (*params.Kprime_0 - 4) * (x23 - 1)));
}

double BM3::compute_volume(double pressure, double temperature [[maybe_unused]],
                           const types::MineralParams &params) const {
  // Find root in [P(V) - P] to find V that fits P
  // Set up GSL params struct - to pass to objective function
  gsl_params::SolverParams_P bm_params{params, pressure};
  // Set a, b limits
  double x_lo = 0.1 * (*params.V_0);
  double x_hi = 1.5 * (*params.V_0);
  double volume_root = optim::roots::brent(&bm_gsl_wrapper, bm_params, x_lo,
                                           x_hi, 1.0e-18 * (*params.V_0));
  return volume_root;
}

double BM3::compute_pressure(double temperature [[maybe_unused]], double volume,
                             const types::MineralParams &params) const {
  // Prefer V_0/V (x^-1) for faster exp
  return compute_birch_murnaghan((*params.V_0) / volume, params);
}

double BM3::compute_bm_bulk_modulus(double volume,
                                    const types::MineralParams &params) {
  double x = (*params.V_0) / volume;
  double x_cbrt = std::cbrt(x);
  double f = 0.5 * ((x_cbrt * x_cbrt) - 1.0);
  // (1 + 2f)^(5/2) = (1+2f)^2 * (1+2f)^(1/2)
  double t = 1 + 2 * f;
  double K =
      t * t * std::sqrt(t) *
      ((*params.K_0) +
       (3.0 * (*params.K_0) * (*params.Kprime_0) - 5.0 * (*params.K_0)) * f +
       13.5 * ((*params.K_0) * (*params.Kprime_0) - 4.0 * (*params.K_0)) * f *
           f);
  return K;
}

double BM3::compute_isothermal_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  return compute_bm_bulk_modulus(volume, params);
}

double BM3::compute_isentropic_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  return compute_bm_bulk_modulus(volume, params);
}

double
BM3::compute_gibbs_free_energy(double pressure, double temperature,
                               double volume,
                               const types::MineralParams &params) const {
  return volume * pressure +
         compute_molar_internal_energy(pressure, temperature, volume, params);
}

double BM3::compute_molar_internal_energy(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  double x = std::cbrt(*params.V_0 / volume);
  x *= x; //(V_0/V)^(2/3)
  double t = x - 1;
  // 9/16 = 0.5625
  double fac = 0.5625 * (*params.V_0) * (*params.K_0);
  double intPdV =
      fac * ((t * t * t * (*params.Kprime_0)) + (t * t * (6 - 4 * x)));
  return *params.E_0 + intPdV;
}

double
BM2::compute_second_order_shear_modulus(double volume,
                                        const types::MineralParams &params) {
  double x = *params.V_0 / volume;
  double cbrt_x = std::cbrt(x);
  double x23 = cbrt_x * cbrt_x;
  // double x53 = x * x23;
  double G = (*params.G_0) * x * x23 *
             (1.0 - 0.5 * (x23 - 1.0) *
                        (5.0 - 3.0 * (*params.Gprime_0) * (*params.K_0) /
                                   (*params.G_0)));
  return G;
}

double
BM3::compute_third_order_shear_modulus(double volume,
                                       const types::MineralParams &params) {
  double x = *params.V_0 / volume;
  double x_cbrt = std::cbrt(x);
  double f = 0.5 * ((x_cbrt * x_cbrt) - 1.0);
  // (1 + 2f)^(5/2) = (1+2f)^2 * (1+2f)^(1/2)
  double t = 1 + 2 * f;
  double G =
      t * t * std::sqrt(t) *
      ((*params.G_0) +
       (3.0 * (*params.K_0) * (*params.Gprime_0) - 5.0 * (*params.G_0)) * f +
       (6.0 * (*params.K_0) * (*params.Gprime_0) - 24.0 * (*params.K_0) -
        14.0 * (*params.G_0) + 4.5 * (*params.K_0) * (*params.Kprime_0)) *
           f * f);
  return G;
}

// Second order (BM2)
double BM2::compute_shear_modulus(double pressure [[maybe_unused]],
                                  double temperature [[maybe_unused]],
                                  double volume,
                                  const types::MineralParams &params) const {
  return compute_second_order_shear_modulus(volume, params);
}

// Third order (BM3)
double BM3::compute_shear_modulus(double pressure [[maybe_unused]],
                                  double temperature [[maybe_unused]],
                                  double volume,
                                  const types::MineralParams &params) const {
  return compute_third_order_shear_modulus(volume, params);
}

// Meaningless parameters for isothermal EOS
// Returning defaults
double BM3::compute_grueneisen_parameter(double pressure [[maybe_unused]],
                                         double temperature [[maybe_unused]],
                                         double volume [[maybe_unused]],
                                         const types::MineralParams &params
                                         [[maybe_unused]]) const {
  return 0.0;
}

double BM3::compute_molar_heat_capacity_v(double pressure [[maybe_unused]],
                                          double temperature [[maybe_unused]],
                                          double volume [[maybe_unused]],
                                          const types::MineralParams &params
                                          [[maybe_unused]]) const {
  return 1.0e99;
}

double BM3::compute_molar_heat_capacity_p(double pressure [[maybe_unused]],
                                          double temperature [[maybe_unused]],
                                          double volume [[maybe_unused]],
                                          const types::MineralParams &params
                                          [[maybe_unused]]) const {
  return 1.0e99;
}

double BM3::compute_thermal_expansivity(double pressure [[maybe_unused]],
                                        double temperature [[maybe_unused]],
                                        double volume [[maybe_unused]],
                                        const types::MineralParams &params
                                        [[maybe_unused]]) const {
  return 0.0;
}

double BM3::compute_entropy(double pressure [[maybe_unused]],
                            double temperature [[maybe_unused]],
                            double volume [[maybe_unused]],
                            const types::MineralParams &params
                            [[maybe_unused]]) const {
  return 0.0;
}

} // namespace burnman::eos
