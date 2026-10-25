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

#include "burnman/eos/components/einstein.hpp"
#include "burnman/utils/constants.hpp"
#include <cmath>

namespace burnman::eos {

double einstein::compute_thermal_energy(double temperature,
                                        double einstein_temperature,
                                        double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = einstein_temperature / temperature;
  double exp_term = std::exp(-x) / -std::expm1(-x);
  // E_th
  return 3.0 * napfu * constants::physics::gas_constant * einstein_temperature *
         exp_term;
}

double einstein::compute_molar_heat_capacity_v(double temperature,
                                               double einstein_temperature,
                                               double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = einstein_temperature / temperature;
  if (x > 745.)
    return 0.;
  double ratio = x / -std::expm1(-x);
  // C_v
  return 3.0 * napfu * constants::physics::gas_constant * ratio * ratio *
         std::exp(-x);
}

double einstein::compute_helmholtz_free_energy(double temperature,
                                               double einstein_temperature,
                                               double napfu) {
  if (temperature <= constants::precision::double_eps)
    return 0.;
  return 3. * napfu * constants::physics::gas_constant * temperature *
         std::log(-std::expm1(-einstein_temperature / temperature));
}

double einstein::compute_entropy(double temperature,
                                 double einstein_temperature, double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = einstein_temperature / temperature;
  if (x > 745.)
    return 0.;
  const double denominator = -std::expm1(-x);
  // S
  return 3.0 * napfu * constants::physics::gas_constant *
         (x * std::exp(-x) / denominator - std::log(denominator));
}

double einstein::compute_dmolar_heat_capacity_v_dT(double temperature,
                                                   double einstein_temperature,
                                                   double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = einstein_temperature / temperature;
  if (x > 745.)
    return 0.;
  if (x < .01) {
    const double x2 = x * x;
    return 3. * napfu * constants::physics::gas_constant / temperature * x2 *
           (1. / 6. - x2 / 60. + x2 * x2 / 1008.);
  }
  const double occupation = std::exp(-x) / -std::expm1(-x);
  return compute_molar_heat_capacity_v(temperature, einstein_temperature,
                                       napfu) /
         temperature * (x * (1. + 2. * occupation) - 2.);
}

double einstein::compute_dhelmholtz_dTheta(double temperature, double theta,
                                           double napfu) {
  return compute_thermal_energy(temperature, theta, napfu) / theta;
}
double einstein::compute_d2helmholtz_dTheta2(double temperature, double theta,
                                             double napfu) {
  return -temperature *
         compute_molar_heat_capacity_v(temperature, theta, napfu) /
         (theta * theta);
}
double einstein::compute_dentropy_dTheta(double temperature, double theta,
                                         double napfu) {
  return -compute_molar_heat_capacity_v(temperature, theta, napfu) / theta;
}

} // namespace burnman::eos
