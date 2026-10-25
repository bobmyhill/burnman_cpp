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

#include "burnman/eos/components/debye.hpp"
#include "burnman/utils/constants.hpp"
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_debye.h>

namespace burnman::eos {

double debye::debye_fn_integrand(double xi, void *) {
  return xi == 0.0 ? 0.0 : (xi * xi * xi) / std::expm1(xi);
}

double debye::debye_fn_quad(double x) {
  if (x == 0.0)
    return 1.0; // D_3(0): the classical high-temperature limit.
  gsl_integration_workspace *gsl_w = gsl_integration_workspace_alloc(1000);
  double result, error;
  gsl_function integrand;
  integrand.function = &debye_fn_integrand;
  gsl_integration_qags(&integrand, 0, x, // limits
                       1.49e-8,
                       1.49e-8, // eps from scipy TODO: move to constants
                       1000,    // limit
                       gsl_w, &result, &error);
  // error unused here
  // result hold integral
  // Free allocated memory
  gsl_integration_workspace_free(gsl_w);
  return 3.0 * result / (x * x * x);
}

// For errors do:
// gsl_sf_result result;
// int status = gsl_sf_debye_3_e(x, &result);
// for a string do: gsl_strerror(status);
// result.val, result.err
double debye::debye_fn_cheb(double x) { return gsl_sf_debye_3(x); }

double debye::compute_thermal_energy(double temperature,
                                     double debye_temperature, double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  // E_th
  return 3.0 * napfu * constants::physics::gas_constant * temperature *
         debye_fn_cheb(debye_temperature / temperature);
}

double debye::compute_molar_heat_capacity_v(double temperature,
                                            double debye_temperature,
                                            double napfu) {
  if (temperature < constants::precision::double_eps) {
    return 0.0;
  }
  double x = debye_temperature / temperature;
  // C_v
  return 3.0 * napfu * constants::physics::gas_constant *
         (4.0 * debye_fn_cheb(x) - 3.0 * x / std::expm1(x));
}

double debye::compute_helmholtz_free_energy(double temperature,
                                            double debye_temperature,
                                            double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = debye_temperature / temperature;
  // F
  return napfu * constants::physics::gas_constant * temperature *
         (3.0 * std::log(-std::expm1(-x)) - debye_fn_cheb(x));
}

double debye::compute_entropy(double temperature, double debye_temperature,
                              double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = debye_temperature / temperature;
  // S
  return napfu * constants::physics::gas_constant *
         (4.0 * debye_fn_cheb(x) - 3.0 * std::log(-std::expm1(-x)));
}

double debye::compute_dmolar_heat_capacity_v_dT(double temperature,
                                                double debye_temperature,
                                                double napfu) {
  if (temperature <= constants::precision::double_eps) {
    return 0.0;
  }
  double x = debye_temperature / temperature;
  const double prefactor =
      3.0 * napfu * constants::physics::gas_constant / temperature;
  if (x < .01) {
    const double x2 = x * x;
    return prefactor * x2 * (1. / 10. - x2 / 140. + x2 * x2 / 3024.);
  }
  const double d = debye_fn_cheb(x);
  if (x > 50.)
    return 12. * prefactor * d; // Exponential terms are below roundoff.
  const double occupation = std::exp(-x) / -std::expm1(-x);
  return prefactor * (12. * d - 9. * x * occupation -
                      3. * x * x * occupation * (1. + occupation));
}

double debye::compute_dhelmholtz_dTheta(double temperature, double theta,
                                        double napfu) {
  return compute_thermal_energy(temperature, theta, napfu) / theta;
}
double debye::compute_d2helmholtz_dTheta2(double temperature, double theta,
                                          double napfu) {
  return -temperature *
         compute_molar_heat_capacity_v(temperature, theta, napfu) /
         (theta * theta);
}
double debye::compute_dentropy_dTheta(double temperature, double theta,
                                      double napfu) {
  return -compute_molar_heat_capacity_v(temperature, theta, napfu) / theta;
}

} // namespace burnman::eos
