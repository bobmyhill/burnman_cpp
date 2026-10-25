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

#include "burnman/eos/murnaghan.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <stdexcept>

namespace burnman::eos {

void Murnaghan::validate_parameters(types::MineralParams &params) {
  // Check for required parameters.
  utils::require_set(params.V_0, "V_0");
  utils::require_set(params.K_0, "K_0");
  utils::require_set(params.Kprime_0, "Kprime_0");
  // Set defaults for missing values, following the other athermal EOS.
  utils::fallback_to_default(params.E_0, 0.0);
  utils::fallback_to_default(params.P_0, 0.0);
  utils::fallback_to_default(params.G_0, std::nan(""));
  utils::fallback_to_default(params.Gprime_0, std::nan(""));
  if (params.F_0)
    params.E_0 = params.F_0;
  // Check values are reasonable.
  utils::check_in_range(params.P_0, 0.0, 1.0e100, "P_0");
  utils::check_in_range(params.V_0, 1.0e-7, 1.0e-3, "V_0");
  utils::check_in_range(params.K_0, 1.0e9, 1.0e13, "K_0");
  utils::check_in_range(params.Kprime_0, -5.0, 30.0, "Kprime_0");
}

double Murnaghan::compute_murnaghan(double inv_compression,
                                    const types::MineralParams &params) {
  if (!std::isfinite(inv_compression) || inv_compression <= 0.0)
    throw std::invalid_argument(
        "Inverse compression must be positive and finite.");
  const double ln_compression = std::log(inv_compression);
  const double kprime = *params.Kprime_0;
  // The Kprime_0 = 0 limit has constant bulk modulus.
  const double pressure =
      kprime == 0.0
          ? *params.K_0 * ln_compression
          : *params.K_0 * std::expm1(kprime * ln_compression) / kprime;
  return pressure + *params.P_0;
}

double Murnaghan::compute_volume(double pressure,
                                 double temperature [[maybe_unused]],
                                 const types::MineralParams &params) const {
  const double relative_pressure = (pressure - *params.P_0) / *params.K_0;
  const double kprime = *params.Kprime_0;
  if (kprime == 0.0)
    return *params.V_0 * std::exp(-relative_pressure);
  if (1.0 + kprime * relative_pressure <= 0.0)
    throw std::invalid_argument(
        "Murnaghan pressure is outside the positive bulk modulus branch.");
  return *params.V_0 *
         std::exp(-std::log1p(kprime * relative_pressure) / kprime);
}

double Murnaghan::compute_pressure(double temperature [[maybe_unused]],
                                   double volume,
                                   const types::MineralParams &params) const {
  return compute_murnaghan(*params.V_0 / volume, params);
}

double Murnaghan::compute_isothermal_bulk_modulus_reuss(
    double pressure, double temperature [[maybe_unused]],
    double volume [[maybe_unused]], const types::MineralParams &params) const {
  return *params.K_0 + *params.Kprime_0 * (pressure - *params.P_0);
}

double Murnaghan::compute_isentropic_bulk_modulus_reuss(
    double pressure, double temperature, double volume,
    const types::MineralParams &params) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               params);
}

double Murnaghan::compute_molar_internal_energy(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume, const types::MineralParams &params) const {
  if (!std::isfinite(volume) || volume <= 0.0)
    throw std::invalid_argument("Volume must be positive and finite.");
  const double strain = (volume - *params.V_0) / *params.V_0;
  const double kprime = *params.Kprime_0;
  double energy;
  if (std::abs(strain) < 1.0e-3 && std::abs((kprime + 1.0) * strain) < 1.0e-3) {
    // The integral starts at second order in strain. Expand locally to avoid
    // subtracting nearly equal first-order terms near V_0.
    double term = 0.5 * strain * strain;
    energy = term;
    for (int order = 3; order <= 8; ++order) {
      term *= -(kprime + order - 2.0) * strain / order;
      energy += term;
    }
    energy *= *params.K_0 * *params.V_0;
  } else {
    const double ln_volume = std::log(volume / *params.V_0);
    if (std::abs(kprime) < 0.5) {
      // This equivalent integral stays finite as Kprime_0 tends to zero.
      const double compression_pressure =
          kprime == 0.0
              ? -*params.K_0 * ln_volume
              : *params.K_0 * std::expm1(-kprime * ln_volume) / kprime;
      energy =
          (volume * compression_pressure + *params.K_0 * *params.V_0 * strain) /
          (kprime - 1.0);
    } else {
      const double integral =
          kprime == 1.0
              ? ln_volume
              : std::expm1((1.0 - kprime) * ln_volume) / (1.0 - kprime);
      energy = *params.K_0 * *params.V_0 * (strain - integral) / kprime;
    }
  }
  return *params.E_0 + energy + *params.P_0 * (*params.V_0 - volume);
}

double
Murnaghan::compute_gibbs_free_energy(double pressure, double temperature,
                                     double volume,
                                     const types::MineralParams &params) const {
  return compute_molar_internal_energy(pressure, temperature, volume, params) +
         pressure * volume;
}

double Murnaghan::compute_helmholtz_free_energy(
    double pressure, double temperature, double volume,
    const types::MineralParams &params) const {
  return compute_molar_internal_energy(pressure, temperature, volume, params);
}

double Murnaghan::compute_enthalpy(double pressure, double temperature,
                                   double volume,
                                   const types::MineralParams &params) const {
  return compute_gibbs_free_energy(pressure, temperature, volume, params);
}

// Meaningless parameters for an athermal EOS; use the BM3/Vinet defaults.
double Murnaghan::compute_grueneisen_parameter(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume [[maybe_unused]],
    const types::MineralParams &params [[maybe_unused]]) const {
  return 0.0;
}

double Murnaghan::compute_shear_modulus(double pressure [[maybe_unused]],
                                        double temperature [[maybe_unused]],
                                        double volume [[maybe_unused]],
                                        const types::MineralParams &params
                                        [[maybe_unused]]) const {
  return 0.0;
}

double Murnaghan::compute_molar_heat_capacity_v(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume [[maybe_unused]],
    const types::MineralParams &params [[maybe_unused]]) const {
  return 1.0e99;
}

double Murnaghan::compute_molar_heat_capacity_p(
    double pressure [[maybe_unused]], double temperature [[maybe_unused]],
    double volume [[maybe_unused]],
    const types::MineralParams &params [[maybe_unused]]) const {
  return 1.0e99;
}

double Murnaghan::compute_thermal_expansivity(double pressure [[maybe_unused]],
                                              double temperature
                                              [[maybe_unused]],
                                              double volume [[maybe_unused]],
                                              const types::MineralParams &params
                                              [[maybe_unused]]) const {
  return 0.0;
}

double Murnaghan::compute_entropy(double pressure [[maybe_unused]],
                                  double temperature [[maybe_unused]],
                                  double volume [[maybe_unused]],
                                  const types::MineralParams &params
                                  [[maybe_unused]]) const {
  return 0.0;
}

} // namespace burnman::eos
