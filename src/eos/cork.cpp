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

#include "burnman/eos/cork.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include "burnman/utils/warnings.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace burnman::eos {
namespace {
std::array<double, 4> cork_variables(const types::CorkParams &cork,
                                     double cork_P, double cork_T,
                                     double temperature) {
  const double a = cork.a_0 * std::pow(cork_T, 2.5) / cork_P +
                   cork.a_1 * std::pow(cork_T, 1.5) / cork_P * temperature;
  const double b = cork.b * cork_T / cork_P;
  const double c = cork.c_0 * cork_T / std::pow(cork_P, 1.5) +
                   cork.c_1 / std::pow(cork_P, 1.5) * temperature;
  const double d = cork.d_0 * cork_T / (cork_P * cork_P) +
                   cork.d_1 / (cork_P * cork_P) * temperature;
  return {a, b, c, d};
}

double int_cp_over_t(double temperature, const types::MineralParams &params) {
  const auto &cp = *params.Cp;
  const double T_0 = *params.T_0;
  return (cp.a * std::log(temperature) + cp.b * temperature -
          0.5 * cp.c / (temperature * temperature) -
          2.0 * cp.d / std::sqrt(temperature)) -
         (cp.a * std::log(T_0) + cp.b * T_0 - 0.5 * cp.c / (T_0 * T_0) -
          2.0 * cp.d / std::sqrt(T_0));
}
} // namespace

double CORK::compute_volume(double pressure, double temperature,
                            const types::MineralParams &params) const {
  const auto [a, b, c, d] = cork_variables(*params.cork_params, *params.cork_P,
                                           *params.cork_T, temperature);
  const double RT = constants::physics::gas_constant * temperature;
  return RT / pressure +
         (b -
          a * constants::physics::gas_constant * std::sqrt(temperature) /
              ((RT + b * pressure) * (RT + 2.0 * b * pressure)) +
          c * std::sqrt(pressure) + d * pressure);
}

double CORK::compute_isothermal_bulk_modulus_reuss(
    double pressure, double temperature, double volume,
    const types::MineralParams &params) const {
  const auto [a, b, c, d] = cork_variables(*params.cork_params, *params.cork_P,
                                           *params.cork_T, temperature);
  const double RT = constants::physics::gas_constant * temperature;
  const double dVdP = -RT / (pressure * pressure) +
                      (-a * b / std::sqrt(temperature) *
                           (1.0 / std::pow(RT + b * pressure, 2.0) -
                            4.0 / std::pow(RT + 2.0 * b * pressure, 2.0)) +
                       0.5 * c / std::sqrt(pressure) + d);
  return -volume / dVdP;
}

double CORK::compute_shear_modulus(double, double, double,
                                   const types::MineralParams &) const {
  return 0.0;
}

double
CORK::compute_thermal_expansivity(double pressure, double temperature,
                                  double volume,
                                  const types::MineralParams &params) const {
  const auto cork = cork_variables(*params.cork_params, *params.cork_P,
                                   *params.cork_T, temperature);
  const auto &dcorkdT = *params.dcorkdT;
  const double RT = constants::physics::gas_constant * temperature;
  const double c1P = cork[1] * pressure;
  const double dVdT =
      constants::physics::gas_constant / pressure +
      (-dcorkdT[0] * constants::physics::gas_constant * std::sqrt(temperature) /
           ((RT + c1P) * (RT + 2.0 * c1P)) +
       0.5 * cork[0] * constants::physics::gas_constant *
           (-2.0 * c1P * c1P + 3.0 * RT * (c1P + RT)) / std::sqrt(temperature) /
           (std::pow(RT + c1P, 2.0) * std::pow(RT + 2.0 * c1P, 2.0)) +
       dcorkdT[2] * std::sqrt(pressure) + dcorkdT[3] * pressure);
  return dVdT / volume;
}

double
CORK::compute_molar_heat_capacity_p0(double temperature,
                                     const types::MineralParams &params) const {
  const auto &cp = *params.Cp;
  return cp.a + cp.b * temperature + cp.c * std::pow(temperature, -2.0) +
         cp.d * std::pow(temperature, -0.5);
}

double
CORK::compute_molar_heat_capacity_p(double pressure, double temperature, double,
                                    const types::MineralParams &params) const {
  const double P_relative = pressure - *params.P_0;
  const double Cp0 = compute_molar_heat_capacity_p0(temperature, params);
  double d2RTlnfdTdT = 0.0;
  if (*params.cork_T != 0.0) {
    const auto cork = cork_variables(*params.cork_params, *params.cork_P,
                                     *params.cork_T, temperature);
    const auto &dcorkdT = *params.dcorkdT;
    const double RT = constants::physics::gas_constant * temperature;
    const double c1P = cork[1] * P_relative;
    const double logterm = std::log(RT + c1P) - std::log(RT + 2.0 * c1P);
    const double dlogtermdT = constants::physics::gas_constant *
                              (1.0 / (RT + c1P) - 1.0 / (RT + 2.0 * c1P));
    const double d2logtermdTdT =
        -constants::physics::gas_constant * constants::physics::gas_constant *
        (1.0 / std::pow(RT + c1P, 2.0) - 1.0 / std::pow(RT + 2.0 * c1P, 2.0));
    const double a =
        dcorkdT[0] / (cork[1] * std::sqrt(temperature)) -
        cork[0] / (2.0 * cork[1] * temperature * std::sqrt(temperature));
    const double dadT =
        -0.5 * dcorkdT[0] / (cork[1] * temperature * std::sqrt(temperature)) -
        dcorkdT[0] / (2.0 * cork[1] * temperature * std::sqrt(temperature)) +
        3.0 * cork[0] /
            (4.0 * cork[1] * temperature * temperature *
             std::sqrt(temperature));
    const double b = cork[0] / (cork[1] * std::sqrt(temperature));
    const double dbdT =
        dcorkdT[0] / (cork[1] * std::sqrt(temperature)) - 0.5 * b / temperature;
    // The c and d fugacity terms are linear in temperature.
    d2RTlnfdTdT =
        dadT * logterm + a * dlogtermdT + dbdT * dlogtermdT + b * d2logtermdTdT;
  }
  return Cp0 - temperature * d2RTlnfdTdT;
}

double
CORK::compute_gibbs_free_energy(double pressure, double temperature, double,
                                const types::MineralParams &params) const {
  const auto &cp = *params.Cp;
  const double T_0 = *params.T_0;
  const double P_relative = pressure - *params.P_0;
  const double intCpdT =
      (cp.a * temperature + 0.5 * cp.b * temperature * temperature -
       cp.c / temperature + 2.0 * cp.d * std::sqrt(temperature)) -
      (cp.a * T_0 + 0.5 * cp.b * T_0 * T_0 - cp.c / T_0 +
       2.0 * cp.d * std::sqrt(T_0));
  const double intCpoverTdT = int_cp_over_t(temperature, params);
  double RTlnf = 0.0;
  if (*params.cork_T != 0.0) {
    const auto [a, b, c, d] = cork_variables(
        *params.cork_params, *params.cork_P, *params.cork_T, temperature);
    const double RT = constants::physics::gas_constant * temperature;
    // Equation 8 in Holland and Powell (1991).
    RTlnf = RT * std::log(1.0e-5 * P_relative) + b * P_relative +
            a / (b * std::sqrt(temperature)) *
                (std::log(RT + b * P_relative) -
                 std::log(RT + 2.0 * b * P_relative)) +
            2.0 / 3.0 * c * P_relative * std::sqrt(P_relative) +
            d / 2.0 * P_relative * P_relative;
  }
  return *params.H_0 + intCpdT - temperature * (*params.S_0 + intCpoverTdT) +
         RTlnf;
}

double CORK::compute_entropy(double pressure, double temperature, double,
                             const types::MineralParams &params) const {
  const double P_relative = pressure - *params.P_0;
  const double intCpoverTdT = int_cp_over_t(temperature, params);
  double dRTlnfdT = 0.0;
  if (*params.cork_T != 0.0) {
    const auto cork = cork_variables(*params.cork_params, *params.cork_P,
                                     *params.cork_T, temperature);
    const auto &dcorkdT = *params.dcorkdT;
    const double RT = constants::physics::gas_constant * temperature;
    const double c1P = cork[1] * P_relative;
    const double logterm = std::log(RT + c1P) - std::log(RT + 2.0 * c1P);
    dRTlnfdT =
        constants::physics::gas_constant * std::log(1.0e-5 * P_relative) +
        (dcorkdT[0] / (cork[1] * std::sqrt(temperature)) -
         cork[0] / (2.0 * cork[1] * temperature * std::sqrt(temperature))) *
            logterm +
        cork[0] / (cork[1] * std::sqrt(temperature)) *
            constants::physics::gas_constant *
            (1.0 / (RT + c1P) - 1.0 / (RT + 2.0 * c1P)) +
        2.0 / 3.0 * dcorkdT[2] * P_relative * std::sqrt(P_relative) +
        dcorkdT[3] / 2.0 * P_relative * P_relative;
  }
  return *params.S_0 + intCpoverTdT - dRTlnfdT;
}

void CORK::validate_parameters(types::MineralParams &params) {
  utils::require_set(params.cork_params, "cork_params");
  utils::require_set(params.cork_T, "cork_T");
  utils::require_set(params.cork_P, "cork_P");
  utils::require_set(params.Cp, "Cp");
  utils::fallback_to_default(params.T_0, 298.15);
  utils::fallback_to_default(params.P_0, 0.0);
  utils::fallback_to_default(params.H_0,
                             std::numeric_limits<double>::quiet_NaN());
  utils::fallback_to_default(params.S_0,
                             std::numeric_limits<double>::quiet_NaN());
  if (!params.dcorkdT) {
    const auto &cork = *params.cork_params;
    const double cork_P = *params.cork_P;
    params.dcorkdT = {cork.a_1 * std::pow(*params.cork_T, 1.5) / cork_P, 0.0,
                      cork.c_1 / std::pow(cork_P, 1.5),
                      cork.d_1 / (cork_P * cork_P)};
  }
  if (*params.S_0 < 0.0)
    utils::warn("Unusual value for S_0");
  if (*params.cork_T < -1.0)
    utils::warn("Unusual value for cork_T");
  if (*params.cork_P < 1.0e4 || *params.cork_P > 1.0e8)
    utils::warn("Unusual value for cork_P");
  if (compute_molar_heat_capacity_p0(*params.T_0, params) < 0.0)
    utils::warn("Negative heat capacity at T_0");
  if (compute_molar_heat_capacity_p0(2000.0, params) < 0.0)
    utils::warn("Negative heat capacity at 2000K");
}

} // namespace burnman::eos
