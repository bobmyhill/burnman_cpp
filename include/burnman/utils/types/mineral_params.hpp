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

#ifndef BURNMAN_UTILS_TYPES_MINERAL_PARAMS_HPP_INCLUDED
#define BURNMAN_UTILS_TYPES_MINERAL_PARAMS_HPP_INCLUDED

#include "burnman/utils/types/simple_types.hpp"
#include <array>
#include <memory>
#include <optional>
#include <string>

namespace burnman {
class EquationOfState;
namespace eos {
namespace debye_temperature_models {
class DebyeTemperatureModelBase;
}
namespace anharmonic_prefactor_models {
class AnharmonicPrefactorModel;
}
namespace anharmonic_thermal_models {
class AnharmonicThermalModelBase;
}
} // namespace eos
namespace types {

/**
 * Four Cp parameters used in HP and CORK
 */
struct CpParams {
  double a, b, c, d;
};

/**
 * Parameters for CORK
 */
struct CorkParams {
  double a_0, a_1, b, c_0, c_1, d_0, d_1;
};

/**
 * Struct to hold mineral EOS parameters
 */
struct MineralParams {
  // Using std::optional so one struct can be used for all EOS types
  // We want to be able to switch between EOS without redefining minerals
  // Sometime we want to calculate default values (e.g. P0, T0) if not set
  // Should remove std::optional for anything always required (V0?, etc.)

  // Mineral properties
  std::optional<std::string> name;
  std::optional<FormulaMap> formula;
  std::optional<EOSType> equation_of_state;
  std::optional<double> napfu; // Atoms per formula unit; may be fractional.
  std::optional<double> molar_mass;
  std::optional<double> Z; // Number of formula units per crystallographic cell.

  // Reference conditions
  std::optional<double> P_0;
  std::optional<double> T_0;
  std::optional<double> E_0;
  std::optional<double> F_0;

  // Reference volume
  std::optional<double> V_0;

  // Elastic moduli
  std::optional<double> K_0;
  std::optional<double> Kprime_0;
  std::optional<double> Kdprime_0;
  std::optional<double> G_0;
  std::optional<double> Gprime_0;

  // Thermodynamic limit (infinite pressure) parameters
  std::optional<double> Kprime_inf;
  std::optional<double> Gprime_inf;

  // Other thermodynamic properties
  std::optional<double> S_0;
  std::optional<double> H_0;
  std::optional<double> Cv; // DKS
  std::optional<CpParams> Cp;

  // Thermal parameters
  std::optional<double> debye_0;
  std::optional<double> grueneisen_0;
  std::optional<double> q_0;
  std::optional<double> a_0; // alpha_0
  std::optional<double> dKdT_0;
  std::optional<double> m;                           // DKS free param
  std::optional<std::vector<std::vector<double>>> a; // DKS bonding coefficients
  std::optional<double>
      eta_s_0; // for shear strain derivative of grueneisen parameter (SLB, DKS)
  std::optional<double>
      T_einstein; // Used for SLB --> can be calculated from S_0 and napfu

  // Electronic parameters (SLB Conductive)
  std::optional<double> bel_0;
  std::optional<double> gel;

  // Anderson-Ahrens: K_0/Kprime_0/Kdprime_0 describe the reference ISENTROPE.
  // Array prefixes avoid the existing DKS bonding-matrix parameter a.
  std::optional<double> grueneisen_prime, grueneisen_n, theta, xi_0;
  std::optional<std::array<double, 2>> aa_a, aa_b, aa_Theta, aa_F;
  std::optional<std::array<double, 3>> aa_lmda;

  // Components and parameters of the modular MGD equation of state.
  std::shared_ptr<EquationOfState> reference_eos;
  std::shared_ptr<eos::debye_temperature_models::DebyeTemperatureModelBase>
      debye_temperature_model;
  std::shared_ptr<eos::anharmonic_prefactor_models::AnharmonicPrefactorModel>
      anharmonic_prefactor_model;
  std::shared_ptr<eos::anharmonic_thermal_models::AnharmonicThermalModelBase>
      anharmonic_thermal_model;
  std::optional<double> c_1, c_2, q_1, q_2;
  std::optional<double> a_anh, b_anh, c_anh, m_anh, mu_anh, sigma_anh;

  // Order of expansion (DKS)
  std::optional<int> order_theta;
  std::optional<int> order_f;

  // DKS free parameters for vol dependence of Tel and zeta
  // Tel - lower conductivity limit
  // zeta -  electronic heat capacity coefficient
  std::optional<double> Tel_0;
  std::optional<double> zeta_0;
  std::optional<double> xi;
  std::optional<double> eta;
  std::optional<double> el_V_0;
  std::optional<std::array<double, 2>> spin_a;
  std::optional<std::array<double, 4>> spin_b;
  // Original Fortran thermal law (default true); false selects finite strain.
  std::optional<bool> dks_s_power_law_grueneisen;

  // Brosh-CALPHAD fitting parameters
  std::optional<double> delta_0;
  std::optional<double> delta_1;
  std::optional<double> b_0;
  std::optional<double> b_1;

  // CORK params
  std::optional<double> cork_T;
  std::optional<double> cork_P;
  std::optional<CorkParams> cork_params;
  std::optional<std::array<double, 4>> dcorkdT;
};

} // namespace types
} // namespace burnman

#endif // BURNMAN_UTILS_TYPES_MINERAL_PARAMS_HPP_INCLUDED
