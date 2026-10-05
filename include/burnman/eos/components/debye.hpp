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

#ifndef BURNMAN_EOS_DEBYE_HPP_INCLUDED
#define BURNMAN_EOS_DEBYE_HPP_INCLUDED

namespace burnman {
namespace eos {

/**
 * @namespace burnman::eos::debye
 * @brief Functions for Debye model.
 *
 * Functions required for Debye model. Used with Mie-Grueneisen and
 * Birch-Murnaghan for a full EOS.
 *
 * @note All functions assume SI units for all properties.
 */
namespace debye {

/**
 @brief Integrand of third-order Debye function.
 */
double debye_fn_integrand(double xi,
                          void *); // TODO - internal use only - maybe put in
                                   // unnamed namespace in .cpp only

/**
 * @brief Evaluates the Debye function using numerical integration.
 */
double debye_fn_quad(double x);

/**
 * @brief Evaluates the Debye function using a Chebyshev series
 *        expansion coupled with asymptotic solutions of the function.
 *
 * @note Uses the GSL implementation adapted in PyBurnman as debye_fn_cheb
 */
double debye_fn_cheb(double x);

// napfu is the number of atoms per formula unit and may be fractional.
// Temperatures are in kelvin.

/** Thermal energy [J/mol], excluding zero-point energy. */
double compute_thermal_energy(double temperature, double debye_temperature,
                              double napfu);

/** Molar heat capacity at constant volume [J/(mol K)]. */
double compute_molar_heat_capacity_v(double temperature,
                                     double debye_temperature, double napfu);

/** Helmholtz energy [J/mol], excluding zero-point energy. */
double compute_helmholtz_free_energy(double temperature,
                                     double debye_temperature, double napfu);

/** Entropy from lattice vibrations [J/(mol K)]. */
double compute_entropy(double temperature, double debye_temperature,
                       double napfu);

/** Temperature derivative of heat capacity at constant volume [J/(mol K^2)]. */
double compute_dmolar_heat_capacity_v_dT(double temperature,
                                         double debye_temperature,
                                         double napfu);

} // namespace debye
} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_DEBYE_HPP_INCLUDED
