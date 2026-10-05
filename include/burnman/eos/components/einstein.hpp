/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
#ifndef BURNMAN_EOS_EINSTEIN_HPP_INCLUDED
#define BURNMAN_EOS_EINSTEIN_HPP_INCLUDED

namespace burnman {
namespace eos {

/**
 * @namespace burnman::eos::einstein
 * @brief Functions for the Einstein model of a solid.
 *
 * @note All functions assume SI units for all properties.
 */
namespace einstein {

// napfu is the number of atoms per formula unit and may be fractional.
// Temperatures are in kelvin.

/** Thermal energy [J/mol], excluding zero-point energy. */
double compute_thermal_energy(double temperature, double einstein_temperature,
                              double napfu);

/** Molar heat capacity at constant volume [J/(mol K)]. */
double compute_molar_heat_capacity_v(double temperature,
                                     double einstein_temperature, double napfu);

/** Helmholtz energy [J/mol], excluding zero-point energy. */
double compute_helmholtz_free_energy(double temperature,
                                     double einstein_temperature, double napfu);

/** Entropy from lattice vibrations [J/(mol K)]. */
double compute_entropy(double temperature, double einstein_temperature,
                       double napfu);

/** Temperature derivative of heat capacity at constant volume [J/(mol K^2)]. */
double compute_dmolar_heat_capacity_v_dT(double temperature,
                                         double einstein_temperature,
                                         double napfu);

} // namespace einstein
} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_EINSTEIN_HPP_INCLUDED
