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

#ifndef BURNMAN_TOOLS_AVERAGING_AVERAGING_UTILS_HPP_INCLUDED
#define BURNMAN_TOOLS_AVERAGING_AVERAGING_UTILS_HPP_INCLUDED

#include <Eigen/Dense>
#include <utility>

namespace burnman {
namespace averaging {

// TODO: maybe rename - internal?
/**
 * @namespace burnman::averaging::utils
 * @brief Internal averaging utilities
 *
 * This namespace collects some internal utilities used by the classes in
 * `burnman::averaging'. The namespace includes basic averaging functions
 * such as Voigt and Reuss functions.
 */
namespace utils {

/**
 * @brief Coupled spherical self-consistent bulk and shear moduli.
 *
 * At a trial effective shear modulus, the bulk equation is solved explicitly.
 * The remaining shear equation is solved with Brent's method between the
 * smallest and largest occupied phase shear moduli. The zero-rigidity limit
 * is handled explicitly for mixtures containing fluids or voids.
 *
 * @param phase_volumes Finite, nonnegative phase volumes, with positive total.
 * @param bulk_moduli Finite, nonnegative bulk moduli of occupied phases.
 * @param shear_moduli Finite, nonnegative shear moduli of occupied phases.
 * @return Effective bulk and shear moduli, in the input modulus units.
 * @throws std::invalid_argument For inconsistent sizes or invalid inputs.
 * @throws std::runtime_error If the root finder fails to converge.
 */
std::pair<double, double>
self_consistent_fn(const Eigen::ArrayXd &phase_volumes,
                   const Eigen::ArrayXd &bulk_moduli,
                   const Eigen::ArrayXd &shear_moduli);

// Utility functions for V & R averages
/**
 * @brief Calculated Voigt (iso-strain) average.
 *
 * @param phase_volumes Partial volumes or molar fractions
 * @param X Phase moduli to average
 *
 * @return Voigt average
 */
double voigt_fn(const Eigen::ArrayXd &phase_volumes, const Eigen::ArrayXd &X);

/**
 * @brief Calculated Reuss (iso-stress) average.
 *
 * @param phase_volumes Partial volumes or molar fractions
 * @param X Phase moduli to average
 *
 * @return Reuss average
 */
double reuss_fn(const Eigen::ArrayXd &phase_volumes, const Eigen::ArrayXd &X);

/**
 * @brief Calculated Voigt-Reuss-Hill average.
 *
 * VRH average is the arithmetic mean of the V & R bounds.
 *
 * @param phase_volumes Partial volumes or molar fractions
 * @param X Phase moduli to average
 *
 * @return VRH average
 */
double voigt_reuss_hill_fn(const Eigen::ArrayXd &phase_volumes,
                           const Eigen::ArrayXd &X);

// Utility functions for HS Averages
/**
 * @see `HashinShtrikmanLower::average_bulk_moduli'.
 */
double lower_hs_bulk_fn(const Eigen::ArrayXd &volumes,
                        const Eigen::ArrayXd &bulk_moduli,
                        const Eigen::ArrayXd &shear_moduli);

/**
 * @see `HashinShtrikmanLower::average_shear_moduli'.
 */
double lower_hs_shear_fn(const Eigen::ArrayXd &volumes,
                         const Eigen::ArrayXd &bulk_moduli,
                         const Eigen::ArrayXd &shear_moduli);

/**
 * @see `HashinShtrikmanUpper::average_bulk_moduli'.
 */
double upper_hs_bulk_fn(const Eigen::ArrayXd &volumes,
                        const Eigen::ArrayXd &bulk_moduli,
                        const Eigen::ArrayXd &shear_moduli);

/**
 * @see `HashinShtrikmanUpper::average_shear_moduli'.
 */
double upper_hs_shear_fn(const Eigen::ArrayXd &volumes,
                         const Eigen::ArrayXd &bulk_moduli,
                         const Eigen::ArrayXd &shear_moduli);

} // namespace utils
} // namespace averaging
} // namespace burnman

#endif // BURNMAN_TOOLS_AVERAGING_AVERAGING_UTILS_HPP_INCLUDED
