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

#ifndef BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_TYPES_HPP_INCLUDED
#define BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_TYPES_HPP_INCLUDED

#include "burnman/optim/roots/damped_newton_types.hpp"
#include "burnman/utils/types/ndarray.hpp"
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

namespace burnman {
namespace equilibration {

/** Type alias for a map of elements and bulk compositional degrees of freedom
 */
using FreeVectorMap = std::unordered_map<std::string, double>;

/**
 * @brief Contains parameters and data required for equilibration.
 */
struct EquilibrationParameters {
  std::vector<std::string> parameter_names;   ///< Names of all parameters.
  Eigen::VectorXd bulk_composition_vector;    ///< Bulk composition vector of
                                              ///< assemblage elements.
  Eigen::VectorXd reduced_composition_vector; ///< Bulk composition resitricted
                                              ///< to independent elements.
  Eigen::MatrixXd
      free_compositional_vectors; ///< Matrix of free compositional vectors
                                  ///< (each row one vector).
  Eigen::MatrixXd
      reduced_free_composition_vectors; ///< Free compositional vectors
                                        ///< restricted to independent elements.
  Eigen::VectorXd
      constraint_vector; ///< Constraint vector, b, for constraints: (A·x + b).
  Eigen::MatrixXd
      constraint_matrix; ///< Constraint matrix, A, for contraints (A·x + b).
  Eigen::ArrayXi
      phase_amount_indices; ///< Indices in parameter vector of phase amounts.
  Eigen::Index
      n_parameters; ///< Number of parameters for the equilibrium problem.
};

/**
 * @brief Holds result of equilibration along with parameters
 */
struct EquilibrateResult {
  types::NDArray<optim::roots::DampedNewtonResult>
      sol_array;               ///< NDArray of solver result iterates.
  EquilibrationParameters prm; ///< Parameters used in equilibration.
};

} // namespace equilibration
} // namespace burnman

#endif // BURNMAN_TOOLS_EQUILIBRATION_EQUILIBRATE_TYPES_HPP_INCLUDED
