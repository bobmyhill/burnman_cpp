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

#pragma once

#include "burnman/core/assemblage.hpp"
#include <Eigen/Core>
#include <memory>
#include <utility>
#include <vector>

namespace burnman::chemistry {
using Materials = std::vector<std::shared_ptr<Material>>;

/// Fugacity relative to a standard Material class object at the same P and T.
double fugacity(const Material &standard_material, Assemblage &assemblage);
/// Fugacity of a chemical component relative to a reference assemblage.
double relative_fugacity(const types::FormulaMap &component_formula,
                         Assemblage &assemblage,
                         Assemblage &reference_assemblage);
/// Reaction equilibrium pressure in Pa at a prescribed temperature in K.
double equilibrium_pressure(const Materials &minerals,
                            const Eigen::VectorXd &stoichiometry,
                            double temperature,
                            double pressure_initial_guess = 1.e5);
/// Reaction equilibrium temperature in K at a prescribed pressure in Pa.
double equilibrium_temperature(const Materials &minerals,
                               const Eigen::VectorXd &stoichiometry,
                               double pressure,
                               double temperature_initial_guess = 1000.);
/// Simultaneously solve two reaction equilibria; return [pressure,
/// temperature].
Eigen::Vector2d invariant_point(
    const Materials &minerals_r1, const Eigen::VectorXd &stoichiometry_r1,
    const Materials &minerals_r2, const Eigen::VectorXd &stoichiometry_r2,
    const Eigen::Vector2d &pressure_temperature_initial_guess =
        Eigen::Vector2d(1.e9, 1000.));
/** Temperatures (K) and molar volumes (m³/mol) satisfying
 * U-U_ref = (P-P_ref)*(V_ref-V)/2. An optional reference material supplies
 * the initial energy and volume when the material changes phase. Energy
 * balance uses a fixed mass, permitting changes in formula-unit moles.
 * Returned volumes are per current mole of material.
 * For an EquilibratedComposite, temperature, phase amounts and compositions
 * satisfy the energy balance and chemical equilibrium simultaneously.
 */
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
hugoniot(Material &mineral, double P_ref, double T_ref,
         const Eigen::ArrayXd &pressures,
         Material *reference_mineral = nullptr);
/** Enumerate all minimal balanced reactions, including both directions.
 * Rows of the input give element amounts in each compound. Output rows give
 * signed compound amounts, normalized to primitive integer coefficients.
 */
Eigen::MatrixXd reactions_from_stoichiometric_matrix(
    const Eigen::MatrixXd &stoichiometric_matrix);
/// Balanced reactions between compounds specified by elemental formulae.
Eigen::MatrixXd
reactions_from_formulae(const std::vector<types::FormulaMap> &formulae);
} // namespace burnman::chemistry
