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

#include "burnman/core/solution.hpp"
#include "burnman/optim/linear_fitting.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace burnman::optim {
/** A Solution class object for chemical compositions without a thermodynamic
 * model.
 *
 * Supply elemental and crystallographic site formulae in the same endmember
 * order, for example {"Mg2SiO4", "Fe2SiO4"} and {"[Mg]2SiO4", "[Fe]2SiO4"}.
 * Elemental stoichiometry and endmember site occupancies are available
 * immediately.
 * Set the endmember molar fractions before requesting the bulk formula or
 * site occupancies. Thermodynamic properties are unavailable.
 */
class DummyCompositionSolution : public Solution {
public:
  DummyCompositionSolution(
      const std::vector<std::string> &endmember_element_formulae,
      const std::vector<std::string> &endmember_site_formulae);

  std::shared_ptr<Material> copy() const override;
  void set_composition(const Eigen::ArrayXd &molar_fractions) override;
};

/// Ordered definitions of new variables as weighted sums of known variables.
using VariableConversions =
    std::vector<std::pair<std::string, std::map<std::string, double>>>;

/** Fit elemental and site-species amounts to a Solution class object.
 *
 * fitted_variables contains element names, site-species names such as Fef_B,
 * or names defined in variable_conversions. Each conversion is a weighted sum
 * of elements, site species or previously defined variables. Values are molar
 * amounts; variable_covariances contains their observation covariance (the
 * identity if omitted). The weighted design must have full column rank.
 * Observations with magnitude below 1e-10 impose exact zero amounts. Site
 * occupancies must be nonnegative, while endmember coordinates may be signed.
 * The solution's composition and state are not changed.
 *
 * Let x, Cx and rx be the fitted amounts, their unconstrained covariance and
 * weighted sum of squared residuals. With normalize=true, s=sum(x) must be
 * positive; return p=x/s and r=rx/s, and take C0=Cx/s^2. Otherwise p=x,
 * r=rx and C0=Cx. In both cases, the returned covariance is J*C0*J', where
 * J=I-p*1'. It is not conditioned on active compositional constraints.
 */
LeastSquaresResult fit_composition_to_solution(
    const Solution &solution, const std::vector<std::string> &fitted_variables,
    const Eigen::VectorXd &variable_values,
    const std::optional<Eigen::MatrixXd> &variable_covariances,
    const VariableConversions &variable_conversions = {},
    bool normalize = true);

/** Fit nonnegative phase amounts to a bulk composition by least squares.
 *
 * phase_compositions has one row per chemical component and one column per
 * phase. bulk_composition uses the same component ordering, units and
 * compositional basis. Mass or molar compositions may be used.
 *
 * Returns the fitted phase amounts, pinv(A' * A), and the sum of squared
 * compositional residuals, where A is phase_compositions. Observations have
 * unit covariance. The amounts are not constrained to sum to one, and the
 * covariance is not conditioned on the nonnegativity constraints. The phase
 * composition matrix must have full column rank.
 */
LeastSquaresResult fit_phase_proportions_to_bulk_composition(
    const Eigen::MatrixXd &phase_compositions,
    const Eigen::VectorXd &bulk_composition);
} // namespace burnman::optim
