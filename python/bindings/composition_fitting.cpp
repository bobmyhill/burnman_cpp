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

#include "burnman/optim/composition_fitting.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_composition_fitting(py::module_ &m) {
  auto optimize = m.attr("optimize").cast<py::module_>();
  auto composition = optimize.def_submodule("composition_fitting");
  py::class_<optim::DummyCompositionSolution, Solution, py::smart_holder>(
      composition, "DummyCompositionSolution",
      R"doc(A Solution class object for compositions without a thermodynamic model.

Provide endmember_element_formulae and endmember_site_formulae as lists of
formula strings in the same order, for example ['Mg2SiO4', 'Fe2SiO4'] and
['[Mg]2SiO4', '[Fe]2SiO4']. Elemental stoichiometry and crystallographic site
occupancies are available without specifying pressure or temperature.

Call set_composition(molar_fractions) before requesting formula,
site_occupancies or site_formula(). Thermodynamic properties are unavailable.
)doc")
      .def(py::init<const std::vector<std::string> &,
                    const std::vector<std::string> &>(),
           py::arg("endmember_element_formulae"),
           py::arg("endmember_site_formulae"));
  composition.def(
      "fit_composition_to_solution",
      [](const Solution &solution,
         const std::vector<std::string> &fitted_variables,
         const Eigen::VectorXd &variable_values,
         const std::optional<Eigen::MatrixXd> &variable_covariances,
         const py::object &variable_conversions, bool normalize) {
        optim::VariableConversions conversions;
        if (!variable_conversions.is_none())
          for (const auto &entry : variable_conversions.cast<py::dict>())
            conversions.emplace_back(
                entry.first.cast<std::string>(),
                entry.second.cast<std::map<std::string, double>>());
        return optim::fit_composition_to_solution(
            solution, fitted_variables, variable_values, variable_covariances,
            conversions, normalize);
      },
      py::arg("solution"), py::arg("fitted_variables"),
      py::arg("variable_values"), py::arg("variable_covariances"),
      py::arg("variable_conversions") = py::none(), py::arg("normalize") = true,
      R"doc(Fit molar compositional observations to a Solution class object.

fitted_variables contains elements such as Fe, site species such as Fef_B,
or user-defined variables. variable_values contains their observed molar
amounts. variable_covariances is the observation covariance matrix, or None
for unit covariance. The weighted design must have full column rank.

variable_conversions maps each new name to a dictionary of known variables
and their coefficients, for example {'Mg_equal': {'Mg_A': 1., 'Mg_B': -1.}}.
Conversions may use previously defined variables in dictionary order.
Observed amounts with magnitude below 1e-10 impose exact zero constraints.
Site occupancies must be nonnegative; endmember coordinates may be signed.
The solution's composition and state are not changed.

Return (molar_fractions, covariance_matrix, residual). With normalize=False,
the first array contains fitted endmember amounts, without a sum-to-one
constraint; the tuple label retains the shared API convention. Let x, Cx and rx
be the
fitted endmember amounts, their unconstrained covariance and weighted sum of
squared residuals. If normalize is True, set s=sum(x), p=x/s, C0=Cx/s**2 and
residual=rx/s; s must be positive. Otherwise p=x, C0=Cx and residual=rx.
The returned covariance is J @ C0 @ J.T, where J=I-p[:,None]. It is not
conditioned on active compositional constraints.
)doc");
  composition.def(
      "fit_phase_proportions_to_bulk_composition",
      &optim::fit_phase_proportions_to_bulk_composition,
      py::arg("phase_compositions"), py::arg("bulk_composition"),
      R"doc(Fit nonnegative phase amounts to a bulk composition by least squares.

phase_compositions contains one row per chemical component and one column per
phase. bulk_composition has one value per component in the same order. Both
inputs must use the same units and compositional basis: mass or molar amounts.
The phase composition matrix must have full column rank.

Return (phase_proportions, covariance_matrix, residual). The fitted phase
amounts are not constrained to sum to one. The covariance is
pinv(phase_compositions.T @ phase_compositions), assuming unit observation
covariance; it is not conditioned on the nonnegativity constraints. The residual
is the sum of squared differences between fitted and observed bulk compositions.
)doc");
}
} // namespace burnman::python
