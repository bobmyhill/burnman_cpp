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
  py::class_<optim::DummyCompositionSolution, Solution,
             std::shared_ptr<optim::DummyCompositionSolution>>(
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
