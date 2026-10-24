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

#include "burnman/tools/chemistry.hpp"
#include "bindings.hpp"
#include "burnman/tools/partitioning.hpp"
#include "burnman/utils/chemistry_utils.hpp"

namespace burnman::python {
void bind_chemistry(py::module_ &m) {
  auto chemistry =
      m.def_submodule("chemistry", "Native reaction chemistry in SI units.");
  chemistry.def("fugacity", &chemistry::fugacity, py::arg("standard_material"),
                py::arg("assemblage"),
                "Return component fugacity relative to a standard Material "
                "class object at the same P and T.");
  chemistry.def("relative_fugacity", &chemistry::relative_fugacity,
                py::arg("component_formula"), py::arg("assemblage"),
                py::arg("reference_assemblage"),
                "Return the component fugacity relative to an equilibrated "
                "reference assemblage at the same P and T.");
  chemistry.def("equilibrium_pressure", &chemistry::equilibrium_pressure,
                py::arg("minerals"), py::arg("stoichiometry"),
                py::arg("temperature"),
                py::arg("pressure_initial_guess") = 1.e5,
                "Solve for reaction equilibrium pressure (Pa) at temperature "
                "(K), leaving the materials at the root.");
  chemistry.def("equilibrium_temperature", &chemistry::equilibrium_temperature,
                py::arg("minerals"), py::arg("stoichiometry"),
                py::arg("pressure"),
                py::arg("temperature_initial_guess") = 1000.,
                "Solve for reaction equilibrium temperature (K) at pressure "
                "(Pa), leaving the materials at the root.");
  chemistry.def("invariant_point", &chemistry::invariant_point,
                py::arg("minerals_r1"), py::arg("stoichiometry_r1"),
                py::arg("minerals_r2"), py::arg("stoichiometry_r2"),
                py::arg("pressure_temperature_initial_guess") =
                    Eigen::Vector2d(1.e9, 1000.),
                "Return [pressure (Pa), temperature (K)] satisfying two "
                "independent reaction equilibria.");
  chemistry.def(
      "hugoniot", &chemistry::hugoniot, py::arg("mineral"), py::arg("P_ref"),
      py::arg("T_ref"), py::arg("pressures"),
      py::arg("reference_mineral") = nullptr,
      "Return temperatures (K) and volumes per current mole (m³/mol). Solve "
      "U-U_ref=(P-P_ref)*(V_ref-V)/2 on a fixed mass basis, allowing phase "
      "changes. For an EquilibratedComposite, solve temperature, phase "
      "amounts and compositions together with chemical equilibrium.");
  chemistry.def("reactions_from_stoichiometric_matrix",
                &chemistry::reactions_from_stoichiometric_matrix,
                py::arg("stoichiometric_matrix"),
                "Enumerate minimal balanced reactions in both directions; "
                "input rows give component amounts in each compound.");
  chemistry.def("reactions_from_formulae", &chemistry::reactions_from_formulae,
                py::arg("formulae"));
  chemistry.def(
      "dictionarize_formula",
      [](const std::string &formula) {
        return utils::dictionarize_formula(formula);
      },
      py::arg("formula"));
  chemistry.def("formula_mass", &utils::formula_mass, py::arg("formula"));
  auto partitioning =
      m.def_submodule("partitioning", "Native iron partitioning.");
  partitioning.def(
      "calculate_nakajima_fp_pv_partition_coefficient",
      &partitioning::calculate_nakajima_fp_pv_partition_coefficient,
      py::arg("pressure"), py::arg("temperature"),
      py::arg("bulk_composition_mol"),
      py::arg("initial_distribution_coefficient"),
      "Return Fe/(Fe+Mg) in ferropericlase and bridgmanite using Nakajima et "
      "al. (2012). Inputs use Pa, K and elemental moles.");
}
} // namespace burnman::python
