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

#include "burnman/core/equilibrated_composite.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_equilibrated_composite(py::module_ &m) {
  py::class_<EquilibratedComposite, Assemblage,
             std::shared_ptr<EquilibratedComposite>>(m, "EquilibratedComposite")
      .def(py::init<const types::FormulaMap &,
                    const std::vector<std::shared_ptr<Material>> &,
                    const phase_diagrams::Settings &, bool,
                    types::AveragingType, bool>(),
           py::arg("composition"), py::arg("candidates"),
           py::arg("settings") = phase_diagrams::Settings{},
           py::arg("relaxed") = true,
           py::arg("averaging_scheme") = types::AveragingType::VRH,
           py::arg("warm_start") = true)
      .def("set_state", &EquilibratedComposite::set_state, py::arg("pressure"),
           py::arg("temperature"), py::call_guard<py::gil_scoped_release>())
      .def("set_averaging_scheme", &EquilibratedComposite::set_averaging_scheme,
           py::arg("averaging_scheme"))
      .def(
          "set_fractions",
          [](EquilibratedComposite &, const py::object &, const py::object &) {
            throw py::type_error(
                "EquilibratedComposite determines phase fractions from its "
                "bulk composition and state.");
          },
          py::arg("fractions"),
          py::arg("fraction_type") = types::FractionType::Molar)
      .def_property_readonly("n_moles", &EquilibratedComposite::get_n_moles)
      .def_property("relaxed", &EquilibratedComposite::get_relaxed,
                    &EquilibratedComposite::set_relaxed)
      .def_property("warm_start", &EquilibratedComposite::get_warm_start,
                    &EquilibratedComposite::set_warm_start)
      .def_property_readonly("averaging_scheme",
                             &EquilibratedComposite::get_averaging_scheme)
      .def_property_readonly("settings", &EquilibratedComposite::get_settings)
      .def_property_readonly("equilibrium_state",
                             &EquilibratedComposite::get_equilibrium_state)
      .def_property_readonly(
          "last_equilibrium_attempt",
          &EquilibratedComposite::get_last_equilibrium_attempt)
      .def_property_readonly("candidate_phases",
                             &EquilibratedComposite::get_candidate_phases)
      .def("snapshot", &EquilibratedComposite::snapshot)
      .def("_clear_state", &EquilibratedComposite::clear_state);
}
} // namespace burnman::python
