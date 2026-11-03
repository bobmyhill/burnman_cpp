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

#include "burnman/minerals/model_sets.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_model_sets(py::module_ &m) {
  namespace sets = minerals::model_sets;
  auto module =
      m.attr("minerals")
          .cast<py::module_>()
          .def_submodule(
              "model_sets",
              "Curated phase-model collections with calibration provenance.");
  py::class_<sets::ModelSet>(module, "ModelSet")
      .def_readonly("name", &sets::ModelSet::name)
      .def_readonly("version", &sets::ModelSet::version)
      .def_readonly("dataset", &sets::ModelSet::dataset)
      .def_readonly("source", &sets::ModelSet::source)
      .def_readonly("citation", &sets::ModelSet::citation)
      .def_readonly("notes", &sets::ModelSet::notes)
      .def_readonly("water", &sets::ModelSet::water)
      .def_readonly("recommended_max_pressure",
                    &sets::ModelSet::recommended_max_pressure)
      .def_readonly("cautious_max_pressure",
                    &sets::ModelSet::cautious_max_pressure)
      .def_readonly("phases", &sets::ModelSet::phases)
      .def_readonly("phase_glossary", &sets::ModelSet::phase_glossary)
      .def("to_dict", [](const sets::ModelSet &self) {
        py::dict data;
        data["name"] = self.name;
        data["version"] = self.version;
        data["dataset"] = self.dataset;
        data["source"] = self.source;
        data["citation"] = self.citation;
        data["water"] = self.water;
        data["recommended_max_pressure_Pa"] = self.recommended_max_pressure;
        data["cautious_max_pressure_Pa"] = self.cautious_max_pressure;
        if (!self.notes.empty())
          data["notes"] = self.notes;
        if (!self.phase_glossary.empty())
          data["phase_glossary"] = self.phase_glossary;
        return data;
      });
  module.def("metapelite", &sets::metapelite);
  module.def("metabasite", &sets::metabasite, py::arg("clinopyroxene") = "dio");
  module.def(
      "igneous", &sets::igneous, py::arg("calibration") = "G25",
      "Matched G25 (hydrous) or W24 (dry) igneous models with dataset 6.36.");
  module.def("slb2024", &sets::slb2024,
             "The 27 active HeFESTo SLB2024 solids, with ordinary solution "
             "models and metallic iron; no melts, fluids or unused lppv.");
}
} // namespace burnman::python
