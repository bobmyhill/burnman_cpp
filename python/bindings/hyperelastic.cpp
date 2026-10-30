// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/hyperelastic.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_hyperelastic(py::module_ &m) {
  py::enum_<OutputFrame>(m, "OutputFrame")
      .value("crystal", OutputFrame::crystal)
      .value("current", OutputFrame::current);
  auto mineral =
      py::class_<HyperelasticMineral, AnisotropicMineral, py::smart_holder>(
          m, "HyperelasticMineral");
  mineral
      .def(py::init<std::shared_ptr<AnisotropicMineral>>(),
           py::arg("hydrostatic_mineral"), py::keep_alive<1, 2>())
      .def(py::init([](const HyperelasticMineral &other) {
             return std::dynamic_pointer_cast<HyperelasticMineral>(
                 other.copy());
           }),
           py::arg("other"))
      .def("set_state_with_molar_cell_vectors",
           &HyperelasticMineral::set_state_with_molar_cell_vectors,
           py::arg("cell_vectors"), py::arg("T"))
      .def("set_output_frame",
           py::overload_cast<const std::string &>(
               &HyperelasticMineral::set_output_frame),
           py::arg("output_frame"))
      .def("set_output_frame",
           py::overload_cast<OutputFrame>(
               &HyperelasticMineral::set_output_frame),
           py::arg("output_frame"))
      .def_property_readonly("output_frame",
                             [](const HyperelasticMineral &self) {
                               return self.get_output_frame() ==
                                              OutputFrame::current
                                          ? "current"
                                          : "crystal";
                             })
      .def_property_readonly("hydrostatic",
                             &HyperelasticMineral::get_hydrostatic);
#define PROPERTY(name)                                                         \
  bind_property<HyperelasticMineral>(mineral, #name, "get_" #name,             \
                                     &HyperelasticMineral::get_##name, true)
  PROPERTY(rotation_to_crystallographic_frame);
  PROPERTY(hencky_strain);
  PROPERTY(molar_isostress_heat_capacity);
#undef PROPERTY
  auto diagnostic = [&](const char *name, auto getter) {
    mineral.def_property_readonly(
        name,
        [getter](const HyperelasticMineral &self) {
          check_state(self);
          return (self.*getter)();
        },
        py::return_value_policy::copy);
  };
  diagnostic("cauchy_stress", &HyperelasticMineral::get_cauchy_stress_tensor);
  diagnostic("thermal_stress", &HyperelasticMineral::get_thermal_stress_tensor);
  diagnostic("_cell_vectors_current",
             &HyperelasticMineral::get_current_cell_vectors);
  mineral.def_property_readonly(
      "_M_unrotated", [](const HyperelasticMineral &self) {
        return self.get_unrotated_cell_vectors().transpose().eval();
      });
  diagnostic("_helmholtz_elastic", &HyperelasticMineral::get_helmholtz_elastic);
  diagnostic("_cauchy_stress_unrotated",
             &HyperelasticMineral::get_cauchy_stress_unrotated);
  diagnostic("_thermal_stress_unrotated",
             &HyperelasticMineral::get_thermal_stress_unrotated);
  diagnostic("_isothermal_stiffness_tensor_unrotated",
             &HyperelasticMineral::get_unrotated_isothermal_stiffness_tensor);
}
} // namespace burnman::python
