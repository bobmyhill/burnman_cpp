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

#include "burnman/core/planet.hpp"
#include "radial.hpp"

namespace burnman::python {
void bind_planet(py::module_ &m) {
  using Array = Eigen::ArrayXd;
  auto planet = py::class_<Planet, std::shared_ptr<Planet>>(m, "Planet", R"doc(
A spherical planet assembled from adjoining Layer class objects.

Layers are sorted by inner radius. Set each layer's material and temperature
mode, then call make(). The default boundary conditions are zero surface
pressure and zero central gravity. Internal arrays retain both material states
at an interface. Queries at an exact interface select the deeper layer.
)doc");
  planet
      .def(py::init<std::string, std::vector<std::shared_ptr<Layer>>, int,
                    double, bool>(),
           py::arg("name"), py::arg("layers"), py::arg("n_max_iterations") = 50,
           py::arg("max_delta") = 1.e-5, py::arg("verbose") = false,
           py::keep_alive<1, 3>())
      .def_readwrite("name", &Planet::name)
      .def_readwrite("verbose", &Planet::verbose)
      .def_property_readonly("layers", &Planet::get_layers)
      .def_property_readonly("n_slices", &Planet::get_n_slices)
      .def_property_readonly("radius_planet", &Planet::get_radius_planet)
      .def_property_readonly("volume", &Planet::get_volume)
      .def_property_readonly("pressure_mode", &Planet::get_pressure_mode)
      .def_property_readonly("pressure_top", &Planet::get_pressure_top)
      .def_property_readonly("gravity_bottom", &Planet::get_gravity_bottom)
      .def_property_readonly("made", &Planet::is_made)
      .def_property_readonly("converged", &Planet::converged)
      .def_property_readonly("n_iterations", &Planet::get_n_iterations)
      .def_property_readonly("n_max_iterations", &Planet::get_n_max_iterations)
      .def_property_readonly("max_delta", &Planet::get_max_delta)
      .def("reset", &Planet::reset)
      .def("get_layer", &Planet::get_layer, py::arg("name"))
      .def("get_layer_by_radius", &Planet::get_layer_by_radius,
           py::arg("radius"))
      .def(
          "set_pressure_mode",
          [](Planet &self, const std::string &mode, const py::object &pressures,
             double top, double bottom, int iterations, double delta) {
            self.set_pressure_mode(mode, radial_array(pressures), top, bottom,
                                   iterations, delta);
          },
          py::arg("pressure_mode") = "self-consistent",
          py::arg("pressures") = py::none(), py::arg("pressure_top") = 0.,
          py::arg("gravity_bottom") = 0., py::arg("n_max_iterations") = 50,
          py::arg("max_delta") = 1.e-5,
          "Set prescribed pressures or iterate hydrostatic equilibrium across "
          "all layers.")
      .def(
          "make",
          [](Planet &self) {
            self.make();
            if (!self.converged() &&
                PyErr_WarnEx(PyExc_RuntimeWarning,
                             "Planet pressure iteration did not converge.",
                             1) < 0)
              throw py::error_already_set();
          },
          "Compute the planet's pressure, temperature and gravity profiles.")
      .def(
          "evaluate",
          [](const Planet &self, const std::vector<std::string> &names,
             const py::object &radii) {
            return radial_values(self.evaluate(
                names, radii.is_none()
                           ? std::nullopt
                           : std::optional<Array>(radial_array(radii))));
          },
          py::arg("properties"), py::arg("radlist") = py::none(),
          "Evaluate properties on the internal grid or at query radii, "
          "preserving query order.")
      .def_property_readonly("mass", &Planet::get_mass)
      .def_property_readonly("average_density", &Planet::get_average_density)
      .def_property_readonly("moment_of_inertia",
                             &Planet::get_moment_of_inertia)
      .def_property_readonly("moment_of_inertia_factor",
                             &Planet::get_moment_of_inertia_factor)
      .def_property_readonly("depth", &Planet::get_depth)
      .def("__str__", &Planet::to_string)
      .def(
          "__iter__",
          [](const Planet &self) {
            return py::make_iterator(self.get_layers().begin(),
                                     self.get_layers().end());
          },
          py::keep_alive<0, 1>());
  bind_radial_properties<Planet>(planet);
}
} // namespace burnman::python
