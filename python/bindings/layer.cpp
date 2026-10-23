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

#include "burnman/core/layer.hpp"
#include "radial.hpp"
#include <cstring>

namespace burnman::python {
void bind_layer(py::module_ &m) {
  using Array = Eigen::ArrayXd;
  auto layer = py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer", R"doc(
A spherical material layer on increasing radii in metres.

Set a Material class object, temperature mode and pressure boundary conditions,
then call make(). Each radius retains an independent material state. Radial
integrals use GSL cubic splines; property queries at other radii interpolate
pressure and temperature before evaluating the material.
)doc");
  layer
      .def(py::init([](const std::optional<std::string> &name,
                       const py::object &radii, bool verbose) {
             return std::make_shared<Layer>(name.value_or(""),
                                            radial_array(radii), verbose);
           }),
           py::arg("name") = py::none(), py::arg("radii") = py::none(),
           py::arg("verbose") = false)
      .def_readwrite("name", &Layer::name)
      .def_readwrite("verbose", &Layer::verbose)
      .def_readonly("inner_radius", &Layer::inner_radius)
      .def_readonly("outer_radius", &Layer::outer_radius)
      .def_readonly("thickness", &Layer::thickness)
      .def_readonly("n_slices", &Layer::n_slices)
      .def_readonly("n_start", &Layer::n_start)
      .def_readonly("n_end", &Layer::n_end)
      .def("reset", &Layer::reset)
      .def("set_material", &Layer::set_material, py::arg("material"),
           py::keep_alive<1, 2>())
      .def_property_readonly("material", &Layer::get_material)
      .def_property_readonly("sublayers", &Layer::get_sublayers)
      .def_property_readonly("pressure_mode", &Layer::get_pressure_mode)
      .def_property_readonly("temperature_mode", &Layer::get_temperature_mode)
      .def_property_readonly("temperature_top", &Layer::get_temperature_top)
      .def_property_readonly("pressure_top", &Layer::get_pressure_top)
      .def_property_readonly("gravity_bottom", &Layer::get_gravity_bottom)
      .def_property_readonly("usertemperatures", &Layer::get_usertemperatures)
      .def_property_readonly("converged", &Layer::converged)
      .def_property_readonly("n_iterations", &Layer::get_n_iterations)
      .def_property_readonly("n_max_iterations", &Layer::get_n_max_iterations)
      .def_property_readonly("max_delta", &Layer::get_max_delta)
      .def_property_readonly("made", &Layer::is_made)
      .def(
          "set_temperature_mode",
          [](Layer &self, const std::string &mode,
             const py::object &temperatures, std::optional<double> top) {
            self.set_temperature_mode(mode, radial_array(temperatures), top);
          },
          py::arg("temperature_mode") = "adiabatic",
          py::arg("temperatures") = py::none(),
          py::arg("temperature_top") = py::none(), R"doc(
Set 'user-defined', 'adiabatic' or 'perturbed-adiabatic' temperatures.

Temperatures use K and match the radius grid. For a perturbed adiabat,
temperature_top is the actual upper-boundary temperature, including the
perturbation there. An absent temperature_top inherits the temperature of the
overlying layer when the layer belongs to a Planet.
)doc")
      .def(
          "set_pressure_mode",
          [](Layer &self, const std::string &mode, const py::object &pressures,
             std::optional<double> bottom, std::optional<double> top,
             int iterations, double delta) {
            self.set_pressure_mode(mode, radial_array(pressures), bottom, top,
                                   iterations, delta);
          },
          py::arg("pressure_mode") = "self-consistent",
          py::arg("pressures") = py::none(),
          py::arg("gravity_bottom") = py::none(),
          py::arg("pressure_top") = py::none(),
          py::arg("n_max_iterations") = 50, py::arg("max_delta") = 1.e-5, R"doc(
Set 'user-defined' or 'self-consistent' pressures in Pa.

A standalone layer requires gravity_bottom in m/s². Self-consistent pressures
also require pressure_top in Pa. max_delta limits the relative change in the
maximum pressure between iterations.
)doc")
      .def("make", &Layer::make,
           "Compute pressure and temperature profiles and independent material "
           "states.")
      .def(
          "evaluate",
          [](const Layer &self, const std::vector<std::string> &names,
             const py::object &radii, std::optional<double> planet_radius) {
            return radial_values(self.evaluate(
                names,
                radii.is_none() ? std::nullopt
                                : std::optional<Array>(radial_array(radii)),
                planet_radius));
          },
          py::arg("properties"), py::arg("radlist") = py::none(),
          py::arg("radius_planet") = py::none(), R"doc(
Evaluate properties at the internal grid or an ordered list of query radii.

Query order is preserved and need not be increasing. A single property returns
one dimension; multiple properties return one row per property. Supply
radius_planet in metres when requesting depth.
)doc")
      .def_property_readonly("mass", &Layer::get_mass)
      .def_property_readonly("moment_of_inertia", &Layer::get_moment_of_inertia)
      .def("_evaluate_temperature", &Layer::evaluate_temperature,
           py::arg("pressures"), py::arg("temperature_top") = py::none())
      .def("_compute_gravity", &Layer::compute_gravity, py::arg("density"),
           py::arg("gravity_bottom"))
      .def("_compute_pressure", &Layer::compute_pressure, py::arg("density"),
           py::arg("gravity"), py::arg("pressure_top"))
      .def("__str__", &Layer::to_string);
  bind_radial_properties<Layer>(layer);

  auto perturbation = py::class_<BoundaryLayerPerturbation>(
      m, "BoundaryLayerPerturbation", R"doc(
Exponential temperature perturbations at the boundaries of a spherical layer.

Radii use m, temperature_change K, and gradients K/m. The exponential scale is
rayleigh_number**0.25. boundary_layer_ratio sets the top-to-bottom amplitude
ratio. Temperature and gradient queries accept scalars or arrays.
)doc");
  perturbation
      .def(py::init<double, double, double, double, double>(),
           py::arg("radius_bottom"), py::arg("radius_top"),
           py::arg("rayleigh_number"), py::arg("temperature_change"),
           py::arg("boundary_layer_ratio"))
      .def_readonly("r0", &BoundaryLayerPerturbation::r0)
      .def_readonly("r1", &BoundaryLayerPerturbation::r1)
      .def_readonly("Ra", &BoundaryLayerPerturbation::Ra)
      .def_readonly("c", &BoundaryLayerPerturbation::c)
      .def_property_readonly("a", &BoundaryLayerPerturbation::get_a)
      .def_property_readonly("b", &BoundaryLayerPerturbation::get_b)
      .def("set_model_thermal_gradients",
           &BoundaryLayerPerturbation::set_model_thermal_gradients,
           py::arg("dTdr_bottom"), py::arg("dTdr_top"));
  for (const bool gradient : {false, true}) {
    perturbation.def(
        gradient ? "dTdr" : "temperature",
        [gradient](const BoundaryLayerPerturbation &self,
                   const py::object &input) -> py::object {
          const auto array =
              py::array_t<double, py::array::c_style |
                                      py::array::forcecast>::ensure(input);
          if (!array)
            throw py::type_error("Radii must be numeric.");
          const Array values =
              Eigen::Map<const Array>(array.data(), array.size());
          const Array result =
              gradient ? self.dTdr(values) : self.temperature(values);
          if (!array.ndim())
            return py::float_(result[0]);
          const std::vector<py::ssize_t> shape(array.shape(),
                                               array.shape() + array.ndim());
          py::array_t<double> output(shape);
          if (result.size())
            std::memcpy(output.mutable_data(), result.data(),
                        static_cast<std::size_t>(result.size()) *
                            sizeof(double));
          return output;
        },
        py::arg("radii"));
  }
}
} // namespace burnman::python
