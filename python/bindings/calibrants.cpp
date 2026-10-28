// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "bindings.hpp"
#include "burnman/calibrants/tools.hpp"
#include "burnman/core/calibrant.hpp"
#include "calibrant_catalogue.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
Eigen::Matrix2d calibrant_covariance(const py::object &value) {
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!array || array.ndim() != 2 || array.shape(0) != 2 || array.shape(1) != 2)
    throw py::value_error("Calibrant input covariance must be a 2x2 matrix.");
  Eigen::Matrix2d C;
  for (py::ssize_t i = 0; i < 2; ++i)
    for (py::ssize_t j = 0; j < 2; ++j)
      C(i, j) = array.at(i, j);
  return C;
}
} // namespace

void bind_calibrants(py::module_ &m) {
  py::class_<Calibrant>(m, "Calibrant")
      .def(py::init<Calibrant::Function, const std::string &,
                    Calibrant::Params>(),
           py::arg("calibrant_function"),
           py::arg("calibrant_function_return_type"), py::arg("params"))
      .def_readwrite("params", &Calibrant::params)
      .def_property_readonly("calibrant_function_return_type",
                             &Calibrant::return_type)
      .def("calibrant_function", &Calibrant::calibrant_function,
           py::arg("value"), py::arg("temperature"), py::arg("params"))
      .def("pressure_function", &Calibrant::pressure_function,
           py::arg("volume"), py::arg("temperature"), py::arg("params"))
      .def("volume_function", &Calibrant::volume_function, py::arg("pressure"),
           py::arg("temperature"), py::arg("params"))
      .def(
          "pressure",
          [](const Calibrant &self, double v, double t,
             const py::object &C) -> py::object {
            if (C.is_none())
              return py::float_(self.pressure(v, t));
            const auto [p, cov] = self.pressure(v, t, calibrant_covariance(C));
            return py::make_tuple(p, cov);
          },
          py::arg("volume"), py::arg("temperature"),
          py::arg("VT_covariance") = py::none())
      .def(
          "volume",
          [](const Calibrant &self, double p, double t,
             const py::object &C) -> py::object {
            if (C.is_none())
              return py::float_(self.volume(p, t));
            const auto [v, cov] = self.volume(p, t, calibrant_covariance(C));
            return py::make_tuple(v, cov);
          },
          py::arg("pressure"), py::arg("temperature"),
          py::arg("PT_covariance") = py::none());
  auto calibrants =
      m.def_submodule("calibrants", "Published pressure calibrants.");
  bind_calibrant_catalogue(calibrants);
  auto tool = calibrants.def_submodule("tools");
  tool.def(
      "pressure_to_pressure",
      [](const Calibrant &old_calibrant, const Calibrant &new_calibrant,
         double p, double t, const py::object &C) -> py::object {
        if (C.is_none())
          return py::float_(calibrants::pressure_to_pressure(
              old_calibrant, new_calibrant, p, t));
        const auto [pressure, cov] = calibrants::pressure_to_pressure(
            old_calibrant, new_calibrant, p, t, calibrant_covariance(C));
        return py::make_tuple(pressure, cov);
      },
      py::arg("old_calibrant"), py::arg("new_calibrant"), py::arg("pressure"),
      py::arg("temperature"), py::arg("PT_covariance") = py::none());
}
} // namespace burnman::python
