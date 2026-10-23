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

#include "burnman/core/geotherm.hpp"
#include "bindings.hpp"
#include <cstring>
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using namespace burnman::geotherm;
using Array = Eigen::ArrayXd;

template <class Base> class PyGeotherm : public Base {
public:
  using Base::Base;
  Array temperatures(const Array &depths) const override {
    PYBIND11_OVERRIDE(Array, Base, temperatures, depths);
  }
};
template <class Base> class PyPressureGeotherm : public PyGeotherm<Base> {
public:
  using PyGeotherm<Base>::PyGeotherm;
  Array temperatures_from_pressures(const Array &pressures) const override {
    PYBIND11_OVERRIDE(Array, Base, temperatures_from_pressures, pressures);
  }
};

// Retain the Python object as well as its pressure(depths) method. This accepts
// arbitrary pressure models and keeps Python seismic subclasses alive while
// a geotherm refers to them, even after their external references are deleted.
class PythonPressureModel : public seismic::Seismic1DModel {
public:
  explicit PythonPressureModel(py::object object)
      : object_(std::move(object)) {}
  Array pressure(const Array &depths) const override {
    py::gil_scoped_acquire gil;
    return object_.attr("pressure")(depths).cast<Array>();
  }
  const py::object &object() const { return object_; }

private:
  py::object object_;
};

std::shared_ptr<seismic::Seismic1DModel>
pressure_model(const py::object &object) {
  if (object.is_none())
    return nullptr;
  if (!py::hasattr(object, "pressure") ||
      !PyCallable_Check(object.attr("pressure").ptr()))
    throw py::type_error(
        "depth_to_pressure_model must provide a pressure(depths) method.");
  return std::make_shared<PythonPressureModel>(object);
}

// Like the seismic bindings, queries accept scalar or arbitrary-shaped numeric
// inputs and always return floating-point values with the original shape.
template <class Getter>
py::object query(const py::object &input, Getter getter) {
  const auto values =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          input);
  if (!values)
    throw py::type_error("Geotherm coordinates must be numeric.");
  const Array x = Eigen::Map<const Array>(values.data(), values.size());
  const Array result = getter(x);
  if (result.size() != x.size())
    throw py::value_error(
        "Geotherm query returned the wrong number of temperatures.");
  if (values.ndim() == 0)
    return py::float_(result[0]);
  const std::vector<py::ssize_t> shape(values.shape(),
                                       values.shape() + values.ndim());
  py::array_t<double> output(shape);
  if (result.size() > 0)
    std::memcpy(output.mutable_data(), result.data(),
                static_cast<std::size_t>(result.size()) * sizeof(double));
  return output;
}
} // namespace

void bind_geotherm(py::module_ &m) {
  auto module = m.def_submodule(
      "geotherm", "Tabulated geotherms and material adiabats (SI units).");
  const py::object prem = m.attr("seismic").attr("prem_model");
  py::class_<Geotherm, PyGeotherm<Geotherm>, std::shared_ptr<Geotherm>> base(
      module, "Geotherm");
  base.def(py::init<const Array &, const Array &>(), py::arg("depths_array"),
           py::arg("temperatures_array"))
      .def(
          "temperatures",
          [](const Geotherm &self, const py::object &depths) {
            return query(depths, [&self](const Array &x) {
              return self.temperatures(x);
            });
          },
          py::arg("depths"))
      .def_readwrite("depths_array", &Geotherm::depths_array)
      .def_readwrite("temperatures_array", &Geotherm::temperatures_array)
      .def_readwrite("minimum_depth", &Geotherm::minimum_depth)
      .def_readwrite("maximum_depth", &Geotherm::maximum_depth);

  py::class_<BrownShankland, Geotherm, PyGeotherm<BrownShankland>,
             std::shared_ptr<BrownShankland>>(module, "BrownShankland")
      .def(py::init<>());
  py::class_<Anderson, Geotherm, PyGeotherm<Anderson>,
             std::shared_ptr<Anderson>>(module, "Anderson")
      .def(py::init<>());
  py::class_<StaceyContinental, Geotherm, PyGeotherm<StaceyContinental>,
             std::shared_ptr<StaceyContinental>>(module, "StaceyContinental")
      .def(py::init<>());
  py::class_<StaceyOceanic, Geotherm, PyGeotherm<StaceyOceanic>,
             std::shared_ptr<StaceyOceanic>>(module, "StaceyOceanic")
      .def(py::init<>());
  py::class_<Katsura2022, Geotherm, PyGeotherm<Katsura2022>,
             std::shared_ptr<Katsura2022>>(module, "Katsura2022")
      .def(py::init<>());
  py::class_<Plesa2022Mars6cm3, Geotherm, PyGeotherm<Plesa2022Mars6cm3>,
             std::shared_ptr<Plesa2022Mars6cm3>>(module, "Plesa2022Mars6cm3")
      .def(py::init<>());

  py::class_<GeothermFromPressures, Geotherm,
             PyPressureGeotherm<GeothermFromPressures>,
             std::shared_ptr<GeothermFromPressures>>
      pressure(module, "GeothermFromPressures");
  pressure
      .def(py::init([](const Array &pressures, const Array &temperatures,
                       const py::object &model) {
             return std::make_shared<PyPressureGeotherm<GeothermFromPressures>>(
                 pressures, temperatures, pressure_model(model));
           }),
           py::arg("pressures_array"), py::arg("temperatures_array"),
           py::arg("depth_to_pressure_model") = prem)
      .def(
          "temperatures_from_pressures",
          [](const GeothermFromPressures &self, const py::object &pressures) {
            return query(pressures, [&self](const Array &x) {
              return self.temperatures_from_pressures(x);
            });
          },
          py::arg("pressures"))
      .def_readwrite("pressures_array", &GeothermFromPressures::pressures_array)
      .def_readwrite("minimum_pressure",
                     &GeothermFromPressures::minimum_pressure)
      .def_readwrite("maximum_pressure",
                     &GeothermFromPressures::maximum_pressure)
      .def_property(
          "depth_to_pressure_model",
          [](const GeothermFromPressures &self) -> py::object {
            if (const auto adapter =
                    std::dynamic_pointer_cast<PythonPressureModel>(
                        self.depth_to_pressure_model))
              return adapter->object();
            return py::cast(self.depth_to_pressure_model);
          },
          [](GeothermFromPressures &self, const py::object &model) {
            if (model.is_none())
              throw py::type_error("depth_to_pressure_model must provide a "
                                   "pressure(depths) method.");
            self.depth_to_pressure_model = pressure_model(model);
          });

  py::class_<Anzellini2013, GeothermFromPressures,
             PyPressureGeotherm<Anzellini2013>, std::shared_ptr<Anzellini2013>>(
      module, "Anzellini2013")
      .def(py::init([](const py::object &model) {
             return std::make_shared<PyPressureGeotherm<Anzellini2013>>(
                 pressure_model(model));
           }),
           py::arg("depth_to_pressure_model") = prem);
  py::class_<AdiabaticGeotherm, GeothermFromPressures,
             PyPressureGeotherm<AdiabaticGeotherm>,
             std::shared_ptr<AdiabaticGeotherm>>(module, "AdiabaticGeotherm")
      .def(py::init([](std::shared_ptr<Material> material, double T_0,
                       double P_0, const py::object &model) {
             return std::make_shared<PyPressureGeotherm<AdiabaticGeotherm>>(
                 std::move(material), T_0, P_0, pressure_model(model));
           }),
           py::arg("material"), py::arg("T_0"), py::arg("P_0"),
           py::arg("depth_to_pressure_model") = prem)
      .def_readwrite("material", &AdiabaticGeotherm::material)
      .def_readwrite("T_0", &AdiabaticGeotherm::T_0)
      .def_readwrite("P_0", &AdiabaticGeotherm::P_0);

  module.def(
      "adiabatic_profile",
      [](const py::object &pressures, Material &rock, double T_0,
         std::optional<double> P_0) {
        return query(pressures, [&](const Array &x) {
          return adiabatic_profile(x, rock, T_0, P_0);
        });
      },
      py::arg("pressures"), py::arg("rock"), py::arg("T_0"),
      py::arg("P_0") = py::none());
}
} // namespace burnman::python
