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

#include "burnman/core/seismic.hpp"
#include "bindings.hpp"
#include "burnman/tools/seismic.hpp"
#include <cstring>
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using namespace burnman::seismic;
using Array = Eigen::ArrayXd;

template <class Base> class PySeismicModel : public Base {
public:
  using Base::Base;
  Array internal_depth_list(double mindepth, double maxdepth,
                            double discontinuity_interval) const override {
    PYBIND11_OVERRIDE(Array, Base, internal_depth_list, mindepth, maxdepth,
                      discontinuity_interval);
  }
#define OVERRIDE_QUERY(name)                                                   \
  Array name(const Array &values) const override {                             \
    PYBIND11_OVERRIDE(Array, Base, name, values);                              \
  }
  OVERRIDE_QUERY(pressure)
  OVERRIDE_QUERY(v_p)
  OVERRIDE_QUERY(v_s)
  OVERRIDE_QUERY(v_phi)
  OVERRIDE_QUERY(density)
  OVERRIDE_QUERY(G)
  OVERRIDE_QUERY(K)
  OVERRIDE_QUERY(QK)
  OVERRIDE_QUERY(QG)
  OVERRIDE_QUERY(depth)
  OVERRIDE_QUERY(gravity)
#undef OVERRIDE_QUERY
};

template <class Base> class PySeismicTable : public PySeismicModel<Base> {
public:
  Array _lookup(const Array &depth, const Array &values) const override {
    PYBIND11_OVERRIDE(Array, Base, _lookup, depth, values);
  }
};

// Accept scalars or arbitrary-shaped arrays, as numpy.interp does. The native
// interface evaluates a flat array and the Python result retains the shape.
template <class Model, class Getter>
py::object query(const Model &model, Getter getter, const py::object &input) {
  const auto values =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          input);
  if (!values)
    throw py::type_error("Seismic coordinates must be numeric.");
  const Array x = Eigen::Map<const Array>(values.data(), values.size());
  const Array result = (model.*getter)(x);
  if (result.size() != x.size())
    throw py::value_error("Seismic query returned the wrong number of values.");
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

void bind_seismic(py::module_ &m) {
  auto module = m.def_submodule(
      "seismic", "One-dimensional seismic reference models (SI units).");
  py::class_<Seismic1DModel, PySeismicModel<Seismic1DModel>,
             std::shared_ptr<Seismic1DModel>>
      base(module, "Seismic1DModel");
  base.def(py::init<>())
      .def("internal_depth_list", &Seismic1DModel::internal_depth_list,
           py::arg("mindepth") = 0.0, py::arg("maxdepth") = 1.0e99,
           py::arg("discontinuity_interval") = 1.0)
      .def(
          "evaluate",
          [](const Seismic1DModel &self,
             const std::vector<std::string> &vars_list,
             const py::object &depth_list) {
            return depth_list.is_none()
                       ? self.evaluate(vars_list)
                       : self.evaluate(vars_list, depth_list.cast<Array>());
          },
          py::arg("vars_list"), py::arg("depth_list") = py::none());
#define BIND_QUERY(name, coordinate)                                           \
  base.def(                                                                    \
      #name,                                                                   \
      [](const Seismic1DModel &self, const py::object &values) {               \
        return query(                                                          \
            self,                                                              \
            static_cast<Array (Seismic1DModel::*)(const Array &) const>(       \
                &Seismic1DModel::name),                                        \
            values);                                                           \
      },                                                                       \
      py::arg(coordinate))
  BIND_QUERY(pressure, "depth");
  BIND_QUERY(v_p, "depth");
  BIND_QUERY(v_s, "depth");
  BIND_QUERY(v_phi, "depth");
  BIND_QUERY(density, "depth");
  BIND_QUERY(G, "depth");
  BIND_QUERY(K, "depth");
  BIND_QUERY(QK, "depth");
  BIND_QUERY(QG, "depth");
  BIND_QUERY(depth, "pressure");
  BIND_QUERY(gravity, "depth");
#undef BIND_QUERY

  py::class_<SeismicTable, Seismic1DModel, PySeismicTable<SeismicTable>,
             std::shared_ptr<SeismicTable>>
      table(module, "SeismicTable");
  table.def(py::init<>())
      .def("internal_depth_list", &SeismicTable::internal_depth_list,
           py::arg("mindepth") = 0.0, py::arg("maxdepth") = 1.0e10,
           py::arg("discontinuity_interval") = 1.0)
      .def(
          "radius",
          [](const SeismicTable &self, const py::object &values) {
            return query(
                self,
                static_cast<Array (SeismicTable::*)(const Array &) const>(
                    &SeismicTable::radius),
                values);
          },
          py::arg("pressure"))
      .def("bullen", &SeismicTable::bullen, py::arg("depth"))
      .def("_lookup", &SeismicTable::_lookup, py::arg("depth"),
           py::arg("value_table"))
      .def("_compute_gravity", &SeismicTable::_compute_gravity)
      .def("_compute_pressure", &SeismicTable::_compute_pressure)
      .def_readwrite("earth_radius", &SeismicTable::earth_radius);
#define BIND_TABLE(name) table.def_readwrite(#name, &SeismicTable::name)
  BIND_TABLE(table_depth);
  BIND_TABLE(table_radius);
  BIND_TABLE(table_pressure);
  BIND_TABLE(table_gravity);
  BIND_TABLE(table_density);
  BIND_TABLE(table_vp);
  BIND_TABLE(table_vs);
  BIND_TABLE(table_QG);
  BIND_TABLE(table_QK);
#undef BIND_TABLE

  py::class_<PREM, SeismicTable, PySeismicTable<PREM>, std::shared_ptr<PREM>>(
      module, "PREM")
      .def(py::init<>());
  py::class_<Slow, SeismicTable, PySeismicTable<Slow>, std::shared_ptr<Slow>>(
      module, "Slow")
      .def(py::init<>());
  py::class_<Fast, SeismicTable, PySeismicTable<Fast>, std::shared_ptr<Fast>>(
      module, "Fast")
      .def(py::init<>());
  py::class_<IASP91, SeismicTable, PySeismicTable<IASP91>,
             std::shared_ptr<IASP91>>(module, "IASP91")
      .def(py::init<>());
  py::class_<AK135, SeismicTable, PySeismicTable<AK135>,
             std::shared_ptr<AK135>>(module, "AK135")
      .def(py::init<>());
  py::class_<STW105, SeismicTable, PySeismicTable<STW105>,
             std::shared_ptr<STW105>>(module, "STW105")
      .def(py::init<>())
      .def_readwrite("table_vpv", &STW105::table_vpv)
      .def_readwrite("table_vsv", &STW105::table_vsv)
      .def_readwrite("table_vph", &STW105::table_vph)
      .def_readwrite("table_vsh", &STW105::table_vsh);
  module.attr("prem_model") =
      py::cast(&prem_model(), py::return_value_policy::reference);
  // numpy-style broadcasting, including scalar quality factors, is performed
  // by pybind11's vectorizer; the Python wrapper returns the upstream tuple.
  m.def("_attenuation_correction",
        py::vectorize(
            [](double vp, double vs, double vphi, double qs, double qphi) {
              return tools::seismic::attenuation_correction(vp, vs, vphi, qs,
                                                            qphi)[0];
            }));
  m.def("_attenuation_correction_s",
        py::vectorize(
            [](double vp, double vs, double vphi, double qs, double qphi) {
              return tools::seismic::attenuation_correction(vp, vs, vphi, qs,
                                                            qphi)[1];
            }));
  m.def("_attenuation_correction_phi",
        py::vectorize(
            [](double vp, double vs, double vphi, double qs, double qphi) {
              return tools::seismic::attenuation_correction(vp, vs, vphi, qs,
                                                            qphi)[2];
            }));
}
} // namespace burnman::python
