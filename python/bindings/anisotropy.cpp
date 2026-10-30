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

#include "burnman/core/anisotropy.hpp"
#include "bindings.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using Directions =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<py::ssize_t> direction_shape(const Directions &directions) {
  if (directions.ndim() == 0 || directions.shape(directions.ndim() - 1) != 3)
    throw std::invalid_argument("Directions must have shape (..., 3).");
  return {directions.shape(), directions.shape() + directions.ndim() - 1};
}

py::array_t<double> full_tensor(const types::ElasticTensor &tensor) {
  py::array_t<double> result({3, 3, 3, 3});
  auto out = result.mutable_unchecked<4>();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          out(i, j, k, l) = tensor[i][j][k][l];
  return result;
}

template <typename Function>
py::object directional_scalar(Directions a, std::optional<Directions> b,
                              Function function) {
  direction_shape(a);
  if (b) {
    direction_shape(*b);
    const py::sequence arrays =
        py::module_::import("numpy").attr("broadcast_arrays")(a, *b);
    a = Directions::ensure(arrays[0]);
    b = Directions::ensure(arrays[1]);
  }
  py::array_t<double> result(direction_shape(a));
  for (py::ssize_t i = 0; i < a.size() / 3; ++i) {
    const Eigen::Vector3d first =
        Eigen::Map<const Eigen::Vector3d>(a.data() + 3 * i);
    const Eigen::Vector3d second =
        b ? Eigen::Vector3d(
                Eigen::Map<const Eigen::Vector3d>(b->data() + 3 * i))
          : first;
    result.mutable_data()[i] = function(first, second);
  }
  if (result.ndim() == 0)
    return py::float_(*result.data());
  return std::move(result);
}

template <typename Class>
void bind_symmetry(py::module_ &m, const char *name, const char *doc) {
  py::class_<Class, AnisotropicMaterial, std::shared_ptr<Class>>(m, name, doc)
      .def(py::init<double, const std::vector<double> &>(), py::arg("rho"),
           py::arg("cijs"));
}
} // namespace

void bind_anisotropy(py::module_ &m) {
  auto material = py::class_<AnisotropicMaterial, Material,
                             std::shared_ptr<AnisotropicMaterial>>(
      m, "AnisotropicMaterial",
      "Elastic material with density (kg/m^3) and symmetric 6x6 isentropic "
      "stiffness (Pa), in Voigt order 11,22,33,23,13,12.");
  material.def(py::init<double, const types::Matrix6d &>(), py::arg("rho"),
               py::arg("cijs"));
#define PROPERTY(name)                                                         \
  bind_property<AnisotropicMaterial>(material, #name, "get_" #name,            \
                                     &AnisotropicMaterial::get_##name)
  PROPERTY(density);
  PROPERTY(isentropic_stiffness_tensor);
  PROPERTY(isentropic_compliance_tensor);
  PROPERTY(isentropic_bulk_modulus_voigt);
  PROPERTY(isentropic_bulk_modulus_reuss);
  PROPERTY(isentropic_bulk_modulus_vrh);
  PROPERTY(isentropic_shear_modulus_voigt);
  PROPERTY(isentropic_shear_modulus_reuss);
  PROPERTY(isentropic_shear_modulus_vrh);
  PROPERTY(isentropic_universal_elastic_anisotropy);
  PROPERTY(isentropic_isotropic_poisson_ratio);
#undef PROPERTY
  auto stiffness = [](const AnisotropicMaterial &self) {
    return full_tensor(self.get_full_isentropic_stiffness_tensor());
  };
  auto compliance = [](const AnisotropicMaterial &self) {
    return full_tensor(self.get_full_isentropic_compliance_tensor());
  };
  material.def_property_readonly("full_isentropic_stiffness_tensor", stiffness)
      .def("get_full_isentropic_stiffness_tensor", stiffness)
      .def_property_readonly("full_isentropic_compliance_tensor", compliance)
      .def("get_full_isentropic_compliance_tensor", compliance);

  auto bind_direction = [&](const char *name, auto method) {
    material.def(
        name,
        [method](const AnisotropicMaterial &self, Directions d) {
          return directional_scalar(
              d, std::nullopt,
              [&](const Eigen::Vector3d &a, const Eigen::Vector3d &) {
                return (self.*method)(a);
              });
        },
        py::arg("direction"));
  };
  bind_direction("isentropic_linear_compressibility",
                 &AnisotropicMaterial::isentropic_linear_compressibility);
  bind_direction("isentropic_youngs_modulus",
                 &AnisotropicMaterial::isentropic_youngs_modulus);
  auto bind_pair = [&](const char *name, auto method, const char *first,
                       const char *second) {
    material.def(
        name,
        [method](const AnisotropicMaterial &self, Directions a, Directions b) {
          return directional_scalar(
              a, b, [&](const Eigen::Vector3d &x, const Eigen::Vector3d &y) {
                return (self.*method)(x, y);
              });
        },
        py::arg(first), py::arg(second));
  };
  bind_pair("isentropic_shear_modulus",
            &AnisotropicMaterial::isentropic_shear_modulus, "plane_normal",
            "shear_direction");
  bind_pair("isentropic_poissons_ratio",
            &AnisotropicMaterial::isentropic_poissons_ratio, "axial_direction",
            "lateral_direction");
  material.def(
      "christoffel_tensor",
      [](const AnisotropicMaterial &self, const Directions &directions) {
        auto shape = direction_shape(directions);
        shape.insert(shape.end(), {3, 3});
        py::array_t<double> result(shape);
        for (py::ssize_t i = 0; i < directions.size() / 3; ++i) {
          const auto tensor = self.christoffel_tensor(
              Eigen::Map<const Eigen::Vector3d>(directions.data() + 3 * i));
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
              result.mutable_data() + 9 * i) = tensor;
        }
        return result;
      },
      py::arg("propagation_direction"));
  material.def(
      "wave_velocities",
      [](const AnisotropicMaterial &self, const Directions &directions) {
        auto shape = direction_shape(directions);
        shape.push_back(3);
        py::array_t<double> velocities(shape);
        shape.push_back(3);
        py::array_t<double> polarizations(shape);
        for (py::ssize_t i = 0; i < directions.size() / 3; ++i) {
          const auto [v, p] = self.wave_velocities(
              Eigen::Map<const Eigen::Vector3d>(directions.data() + 3 * i));
          Eigen::Map<Eigen::Vector3d>(velocities.mutable_data() + 3 * i) = v;
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
              polarizations.mutable_data() + 9 * i) = p;
        }
        return py::make_tuple(velocities, polarizations);
      },
      py::arg("propagation_direction"),
      "Return descending wave speeds and matching polarization columns.");

  bind_symmetry<IsotropicMaterial>(m, "IsotropicMaterial", "Cijs: [C12, C44].");
  bind_symmetry<CubicMaterial>(m, "CubicMaterial", "Cijs: [C11, C12, C44].");
  bind_symmetry<HexagonalMaterial>(m, "HexagonalMaterial",
                                   "Cijs: [C11, C12, C13, C33, C44].");
  bind_symmetry<TetragonalMaterial>(
      m, "TetragonalMaterial", "Cijs: [C11, C12, C13, (C16), C33, C44, C66].");
  bind_symmetry<RhombohedralMaterial>(
      m, "RhombohedralMaterial",
      "Cijs: [C11, C12, C13, C14, (C15), C33, C44, C66].");
  bind_symmetry<OrthorhombicMaterial>(
      m, "OrthorhombicMaterial",
      "Cijs: [C11, C12, C13, C22, C23, C33, C44, C55, C66].");
  bind_symmetry<MonoclinicMaterial>(m, "MonoclinicMaterial",
                                    "Cijs: [C11, C12, C13, C15, C22, C23, C25, "
                                    "C33, C35, C44, C46, C55, C66].");
  bind_symmetry<TriclinicMaterial>(
      m, "TriclinicMaterial",
      "Cijs: upper triangular entries in row order (21 coefficients).");
}
} // namespace burnman::python
