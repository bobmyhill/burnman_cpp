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

#include "bindings.hpp"
#include "burnman/utils/anisotropy.hpp"
#include "burnman/utils/polygon_utils.hpp"
#include "burnman/utils/unitcell.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
types::ElasticTensor unpack_tensor(const Array &array) {
  if (array.ndim() != 4 || array.shape(0) != 3 || array.shape(1) != 3 ||
      array.shape(2) != 3 || array.shape(3) != 3)
    throw std::invalid_argument("Elastic tensors require shape (3,3,3,3).");
  const auto input = array.unchecked<4>();
  types::ElasticTensor tensor{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          tensor[i][j][k][l] = input(i, j, k, l);
  return tensor;
}
py::array_t<double> pack_tensor(const types::ElasticTensor &tensor) {
  py::array_t<double> result({3, 3, 3, 3});
  auto out = result.mutable_unchecked<4>();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          out(i, j, k, l) = tensor[i][j][k][l];
  return result;
}
} // namespace
void bind_utils(py::module_ &m) {
  m.def("polar_decomposition", &utils::polar_decomposition, py::arg("F"),
        py::arg("strain_measure") = "U");
  m.def("contract_stresses", &utils::contract_stresses, py::arg("tensor"));
  m.def("contract_strains", &utils::contract_strains, py::arg("tensor"));
  m.def("expand_stresses", &utils::expand_stresses, py::arg("voigt"));
  m.def("expand_strains", &utils::expand_strains, py::arg("voigt"));
  m.def(
      "voigt_notation_to_stiffness_tensor",
      [](const types::Matrix6d &voigt) {
        return pack_tensor(utils::voigt_notation_to_stiffness_tensor(voigt));
      },
      py::arg("voigt"));
  m.def(
      "voigt_notation_to_compliance_tensor",
      [](const types::Matrix6d &voigt) {
        return pack_tensor(utils::voigt_notation_to_compliance_tensor(voigt));
      },
      py::arg("voigt"));
  m.def(
      "contract_stiffnesses",
      [](const Array &tensor) {
        return utils::contract_stiffnesses(unpack_tensor(tensor));
      },
      py::arg("tensor"));
  m.def(
      "contract_compliances",
      [](const Array &tensor) {
        return utils::contract_compliances(unpack_tensor(tensor));
      },
      py::arg("tensor"));
  m.def("voigt_index_to_ij", &utils::voigt_index_to_ij, py::arg("m"));
  m.def("voigt_array_from_cijs", &utils::voigt_array_from_cijs, py::arg("cijs"),
        py::arg("index_lists"));
  m.def("cell_parameters_to_vectors", &utils::cell_parameters_to_vectors,
        py::arg("cell_parameters"),
        py::arg("frame_convention") = utils::FrameConvention{0, 1, 2});
  m.def("rotation_to_crystallographic_frame",
        &utils::rotation_to_crystallographic_frame, py::arg("cell_vectors"),
        py::arg("frame_convention") = utils::FrameConvention{0, 1, 2});
  m.def("cell_vectors_to_parameters", &utils::cell_vectors_to_parameters,
        py::arg("vectors"),
        py::arg("frame_convention") = utils::FrameConvention{0, 1, 2},
        py::arg("rotate") = false);
  m.def(
      "visual_center_of_polygon",
      [](const std::vector<Eigen::MatrixXd> &rings, double precision,
         const std::vector<Eigen::Vector2d> &seeds) {
        const auto centre =
            utils::visual_center_of_polygon(rings, precision, seeds);
        return py::make_tuple(centre.point, centre.distance);
      },
      py::arg("rings"), py::arg("precision") = 0.,
      py::arg("seeds") = std::vector<Eigen::Vector2d>{},
      "Pole of inaccessibility of a polygon given as an exterior ring "
      "followed by holes (N x 2 arrays). Returns (point, distance to the "
      "nearest edge). A non-positive precision selects 0.2% of the shorter "
      "bounding-box side.");
  m.def(
      "signed_distance_to_polygon",
      [](const Eigen::Vector2d &point,
         const std::vector<Eigen::MatrixXd> &rings) {
        return utils::signed_distance_to_polygon(point, rings);
      },
      py::arg("point"), py::arg("rings"),
      "Distance to the nearest ring edge: positive inside, negative outside.");
}
} // namespace burnman::python
