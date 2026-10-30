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
#include "burnman/utils/polygon_utils.hpp"
#include "burnman/utils/unitcell.hpp"

namespace burnman::python {
void bind_utils(py::module_ &m) {
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
