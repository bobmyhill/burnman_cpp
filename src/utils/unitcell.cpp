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

#include "burnman/utils/unitcell.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace burnman::utils {
namespace {
void check_frame(FrameConvention frame) {
  std::sort(frame.begin(), frame.end());
  if (frame != FrameConvention{0, 1, 2})
    throw std::invalid_argument(
        "Frame convention must be a permutation of 0,1,2.");
}
constexpr double radians = 0.017453292519943295;
} // namespace

Eigen::Matrix3d cell_parameters_to_vectors(const CellParameters &parameters,
                                           const FrameConvention &c) {
  check_frame(c);
  if (!parameters.allFinite() || (parameters.head<3>().array() <= 0.).any() ||
      (parameters.tail<3>().array() <= 0.).any() ||
      (parameters.tail<3>().array() >= 180.).any())
    throw std::invalid_argument(
        "Require positive finite cell lengths and angles in (0,180) degrees.");
  const Eigen::Vector3d angles = parameters.tail<3>() * radians;
  const double n2 = (std::cos(angles[c[0]]) -
                     std::cos(angles[c[2]]) * std::cos(angles[c[1]])) /
                    std::sin(angles[c[2]]);
  const double height2 = std::pow(std::sin(angles[c[1]]), 2) - n2 * n2;
  if (height2 <= 0.)
    throw std::invalid_argument(
        "Cell angles must define a nondegenerate cell.");
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  result(c[0], c[0]) = parameters[c[0]];
  result(c[1], c[0]) = parameters[c[1]] * std::cos(angles[c[2]]);
  result(c[1], c[1]) = parameters[c[1]] * std::sin(angles[c[2]]);
  result(c[2], c[0]) = parameters[c[2]] * std::cos(angles[c[1]]);
  result(c[2], c[1]) = parameters[c[2]] * n2;
  result(c[2], c[2]) = parameters[c[2]] * std::sqrt(height2);
  return result;
}

Eigen::Matrix3d
rotation_to_crystallographic_frame(const Eigen::Matrix3d &vectors,
                                   const FrameConvention &c) {
  check_frame(c);
  if (!vectors.allFinite() || vectors.row(c[0]).norm() == 0. ||
      vectors.row(c[0]).cross(vectors.row(c[1])).norm() == 0.)
    throw std::invalid_argument("Cell vectors must be finite and independent.");
  Eigen::Matrix3d result;
  result.row(c[0]) = vectors.row(c[0]).normalized();
  result.row(c[2]) = vectors.row(c[0]).cross(vectors.row(c[1])).normalized();
  result.row(c[1]) = result.row(c[2]).cross(result.row(c[0]));
  return result;
}

CellParameters cell_vectors_to_parameters(const Eigen::Matrix3d &vectors,
                                          const FrameConvention &c,
                                          bool rotate) {
  check_frame(c);
  const Eigen::Matrix3d cell =
      rotate ? Eigen::Matrix3d(
                   vectors *
                   rotation_to_crystallographic_frame(vectors, c).transpose())
             : vectors;
  const double tolerance = 100. * std::numeric_limits<double>::epsilon() *
                           cell.cwiseAbs().maxCoeff();
  if (!cell.allFinite() || std::abs(cell(c[0], c[1])) > tolerance ||
      std::abs(cell(c[0], c[2])) > tolerance ||
      std::abs(cell(c[1], c[2])) > tolerance)
    throw std::invalid_argument(
        "Cell vectors do not satisfy the frame convention.");
  CellParameters result;
  for (int i = 0; i < 3; ++i) {
    result[i] = cell.row(i).norm();
    if (result[i] == 0.)
      throw std::invalid_argument("Cell lengths must be positive.");
  }
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const double cosine =
        cell.row(j).dot(cell.row(k)) / (result[j] * result[k]);
    result[i + 3] = std::acos(std::clamp(cosine, -1., 1.)) / radians;
  }
  return result;
}
} // namespace burnman::utils
