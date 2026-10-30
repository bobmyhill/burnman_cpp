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

#ifndef BURNMAN_UTILS_UNITCELL_HPP_INCLUDED
#define BURNMAN_UTILS_UNITCELL_HPP_INCLUDED

#include <Eigen/Core>
#include <array>

namespace burnman::utils {
using FrameConvention = std::array<int, 3>;
using CellParameters = Eigen::Matrix<double, 6, 1>;

/// Molar cell lengths in metres; angles alpha, beta, gamma in degrees.
Eigen::Matrix3d
cell_parameters_to_vectors(const CellParameters &parameters,
                           const FrameConvention &frame_convention = {0, 1, 2});
Eigen::Matrix3d rotation_to_crystallographic_frame(
    const Eigen::Matrix3d &cell_vectors,
    const FrameConvention &frame_convention = {0, 1, 2});
CellParameters
cell_vectors_to_parameters(const Eigen::Matrix3d &vectors,
                           const FrameConvention &frame_convention = {0, 1, 2},
                           bool rotate = false);
} // namespace burnman::utils

#endif
