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

#ifndef BURNMAN_UTILS_ANISOTROPY_HPP_INCLUDED
#define BURNMAN_UTILS_ANISOTROPY_HPP_INCLUDED

#include <Eigen/Core>
#include <array>
#include <utility>
#include <vector>

namespace burnman::types {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using ElasticTensor =
    std::array<std::array<std::array<std::array<double, 3>, 3>, 3>, 3>;
} // namespace burnman::types

namespace burnman::utils {
types::Vector6d contract_stresses(const Eigen::Matrix3d &tensor);
types::Vector6d contract_strains(const Eigen::Matrix3d &tensor);
Eigen::Matrix3d expand_stresses(const types::Vector6d &voigt);
Eigen::Matrix3d expand_strains(const types::Vector6d &voigt);
/// Voigt order: 11, 22, 33, 23, 13, 12 (indices are zero based).
std::pair<std::size_t, std::size_t> voigt_index_to_ij(int m);
types::ElasticTensor
voigt_notation_to_stiffness_tensor(const types::Matrix6d &voigt);
/// Engineering shear compliance requires factors of two on each shear index.
types::ElasticTensor
voigt_notation_to_compliance_tensor(const types::Matrix6d &voigt);
types::Matrix6d contract_compliances(const types::ElasticTensor &tensor);
types::Matrix6d contract_stiffnesses(const types::ElasticTensor &tensor);
types::ElasticTensor rotate_elastic_tensor(const types::ElasticTensor &tensor,
                                           const Eigen::Matrix3d &rotation);
types::Matrix6d voigt_array_from_cijs(
    const std::vector<double> &cijs,
    const std::vector<std::vector<std::pair<int, int>>> &index_lists);
} // namespace burnman::utils

#endif
