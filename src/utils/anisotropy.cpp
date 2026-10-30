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

#include "burnman/utils/anisotropy.hpp"
#include <stdexcept>

namespace burnman::utils {
types::Vector6d contract_stresses(const Eigen::Matrix3d &tensor) {
  types::Vector6d result;
  result << tensor(0, 0), tensor(1, 1), tensor(2, 2), tensor(1, 2),
      tensor(0, 2), tensor(0, 1);
  return result;
}
types::Vector6d contract_strains(const Eigen::Matrix3d &tensor) {
  auto result = contract_stresses(tensor);
  result.tail<3>() *= 2.;
  return result;
}
Eigen::Matrix3d expand_stresses(const types::Vector6d &voigt) {
  Eigen::Matrix3d result;
  result << voigt[0], voigt[5], voigt[4], voigt[5], voigt[1], voigt[3],
      voigt[4], voigt[3], voigt[2];
  return result;
}
Eigen::Matrix3d expand_strains(const types::Vector6d &voigt) {
  types::Vector6d result = voigt;
  result.tail<3>() *= .5;
  return expand_stresses(result);
}
std::pair<std::size_t, std::size_t> voigt_index_to_ij(int m) {
  static constexpr std::pair<std::size_t, std::size_t> indices[] = {
      {0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
  if (m < 0 || m >= 6)
    throw std::invalid_argument("Voigt indices must lie between 0 and 5.");
  return indices[m];
}

types::ElasticTensor
voigt_notation_to_stiffness_tensor(const types::Matrix6d &voigt) {
  types::ElasticTensor tensor{};
  for (int m = 0; m < 6; ++m) {
    const auto [i, j] = voigt_index_to_ij(m);
    for (int n = 0; n < 6; ++n) {
      const auto [k, l] = voigt_index_to_ij(n);
      tensor[i][j][k][l] = tensor[j][i][k][l] = tensor[i][j][l][k] =
          tensor[j][i][l][k] = voigt(m, n);
    }
  }
  return tensor;
}

types::ElasticTensor
voigt_notation_to_compliance_tensor(const types::Matrix6d &voigt) {
  types::Matrix6d compliance = voigt;
  compliance.bottomRows<3>() *= 0.5;
  compliance.rightCols<3>() *= 0.5;
  return voigt_notation_to_stiffness_tensor(compliance);
}

types::Matrix6d contract_stiffnesses(const types::ElasticTensor &tensor) {
  types::Matrix6d result;
  for (int m = 0; m < 6; ++m) {
    const auto [i, j] = voigt_index_to_ij(m);
    for (int n = 0; n < 6; ++n) {
      const auto [k, l] = voigt_index_to_ij(n);
      result(m, n) = tensor[i][j][k][l];
    }
  }
  return result;
}

types::Matrix6d contract_compliances(const types::ElasticTensor &tensor) {
  auto result = contract_stiffnesses(tensor);
  result.bottomRows<3>() *= 2.;
  result.rightCols<3>() *= 2.;
  return result;
}

types::ElasticTensor rotate_elastic_tensor(const types::ElasticTensor &tensor,
                                           const Eigen::Matrix3d &rotation) {
  // Rotate one index at a time, keeping the contraction small.
  auto result = tensor;
  for (int axis = 0; axis < 4; ++axis) {
    const auto input = result;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t k = 0; k < 3; ++k)
          for (std::size_t l = 0; l < 3; ++l) {
            std::array<std::size_t, 4> indices{i, j, k, l};
            const auto row = indices[static_cast<std::size_t>(axis)];
            double value = 0.;
            for (std::size_t p = 0; p < 3; ++p) {
              indices[static_cast<std::size_t>(axis)] = p;
              value += rotation(static_cast<Eigen::Index>(row),
                                static_cast<Eigen::Index>(p)) *
                       input[indices[0]][indices[1]][indices[2]][indices[3]];
            }
            result[i][j][k][l] = value;
          }
  }
  return result;
}

types::Matrix6d voigt_array_from_cijs(
    const std::vector<double> &cijs,
    const std::vector<std::vector<std::pair<int, int>>> &index_lists) {
  if (cijs.size() != index_lists.size())
    throw std::invalid_argument("Incorrect number of independent Cijs.");
  types::Matrix6d result = types::Matrix6d::Zero();
  for (std::size_t n = 0; n < cijs.size(); ++n)
    for (const auto &[i, j] : index_lists[n]) {
      if (i < 0 || i >= 6 || j < 0 || j >= 6)
        throw std::invalid_argument("Voigt indices must lie between 0 and 5.");
      result(i, j) = result(j, i) = cijs[n];
    }
  return result;
}
} // namespace burnman::utils
