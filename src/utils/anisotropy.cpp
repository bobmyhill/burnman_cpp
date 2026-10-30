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
