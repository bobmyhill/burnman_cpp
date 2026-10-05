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

#ifndef BURNMAN_CORE_SOLUTION_MODELS_SYMMETRIC_REGULAR_HPP_INCLUDED
#define BURNMAN_CORE_SOLUTION_MODELS_SYMMETRIC_REGULAR_HPP_INCLUDED

#include "burnman/core/solution_models/asymmetric_regular_solution.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include <vector>

namespace burnman {
namespace solution_models {

/**
 * @class SymmetricRegularSolution
 * @brief Convenience class for a symmetric regular solution model.
 *
 * This class is a special case of `AsymmetricRegularSolution' with all
 * alphas set to 1.
 *
 */
class SymmetricRegularSolution : public AsymmetricRegularSolution {
public:
  SymmetricRegularSolution(
      const types::PairedEndmemberList &endmember_list,
      std::vector<std::vector<double>> energy_interaction,
      std::vector<std::vector<double>> volume_interaction = {},
      std::vector<std::vector<double>> entropy_interaction = {});
};

} // namespace solution_models
} // namespace burnman

#endif // BURNMAN_CORE_SOLUTION_MODELS_SYMMETRIC_REGULAR_HPP_INCLUDED
