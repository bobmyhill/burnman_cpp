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

#include "burnman/core/solution_models/symmetric_regular_solution.hpp"

namespace burnman::solution_models {

// SymmetricRegularSolution Constructor
SymmetricRegularSolution::SymmetricRegularSolution(
    const types::PairedEndmemberList &endmember_list,
    std::vector<std::vector<double>> energy_interaction,
    std::vector<std::vector<double>> volume_interaction,
    std::vector<std::vector<double>> entropy_interaction)
    : AsymmetricRegularSolution(
          endmember_list, std::vector<double>(endmember_list.size(), 1.0),
          energy_interaction, volume_interaction, entropy_interaction) {}

} // namespace burnman::solution_models
