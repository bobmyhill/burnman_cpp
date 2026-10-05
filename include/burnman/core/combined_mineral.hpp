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

#pragma once

#include "burnman/core/mineral.hpp"
#include <Eigen/Dense>
#include <vector>

namespace burnman {
/**
 * Construct a mineral from a signed linear combination of endmembers.
 * energy_adjustment is [delta_E (J/mol), delta_S (J/K/mol), delta_V (m^3/mol)].
 * The custom EOS remains valid when Mineral values are copied into solutions.
 */
Mineral make_combined_mineral(
    const std::vector<Mineral> &minerals, const Eigen::ArrayXd &molar_amounts,
    const Eigen::Vector3d &energy_adjustment = Eigen::Vector3d::Zero(),
    const std::string &name = "Combined mineral");
} // namespace burnman
