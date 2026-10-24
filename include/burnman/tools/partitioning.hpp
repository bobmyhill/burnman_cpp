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
#include "burnman/utils/types/simple_types.hpp"
#include <utility>

namespace burnman::partitioning {
/** Nakajima et al. (2012) Fe-Mg exchange between ferropericlase and
 * bridgmanite. Return their Fe/(Fe+Mg) ratios, respectively. Pressure uses Pa,
 * temperature K, and bulk composition gives elemental molar amounts.
 */
std::pair<double, double> calculate_nakajima_fp_pv_partition_coefficient(
    double pressure, double temperature,
    const types::FormulaMap &bulk_composition_mol,
    double initial_distribution_coefficient);
} // namespace burnman::partitioning
