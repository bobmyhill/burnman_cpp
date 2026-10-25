// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/mineral.hpp"

namespace burnman::minerals::SE_2015 {
// Iron phases with the full Brosh cold curve. HCP b_1=2.49614 is from
// Perple_X SE15ver.dat; FCC magnetic parameters use the AFM factor -3.
std::shared_ptr<Mineral> bcc_iron();
std::shared_ptr<Mineral> fcc_iron();
std::shared_ptr<Mineral> hcp_iron();
std::shared_ptr<Mineral> liquid_iron();
} // namespace burnman::minerals::SE_2015
