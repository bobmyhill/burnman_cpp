// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#pragma once

#include <map>
#include <string>

namespace burnman::minerals::SLB_2024 {
/// Preferred plot labels keyed by full model name. These labels do not rename
/// minerals or alter the identities retained in equilibrium and diagram files.
const std::map<std::string, std::string> &phase_aliases();
} // namespace burnman::minerals::SLB_2024
