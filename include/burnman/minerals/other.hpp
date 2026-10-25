// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/mineral.hpp"

namespace burnman::minerals::other {
/// AA1994 liquid Fe, matching Python BurnMan's Cv-based Helmholtz model.
/// S_0=100 J/(mol K) is the legacy unfitted entropy reference, not a phase fit.
std::shared_ptr<Mineral> liquid_iron();
} // namespace burnman::minerals::other
