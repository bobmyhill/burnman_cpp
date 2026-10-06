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
#include "burnman/core/solution.hpp"
#include <string>
#include <vector>

namespace burnman::minerals::model_sets {
/// HPx phase-model collections and their calibration provenance. Each call
/// creates independently owned phases. Pressures are in Pa; water is PS94 with
/// the Holland-Powell (2011) ideal-gas thermal reference.
struct ModelSet {
  std::string name, version, dataset, source, citation;
  double recommended_max_pressure, cautious_max_pressure;
  std::vector<std::shared_ptr<Material>> phases;
  std::string notes{};
};

ModelSet metapelite();
/// Choose one clinopyroxene calibration: "dio" (ordered omphacitic) or "aug"
/// (augitic, with tschermaks substitution). These must not be used together.
ModelSet metabasite(const std::string &clinopyroxene = "dio");
/// G25 (hydrous/subalkaline) or W24 (dry/alkaline) models with dataset 6.36.
/// G25 includes its silicate-bearing aqueous fluid, with PS94 H2O.
ModelSet igneous(const std::string &calibration = "G25");
} // namespace burnman::minerals::model_sets
