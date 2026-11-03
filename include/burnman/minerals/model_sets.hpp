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
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace burnman::minerals::model_sets {
/// Curated phase models and their calibration provenance. Each factory creates
/// independently owned phases. Pressure guidance is in Pa, or absent when the
/// source gives no collection-wide limit. Water models are recorded explicitly.
struct ModelSet {
  std::string name, version, dataset, source, citation;
  std::optional<double> recommended_max_pressure, cautious_max_pressure;
  std::vector<std::shared_ptr<Material>> phases;
  std::string notes{};
  std::string water{"none"};
  /// Phase abbreviation to display description, when supplied by the set.
  std::map<std::string, std::string> phase_glossary{};
};

/// HPx collections include q, trd, crst, coe and stv from the matching dataset.
ModelSet metapelite();
/// Choose one clinopyroxene calibration: "dio" (ordered omphacitic) or "aug"
/// (augitic, with tschermaks substitution). These must not be used together.
ModelSet metabasite(const std::string &clinopyroxene = "dio");
/// G25 (hydrous/subalkaline) or W24 (dry/alkaline) models with dataset 6.36.
/// G25 includes its silicate-bearing aqueous fluid, with PS94 H2O.
ModelSet igneous(const std::string &calibration = "G25");
/// The 27 active HeFESTo SLB2024 solids: 15 ordinary solution models and
/// 12 pure phases. Includes metallic iron; excludes the unused lppv endmember.
ModelSet slb2024();
} // namespace burnman::minerals::model_sets
