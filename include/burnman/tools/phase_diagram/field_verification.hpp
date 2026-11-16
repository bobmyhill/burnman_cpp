// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once
#include "burnman/tools/phase_diagram/common.hpp"

namespace burnman::phase_diagrams {
/// Verify field labels through stable bounding reactions, common junctions
/// and corrected crossing checks. Preserve coordinate precision notes
/// separately from blocking diagnostics; a label certificate is not a width.
Result verify_field_perimeters(
    const Result &, const std::vector<std::shared_ptr<Material>> &candidates);
} // namespace burnman::phase_diagrams

namespace burnman::phase_diagrams::detail {
struct Engine;
/// Invalidation token tying field certificates to their saved reaction graph.
std::string field_verification_signature(const Result &);
/// Verify field perimeters in place within an existing thermodynamic workspace.
void verify_field_perimeters(Result &, Engine &,
                             bool recheck_closed_rings = false);
} // namespace burnman::phase_diagrams::detail
