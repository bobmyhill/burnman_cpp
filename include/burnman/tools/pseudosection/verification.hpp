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
#include "burnman/tools/pseudosection/common.hpp"

namespace burnman::pseudosections {
/// Verify whether a dangling solvus reaction ends at a genuine critical point.
/// Re-solve coalescing compositions and limiting univariant states using the
/// original candidate models in their original order. Return a copy with
/// updated certificates/diagnostics, preserving traces, samples and unrelated
/// messages. A geometric endpoint alone does not certify critical behaviour.
Result verify_critical_endpoints(
    const Result &previous,
    const std::vector<std::shared_ptr<Material>> &candidates);

} // namespace burnman::pseudosections

namespace burnman::pseudosections::detail {
struct Engine;
/// Update critical-endpoint certificates in place using thermodynamic tests.
void verify_critical_endpoints(Result &, Engine &);

/// Recover right/left assemblages from trusted neighbouring closed loops.
/// Only unambiguous, reaction-compatible perimeter evidence is transferred.
void recover_boundary_labels(Result &);

/// Diagnose unresolved field labels, unpaired junctions and sampled assemblages
/// with no corresponding closed field. Constructing the messages does not trace
/// new boundaries or modify the result.
std::vector<std::string> field_diagnostics(const Result &, Engine &);
} // namespace burnman::pseudosections::detail
