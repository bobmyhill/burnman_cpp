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
/// Recheck terminal critical reactions without changing traces or samples.
/// Supply the original candidate models in their original order. Old results
/// have no certificates until checked; unrelated diagnostics are preserved.
Result verify_critical_endpoints(
    const Result &previous,
    const std::vector<std::shared_ptr<Material>> &candidates);

} // namespace burnman::pseudosections

namespace burnman::pseudosections::detail {
struct Engine;
void verify_critical_endpoints(Result &, Engine &);

void recover_boundary_labels(Result &);

std::vector<std::string> field_diagnostics(const Result &, Engine &);
} // namespace burnman::pseudosections::detail
