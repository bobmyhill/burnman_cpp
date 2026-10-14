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
#include <functional>
#include <set>

namespace burnman::pseudosections::detail {
struct Engine;
// Infer Gibbs variance from saved endpoint metadata; -1 means unavailable.
int boundary_variance(const Boundary &, const Result &);

std::set<int> active_boundary_nodes(const Result &);

// Drop lines whose interior is metastable when a parallel line joins the same
// nodes. Returns the number removed.
std::size_t remove_metastable_parallel_lines(Result &, Engine &);

// Split verified arcs at stable nodes, then recover side labels.
void repair_field_boundaries(
    Result &, Engine &, const std::function<void(Boundary &)> &,
    const std::function<bool(Boundary &, std::size_t)> &);
} // namespace burnman::pseudosections::detail
