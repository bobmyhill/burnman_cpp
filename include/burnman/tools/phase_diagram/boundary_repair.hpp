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
#include "burnman/tools/phase_diagram/common.hpp"
#include <functional>
#include <set>

namespace burnman::phase_diagrams::detail {
struct Engine;
/// Infer a reaction assemblage's Gibbs variance from its saved junctions.
/// Return -1 when suitable endpoint metadata is unavailable.
int boundary_variance(const Boundary &, const Result &);

/// Junction/endpoint IDs used by retained traces; also supports geometry-only
/// input files whose nodes have no explicit endpoint references.
std::set<int> active_boundary_nodes(const Result &);

/// Re-solve interior points of alternative traces joining the same two nodes.
/// Remove a failing trace only when a competing trace passes the checks.
/// Inverse sections require verified metastability; a failed corrector
/// leaves stability unresolved.
/// sharing endpoints does not by itself make two curves duplicates.
/// Return the number of removed records.
std::size_t remove_metastable_parallel_lines(Result &, Engine &);

/// Repair reaction-line connectivity using verified equilibrium junctions.
/// Split curves at accepted nodes and use the supplied label/refinement
/// callbacks to recover neighbouring assemblages and test suspect segments.
void repair_field_boundaries(
    Result &, Engine &, const std::function<void(Boundary &)> &,
    const std::function<bool(Boundary &, std::size_t)> &);
} // namespace burnman::phase_diagrams::detail
