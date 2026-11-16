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

/// A P-T reaction exchanging two limiting assemblages has no finite full field,
/// even when frozen composition faces overestimate nominal Gibbs variance.
bool univariant_reaction(const Boundary &, const Result &);

/// Junction/endpoint IDs used by retained traces; also supports geometry-only
/// input files whose nodes have no explicit endpoint references.
std::set<int> active_boundary_nodes(const Result &);

/// Re-solve interior points of alternative traces joining the same two nodes.
/// Remove a trace only when corrected equilibria verify its metastability and
/// a competing trace verifies stable. Failed correctors leave stability
/// unresolved in every section. Shared endpoints alone do not prove duplicates.
/// Return the number of removed records.
std::size_t remove_metastable_parallel_lines(Result &, Engine &);

/// Find a replacement trace that retraces two independently labelled reactions
/// meeting at a saved junction. Its single side pair cannot describe both
/// branches. Only the fully covered trace is redundant; the two phase-checked
/// branches and their junction remain in the diagram.
std::set<int> reaction_overtraces(const Result &);

/// Continue a phase boundary between two local coalescence records only when
/// equilibria along its path confirm the same neighbouring assemblages. The
/// coalescences are numerical trace stops, not necessarily reaction endpoints.
/// Return the number of accepted connections.
std::size_t recover_solvus_branch_gaps(Result &, Engine &);

/// Continue a boundary past a locally coalesced solution state to the
/// temperature edge when stable equilibria confirm both sides throughout.
std::size_t continue_boundary_past_coalescence(Result &, Engine &);

/// Repair reaction-line connectivity using verified equilibrium junctions.
/// Split curves at accepted nodes and use the supplied label/refinement
/// callbacks to recover neighbouring assemblages and test suspect segments.
void repair_field_boundaries(
    Result &, Engine &, const std::function<void(Boundary &)> &,
    const std::function<bool(Boundary &, std::size_t)> &);
} // namespace burnman::phase_diagrams::detail
