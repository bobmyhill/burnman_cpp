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
#include "burnman/tools/phase_diagram/types.hpp"

namespace burnman::phase_diagrams {
/// Reconstruct the closed phase fields bounded by saved equilibrium lines.
/// A loop follows connected boundary arcs back to its starting vertex. Its
/// coordinates form a ring: an ordered list whose last point repeats the first.
/// A field has an outer ring and may contain holes, e.g. islands with a
/// different assemblage or areas excluded by the phase models. close_domain
/// allows the calculation frame to complete fields at its edges. No equilibrium
/// calculations are performed and result is unchanged.
///
/// Stable junctions and accepted endpoints determine which arcs connect. A
/// crossing of straight drawing segments (chords) does not establish a new
/// thermodynamic junction. Fields are filled independently, so coarse curved
/// boundaries can overlap visually without changing their connectivity.
/// tolerance is a distance after each diagram axis is rescaled to 0..1.
///
/// Identify each field from the assemblages permitted by its perimeter
/// reactions. Verified reaction conditions constrain the whole loop, including
/// very thin fields whose side probes skipped a neighbour. Agreed outer-ring
/// labels take priority over saved interior samples.
/// Conflicting or incomplete evidence produces a provisional label or leaves
/// the assemblage unknown, with a diagnostic. Boundary length gives no extra
/// authority to a label. EOS-domain exclusions have no assigned assemblage.
FieldPolygons field_polygons(const Result &result, double tolerance = 1.e-8,
                             bool close_domain = true);

} // namespace burnman::phase_diagrams
