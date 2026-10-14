// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#pragma once
#include "burnman/tools/pseudosection/types.hpp"

namespace burnman::pseudosections {
/// Find closed loops through saved stable vertices and traced lines, retaining
/// holes and disconnected regions. Optionally use the calculation frame to
/// close edge fields. No equilibrium solves or raster interpolation are
/// performed. tolerance measures distance after each diagram axis is rescaled
/// to 0..1. Agreed exterior-loop labels take priority; otherwise intersect
/// admissible boundary assemblages around the loop, then use a saved interior
/// sample. With no perimeter labels, the nearest saved sample is provisional
/// evidence. Unresolved conflicts remain unidentified; boundary lengths never
/// select an assemblage. Missing or conflicting evidence adds a diagnostic.
/// Model-domain exclusions remain unidentified.
/// Interior chord crossings do not connect traces. Each closed loop is filled
/// independently; no polygon union or overlap repair is performed.
/// merge_fields is retained for compatibility and has no effect.
FieldPolygons field_polygons(const Result &result, double tolerance = 1.e-8,
                             bool close_domain = true,
                             bool merge_fields = true);

} // namespace burnman::pseudosections
