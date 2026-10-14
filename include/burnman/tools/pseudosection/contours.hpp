// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#pragma once
#include "burnman/tools/pseudosection/types.hpp"
#include <functional>

namespace burnman::pseudosections {
struct ContourSettings {
  // Cross-field sweeps also connect their interior samples along the polygon;
  // geometry subdivisions follow bends and avoid crossing excluded holes.
  int seed_grid = 5, max_trace_steps = 1000;
  // Distances are fractions of the complete diagram ranges, as in Settings.
  double step = .02, min_step = 1.e-6;
};

struct ContourLine {
  // Same one-based number as the source FieldPolygon.
  int field_number = 0;
  // Internal sampled-assemblage ID, retained for compatibility.
  int field_id = -1;
  std::vector<int> phases;
  std::vector<BoundaryPoint> points;
  bool closed = false;
  std::string termination;
};

struct ContourResult {
  DiagramType diagram_type = DiagramType::PT;
  std::vector<ContourLine> lines;
  std::vector<std::string> diagnostics;
  int equilibrium_solves = 0;
  bool resolved = true;
};

/// Construct an equality constraint for the field's actual assemblage and
/// parameter layout. Return nullptr where it is undefined
/// (e.g. a garnet composition in a field without garnet). Called once per
/// field; the constraint and its native derivatives drive continuation.
using ContourConstraintFactory =
    std::function<std::unique_ptr<equilibration::EqualityConstraint>(
        const Assemblage &, const equilibration::EquilibrationParameters &,
        const std::vector<int> &)>;

/// Trace a constraint through saved, identified closed fields, using their
/// accepted phase amounts/compositions as warm starts. No phase boundaries
/// are recalculated and the source is not modified. Candidates and phase-ID
/// spacing must match the original calculation. Seeds sweep across each field
/// and along paths inside its polygon. Finite sampling can miss disconnected
/// contours; increase seed_grid to assess their discovery.
/// Use the saved field's phases, check the equilibrium constraints and reject
/// negative Gibbs curvature along mass-conserving reactions;
/// discovering phases and checking alternative assemblages is not repeated.
ContourResult
pseudosection_contours(const Result &previous,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const ContourConstraintFactory &constraint,
                       const ContourSettings &settings = ContourSettings{});

} // namespace burnman::pseudosections
