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
#include "burnman/tools/pseudosection/types.hpp"
#include <functional>

namespace burnman::pseudosections {
/// Sampling and continuation controls for property contours within saved
/// fields.
struct ContourSettings {
  // Cross-field sweeps also connect their interior samples along the polygon;
  // geometry subdivisions follow bends and avoid crossing excluded holes.
  int seed_grid = 5, max_trace_steps = 1000;
  // Distances are fractions of the complete diagram ranges, as in Settings.
  double step = .02, min_step = 1.e-6;
};

/// One connected contour segment in a particular closed assemblage field.
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

/// Solved contour segments and any failed-tracing diagnostics, separate from
/// the resolution status of the source pseudosection.
struct ContourResult {
  DiagramType diagram_type = DiagramType::PT;
  std::vector<ContourLine> lines;
  std::vector<std::string> diagnostics;
  int equilibrium_solves = 0;
  bool resolved = true;
};

/// Build the equation defining a contour, e.g. a fixed garnet X_Mg or density.
/// Called once per field with its actual assemblage and parameter layout.
/// Return nullptr where the quantity is undefined, such as garnet X_Mg in a
/// garnet-free field. Its derivatives guide subsequent equilibrium
/// continuation.
using ContourConstraintFactory =
    std::function<std::unique_ptr<equilibration::EqualityConstraint>(
        const Assemblage &, const equilibration::EquilibrationParameters &,
        const std::vector<int> &)>;

/// Calculate property or mineral-composition contours within saved phase
/// fields. The supplied constraint fixes the contour value; equilibrium
/// continuation solves phase amounts/compositions along it. Saved states
/// initialise the solves, and the field's outer ring and holes limit the traced
/// segments. Use the original candidate order and phase-ID spacing. previous is
/// unchanged.
///
/// The assemblage is held to the identified field's phases. Check equilibrium
/// and reject negative Gibbs curvature along mass-conserving reactions, but do
/// not repeat phase discovery or alternative-candidate searches. Consequently,
/// contour success is not an independent validation of the source field labels.
/// Finite seeding can miss disconnected contours; vary seed_grid to assess
/// this.
ContourResult
pseudosection_contours(const Result &previous,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const ContourConstraintFactory &constraint,
                       const ContourSettings &settings = ContourSettings{});

} // namespace burnman::pseudosections
