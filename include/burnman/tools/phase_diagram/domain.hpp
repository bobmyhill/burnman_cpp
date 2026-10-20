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
#include <utility>

namespace burnman::phase_diagrams::detail {
struct Engine;
/// Whether tracing stopped before a verified junction, critical or domain end.
bool unfinished_boundary(const std::string &status);

/// Read the saved start/end termination messages in boundary point order.
std::pair<std::string, std::string> boundary_statuses(const Boundary &);

/// Every saved point lies on one calculation edge within floating-point
/// roundoff. Such an arc has only one in-domain neighbouring field.
bool lies_on_calculation_frame(const Boundary &, const Result &);

/// Trace the EOS admissibility envelope, returning closed rings in native
/// diagram coordinates. origin/range map the selected axes to the unit square.
/// These exclusions describe model validity, not mineral reaction boundaries.
std::vector<Eigen::MatrixXd>
excluded_model_regions(Engine &, const Eigen::Vector2d &origin,
                       const Eigen::Vector2d &range);

/// Correct terminated traces onto the saved EOS envelope within node tolerance,
/// retaining the verified boundary equilibria at their model limits.
void join_model_limits(Result &, Engine &);

/// Re-solve near-frame endpoints at the exact calculation edge before joining.
void join_frame_limits(Result &, Engine &);

/// Gibbs phase-rule variance C-P+2, using the assemblage's independent elements
/// for C and the supplied phase count for P. Negative values are retained.
int field_variance(const Assemblage &, std::size_t n_phases);
} // namespace burnman::phase_diagrams::detail
