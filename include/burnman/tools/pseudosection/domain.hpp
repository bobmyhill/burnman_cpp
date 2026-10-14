// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#pragma once
#include "burnman/tools/pseudosection/common.hpp"
#include <utility>

namespace burnman::pseudosections::detail {
struct Engine;
// Boundary termination and EOS envelopes are independent of field labelling.
bool unfinished_boundary(const std::string &status);

std::pair<std::string, std::string> boundary_statuses(const Boundary &);

std::vector<Eigen::MatrixXd>
excluded_model_regions(Engine &, const Eigen::Vector2d &origin,
                       const Eigen::Vector2d &range);

void join_model_limits(Result &);

void join_frame_limits(Result &, Engine &);

int field_variance(const Assemblage &, std::size_t n_phases);
} // namespace burnman::pseudosections::detail
