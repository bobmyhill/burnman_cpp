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

#include <Eigen/Core>

namespace burnman::data {
// Read all columns of a seismic table, retaining their original ordering.
Eigen::ArrayXXd read_table(const char *text, Eigen::Index columns);

// Read geotherm coordinates and temperatures, ignoring uncertainty columns.
Eigen::ArrayXXd read_table(const char *text);
} // namespace burnman::data
