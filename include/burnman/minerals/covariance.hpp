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
#include <string>
#include <vector>

namespace burnman::minerals {
/** Covariances of endmember zero-point energies, in (J/mol)².
 * endmember_names gives the ordering of both matrix axes. Published values
 * retain their original precision and include cross-covariances.
 */
struct CovarianceData {
  std::vector<std::string> endmember_names;
  Eigen::MatrixXd covariance_matrix;
};
} // namespace burnman::minerals
