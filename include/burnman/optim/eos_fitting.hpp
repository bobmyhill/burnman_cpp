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

namespace burnman {
class Material;

namespace optim {
/** Default MLE tolerances, following Python BurnMan's property flags.
 *
 * Sets and leaves material at 1e5 Pa and 300 K. Energy flags use 1 J/mol,
 * pressure flags use 100 Pa, and other flags use 1e-5 times the property
 * value at that state. Throws if any resulting tolerance is not positive.
 */
Eigen::VectorXd default_mle_tolerances(Material &material,
                                       const std::vector<std::string> &flags);
} // namespace optim
} // namespace burnman
