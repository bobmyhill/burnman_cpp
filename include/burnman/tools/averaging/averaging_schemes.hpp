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

#ifndef BURNMAN_TOOLS_AVERAGING_AVERAGING_SCHEMES_HPP_INCLUDED
#define BURNMAN_TOOLS_AVERAGING_AVERAGING_SCHEMES_HPP_INCLUDED

// Convenience header to get all averaging schemes
#include "burnman/tools/averaging/averaging_base.hpp"
#include "burnman/tools/averaging/averaging_utils.hpp"
#include "burnman/tools/averaging/hs.hpp"
#include "burnman/tools/averaging/hs_lower.hpp"
#include "burnman/tools/averaging/hs_upper.hpp"
#include "burnman/tools/averaging/reuss.hpp"
#include "burnman/tools/averaging/voigt.hpp"
#include "burnman/tools/averaging/vrh.hpp"

// TODO: add to tools group / module in docs
namespace burnman {
/**
 * @namespace burnman::averaging
 * @brief Tools and classes for averaging composite properties.
 *
 * This namespace collects classes that provide functions for
 * averaging elastic moduli and thermodynamic properties.
 * Implemented averaging schemes are:
 * - Voigt
 * - Reuss
 * - Voigt-Reuss-Hill
 * - Hashin-Shtrikman
 */
namespace averaging {
// blank for documentation only
} // namespace averaging
} // namespace burnman

#endif // BURNMAN_TOOLS_AVERAGING_SCHEMES_HPP_INCLUDED
