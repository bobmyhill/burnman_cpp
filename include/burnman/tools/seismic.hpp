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

#include <array>

namespace burnman::tools::seismic {
/// Matas et al. (2007) attenuation correction (beta = 0.3).
/// Returns corrected P, S and bulk sound velocities, in m/s.
std::array<double, 3> attenuation_correction(double v_p, double v_s,
                                             double v_phi, double Qs,
                                             double Qphi);
} // namespace burnman::tools::seismic
