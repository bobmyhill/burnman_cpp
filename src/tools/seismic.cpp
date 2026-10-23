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

#include "burnman/tools/seismic.hpp"
#include <cmath>

namespace burnman::tools::seismic {
std::array<double, 3> attenuation_correction(double v_p, double v_s,
                                             double v_phi, double Qs,
                                             double Qphi) {
  const double cot = 1.0 / std::tan(0.3 * std::acos(-1.0) / 2.0);
  // Algebraically equivalent to Qp = 3/4 (vp/vs)^2 Qs, including vs = 0.
  const double inv_Qp = (4.0 / 3.0) * (v_s / v_p) * (v_s / v_p) / Qs;
  return {v_p * (1.0 - 0.5 * cot * inv_Qp), v_s * (1.0 - 0.5 * cot / Qs),
          v_phi * (1.0 - 0.5 * cot / Qphi)};
}
} // namespace burnman::tools::seismic
