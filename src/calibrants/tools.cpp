// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "burnman/calibrants/tools.hpp"

namespace burnman::calibrants {
double pressure_to_pressure(const Calibrant &old_calibrant,
                            const Calibrant &new_calibrant, double pressure,
                            double temperature) {
  return new_calibrant.pressure(old_calibrant.volume(pressure, temperature),
                                temperature);
}

std::pair<double, Eigen::Matrix2d>
pressure_to_pressure(const Calibrant &old_calibrant,
                     const Calibrant &new_calibrant, double pressure,
                     double temperature, const Eigen::Matrix2d &C) {
  const auto [v, VPT] = old_calibrant.volume(pressure, temperature, C);
  Eigen::Matrix2d VT;
  VT << VPT(0, 0), VPT(0, 2), VPT(2, 0), VPT(2, 2);
  const auto [p, PVT] = new_calibrant.pressure(v, temperature, VT);
  Eigen::Matrix2d PT;
  PT << PVT(0, 0), PVT(0, 2), PVT(2, 0), PVT(2, 2);
  return {p, PT};
}
} // namespace burnman::calibrants
