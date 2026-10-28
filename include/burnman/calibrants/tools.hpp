// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/core/calibrant.hpp"

namespace burnman::calibrants {
/// Convert pressure scales by preserving the measured volume and temperature.
double pressure_to_pressure(const Calibrant &old_calibrant,
                            const Calibrant &new_calibrant, double pressure,
                            double temperature);
/// Input and output covariance order [P,T].
std::pair<double, Eigen::Matrix2d>
pressure_to_pressure(const Calibrant &old_calibrant,
                     const Calibrant &new_calibrant, double pressure,
                     double temperature, const Eigen::Matrix2d &PT_covariance);
} // namespace burnman::calibrants
