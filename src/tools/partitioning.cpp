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

#include "burnman/tools/partitioning.hpp"
#include "burnman/utils/constants.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burnman::partitioning {
std::pair<double, double> calculate_nakajima_fp_pv_partition_coefficient(
    double pressure, double temperature, const types::FormulaMap &bulk,
    double Kd_0) {
  if (!std::isfinite(pressure) || !std::isfinite(temperature) ||
      temperature <= 0. || !std::isfinite(Kd_0) || Kd_0 <= 0.)
    throw std::invalid_argument(
        "Supply finite pressure and positive temperature and Kd_0.");
  auto amount = [&bulk](const char *element) {
    const auto it = bulk.find(element);
    const double value = it == bulk.end() ? 0. : it->second;
    if (!std::isfinite(value) || value < 0.)
      throw std::invalid_argument(
          "Element amounts must be finite and nonnegative.");
    return value;
  };
  const double mg = amount("Mg"), fe = amount("Fe"), si = amount("Si");
  const double total = mg + fe;
  if (!std::isfinite(total) || total <= 0. || si <= 0. || si >= total)
    throw std::invalid_argument("Both phases require 0 < Si < Mg+Fe.");
  const double f = fe / total, s = si / total;
  const double log_K = (25.e9 - pressure) * 2.e-7 /
                           (constants::physics::gas_constant * temperature) +
                       std::log(Kd_0);
  if (f == 0. || f == 1. || log_K == 0.)
    return {f, f};
  // The limiting solutions also avoid exponentially small root intervals.
  if (log_K > 700.)
    return f <= s ? std::make_pair(0., f / s)
                  : std::make_pair((f - s) / (1. - s), 1.);
  if (log_K < -700.)
    return f <= 1. - s ? std::make_pair(f / (1. - s), 0.)
                       : std::make_pair(1., (f - 1. + s) / s);
  // Solve bulk balance in log odds, avoiding cancellation near K=1.
  auto logistic = [](double logit) {
    return logit >= 0. ? 1. / (1. + std::exp(-logit))
                       : std::exp(logit) / (1. + std::exp(logit));
  };
  const double logit_f = std::log(f) - std::log1p(-f);
  double lo = logit_f - std::max(0., log_K);
  double hi = logit_f - std::min(0., log_K);
  for (int i = 0; i < 100; ++i) {
    const double logit = .5 * (lo + hi);
    if ((1. - s) * logistic(logit) + s * logistic(logit + log_K) < f)
      lo = logit;
    else
      hi = logit;
  }
  const double logit = .5 * (lo + hi);
  return {logistic(logit), logistic(logit + log_K)};
}
} // namespace burnman::partitioning
