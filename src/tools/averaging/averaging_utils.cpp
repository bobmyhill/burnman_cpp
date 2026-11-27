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

#include "burnman/tools/averaging/averaging_utils.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/constants.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace burnman::averaging {

namespace utils {

// blank namespace for internals
namespace {

struct SelfConsistentData {
  Eigen::ArrayXd fractions;
  Eigen::ArrayXd bulk;
  Eigen::ArrayXd shear;
  double zero_bulk;
  double zero_limit;

  double effective_bulk(double g) const {
    if (g == 0.0 && zero_bulk > 0.0)
      return 0.0;
    const Eigen::ArrayXd denominators = bulk + 4.0 * g / 3.0;
    // The weighted form avoids cancellation in a shifted harmonic mean.
    const Eigen::ArrayXd weights =
        fractions * (denominators.minCoeff() / denominators);
    return (weights * bulk).sum() / weights.sum();
  }
};

double self_consistent_residual(double g, void *parameters) {
  const auto &data = *static_cast<const SelfConsistentData *>(parameters);
  if (g == 0.0)
    return data.zero_limit;
  const double k = data.effective_bulk(g);
  const double zeta = g * (9.0 * k + 8.0 * g) / (6.0 * (k + 2.0 * g));
  return (data.fractions * ((data.shear - g) / (data.shear + zeta))).sum();
}

double hs_bulk_fn(const Eigen::ArrayXd &volumes,
                  const Eigen::ArrayXd &bulk_moduli,
                  const Eigen::ArrayXd &shear_moduli, bool n) {
  Eigen::ArrayXd vol_frac = volumes / volumes.sum();
  double K_n = n ? bulk_moduli.maxCoeff() : bulk_moduli.minCoeff();
  double G_n = n ? shear_moduli.maxCoeff() : shear_moduli.minCoeff();
  double alpha_n = -3.0 / (3.0 * K_n + 4.0 * G_n);
  Eigen::Array<bool, Eigen::Dynamic, 1> mask = (bulk_moduli != K_n);
  Eigen::ArrayXd denominator = (1.0 / (bulk_moduli - K_n)) - alpha_n;
  Eigen::ArrayXd A_terms = vol_frac / denominator;
  A_terms = mask.select(A_terms, 0.0);
  double A_n = A_terms.sum();
  return K_n + A_n / (1.0 + alpha_n * A_n);
}

double hs_shear_fn(const Eigen::ArrayXd &volumes,
                   const Eigen::ArrayXd &bulk_moduli,
                   const Eigen::ArrayXd &shear_moduli, bool n) {
  Eigen::ArrayXd vol_frac = volumes / volumes.sum();
  double K_n = n ? bulk_moduli.maxCoeff() : bulk_moduli.minCoeff();
  double G_n = n ? shear_moduli.maxCoeff() : shear_moduli.minCoeff();
  double beta_n =
      -3.0 * (K_n + 2.0 * G_n) / (5.0 * G_n * (3.0 * K_n + 4.0 * G_n));
  Eigen::Array<bool, Eigen::Dynamic, 1> mask = (shear_moduli != G_n);
  Eigen::ArrayXd denominator = (1.0 / (2.0 * (shear_moduli - G_n)) - beta_n);
  Eigen::ArrayXd B_terms = vol_frac / denominator;
  B_terms = mask.select(B_terms, 0);
  double B_n = B_terms.sum();
  return G_n + 0.5 * B_n / (1.0 + beta_n * B_n);
}

} // namespace

std::pair<double, double>
self_consistent_fn(const Eigen::ArrayXd &phase_volumes,
                   const Eigen::ArrayXd &bulk_moduli,
                   const Eigen::ArrayXd &shear_moduli) {
  if (phase_volumes.size() == 0 || phase_volumes.size() != bulk_moduli.size() ||
      phase_volumes.size() != shear_moduli.size())
    throw std::invalid_argument(
        "Phase volumes and moduli must be nonempty arrays of equal length.");
  if (!phase_volumes.isFinite().all() || (phase_volumes < 0.0).any())
    throw std::invalid_argument(
        "Phase volumes must be finite and nonnegative.");

  const Eigen::Index count = (phase_volumes > 0.0).count();
  if (count == 0)
    throw std::invalid_argument(
        "At least one phase must have positive volume.");
  Eigen::ArrayXd volumes(count), bulk(count), shear(count);
  Eigen::Index occupied = 0;
  for (Eigen::Index i = 0; i < phase_volumes.size(); ++i) {
    if (phase_volumes[i] == 0.0)
      continue;
    if (!std::isfinite(bulk_moduli[i]) || !std::isfinite(shear_moduli[i]) ||
        bulk_moduli[i] < 0.0 || shear_moduli[i] < 0.0)
      throw std::invalid_argument("Occupied phases must have finite, "
                                  "nonnegative bulk and shear moduli.");
    volumes[occupied] = phase_volumes[i];
    bulk[occupied] = bulk_moduli[i];
    shear[occupied] = shear_moduli[i];
    ++occupied;
  }
  if (count == 1)
    return {bulk[0], shear[0]};

  // Normalize both amounts and stiffnesses before solving. This avoids
  // overflowing volume sums and makes the root tolerances scale-independent.
  Eigen::ArrayXd fractions = volumes / volumes.maxCoeff();
  fractions /= fractions.sum();
  const double scale = std::max(bulk.maxCoeff(), shear.maxCoeff());
  if (scale == 0.0)
    return {0.0, 0.0};
  bulk /= scale;
  shear /= scale;
  const double zero_bulk = (bulk == 0.0).select(fractions, 0.0).sum();
  const double zero_shear = (shear == 0.0).select(fractions, 0.0).sum();

  // For G -> 0+, zeta/G -> (3-p)/(2+p), with p the zero-bulk fraction.
  // The shear residual tends to 1-5*q/(3-p), with q the zero-shear fraction.
  // Dry pores lose rigidity at fraction 1/2, fluids with K>0 at fraction 3/5.
  const double zero_limit = 1.0 - 5.0 * zero_shear / (3.0 - zero_bulk);
  const SelfConsistentData data{fractions, bulk, shear, zero_bulk, zero_limit};
  if (zero_shear > 0.0 && zero_limit <= 0.0)
    return {scale * data.effective_bulk(0.0), 0.0};

  const double lower = shear.minCoeff();
  const double upper = shear.maxCoeff();
  const double g =
      lower == upper
          ? lower
          : optim::roots::brent(self_consistent_residual, data, lower, upper,
                                std::numeric_limits<double>::min(), 1.0e-12);
  return {scale * data.effective_bulk(g), scale * g};
}

double voigt_fn(const Eigen::ArrayXd &phase_volumes, const Eigen::ArrayXd &X) {
  Eigen::ArrayXd vol_frac = phase_volumes / phase_volumes.sum();
  return (vol_frac * X).sum();
}

double reuss_fn(const Eigen::ArrayXd &phase_volumes, const Eigen::ArrayXd &X) {
  Eigen::ArrayXd vol_frac = phase_volumes / phase_volumes.sum();
  double eps = constants::precision::abs_tolerance;
  // Warn when X <= 0 and |vol_frac| >= eps
  Eigen::Array<bool, Eigen::Dynamic, 1> problematic =
      (X <= 0.0).select(vol_frac.abs() > eps, false);
  if (problematic.any()) {
    // Warn!
    // Could use cerr?
    // std::cerr << "Oops, called reuss_average with Xi <= 0!" << std::endl;
    return 0.0;
  } else {
    return 1.0 / (vol_frac / X).sum();
  }
}

double voigt_reuss_hill_fn(const Eigen::ArrayXd &phase_volumes,
                           const Eigen::ArrayXd &X) {
  return (voigt_fn(phase_volumes, X) + reuss_fn(phase_volumes, X)) / 2.0;
}

double lower_hs_bulk_fn(const Eigen::ArrayXd &volumes,
                        const Eigen::ArrayXd &bulk_moduli,
                        const Eigen::ArrayXd &shear_moduli) {
  return hs_bulk_fn(volumes, bulk_moduli, shear_moduli, 0);
}

double lower_hs_shear_fn(const Eigen::ArrayXd &volumes,
                         const Eigen::ArrayXd &bulk_moduli,
                         const Eigen::ArrayXd &shear_moduli) {
  return hs_shear_fn(volumes, bulk_moduli, shear_moduli, 0);
}

double upper_hs_bulk_fn(const Eigen::ArrayXd &volumes,
                        const Eigen::ArrayXd &bulk_moduli,
                        const Eigen::ArrayXd &shear_moduli) {
  return hs_bulk_fn(volumes, bulk_moduli, shear_moduli, 1);
}

double upper_hs_shear_fn(const Eigen::ArrayXd &volumes,
                         const Eigen::ArrayXd &bulk_moduli,
                         const Eigen::ArrayXd &shear_moduli) {
  return hs_shear_fn(volumes, bulk_moduli, shear_moduli, 1);
}

} // namespace utils

} // namespace burnman::averaging
