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
#include <algorithm>
#include <cmath>
#include <functional>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_poly.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace burnman::detail {
using RadialArray = Eigen::ArrayXd;
inline void check_profile(const RadialArray &values, Eigen::Index size,
                          const char *name) {
  if (values.size() != size || !values.isFinite().all())
    throw std::invalid_argument(std::string(name) +
                                " must be finite and match the radius grid.");
}
inline RadialArray checked_radii(const RadialArray &radii) {
  if (radii.size() < 2 || !radii.isFinite().all() || radii[0] < 0. ||
      (radii.tail(radii.size() - 1) <= radii.head(radii.size() - 1)).any())
    throw std::invalid_argument("Supply at least two finite, nonnegative, "
                                "strictly increasing radii.");
  return radii;
}
inline void check_iterations(int iterations, double delta) {
  if (iterations < 1 || !std::isfinite(delta) || delta <= 0.)
    throw std::invalid_argument(
        "Iteration count and relative tolerance must be positive.");
}
inline double interpolate(double radius, const RadialArray &radii,
                          const RadialArray &values) {
  if (!std::isfinite(radius) || radius < radii[0] ||
      radius > radii[radii.size() - 1])
    throw std::out_of_range("Radius is outside the layer.");
  const auto upper =
      std::lower_bound(radii.data(), radii.data() + radii.size(), radius);
  const Eigen::Index i = upper - radii.data();
  if (!i)
    return values[0];
  const double fraction = (radius - radii[i - 1]) / (radii[i] - radii[i - 1]);
  return values[i - 1] + fraction * (values[i] - values[i - 1]);
}
// A not-a-knot CubicSpline, including two/three-point grids and
// local linear fallback for overshoot on monotone neighbouring samples
// (which can occur at density jumps at layer boundaries).
class RadialSpline {
public:
  RadialSpline(const RadialArray &radii, const RadialArray &values)
      : radii_(checked_radii(radii)), coefficients_(4, radii.size() - 1) {
    check_profile(values, radii.size(), "Density or gravity");
    const auto n = radii.size();
    const RadialArray h = radii.tail(n - 1) - radii.head(n - 1);
    const RadialArray delta = values.tail(n - 1) - values.head(n - 1);
    const RadialArray slope = delta / h;
    RadialArray derivatives(n);
    if (n == 2)
      derivatives.setConstant(slope[0]);
    else if (n == 3) {
      derivatives[1] = (h[0] * slope[1] + h[1] * slope[0]) / h.sum();
      derivatives[0] = 2. * slope[0] - derivatives[1];
      derivatives[2] = 2. * slope[1] - derivatives[1];
    } else {
      RadialArray lower = RadialArray::Zero(n), diagonal(n),
                  upper = RadialArray::Zero(n), rhs(n);
      for (Eigen::Index i = 1; i + 1 < n; ++i) {
        lower[i] = h[i];
        diagonal[i] = 2. * (h[i - 1] + h[i]);
        upper[i] = h[i - 1];
        rhs[i] = 3. * (h[i] * slope[i - 1] + h[i - 1] * slope[i]);
      }
      const double first = h[0] + h[1], last = h[n - 3] + h[n - 2];
      diagonal[0] = h[1];
      upper[0] = first;
      rhs[0] =
          ((h[0] + 2. * first) * h[1] * slope[0] + h[0] * h[0] * slope[1]) /
          first;
      lower[n - 1] = last;
      diagonal[n - 1] = h[n - 3];
      rhs[n - 1] = (h[n - 2] * h[n - 2] * slope[n - 3] +
                    (2. * last + h[n - 2]) * h[n - 3] * slope[n - 2]) /
                   last;
      // Solve the tridiagonal derivative system in linear time.
      for (Eigen::Index i = 1; i < n; ++i) {
        const double factor = lower[i] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
      }
      derivatives[n - 1] = rhs[n - 1] / diagonal[n - 1];
      for (Eigen::Index i = n - 2; i >= 0; --i)
        derivatives[i] = (rhs[i] - upper[i] * derivatives[i + 1]) / diagonal[i];
    }
    const RadialArray tolerance =
        64. * std::numeric_limits<double>::epsilon() *
        values.head(n - 1).abs().max(values.tail(n - 1).abs());
    const RadialArray trend = (delta.abs() <= tolerance).select(0., delta);
    for (Eigen::Index i = 0; i + 1 < n; ++i) {
      auto c = coefficients_.col(i);
      const double left = h[i] * derivatives[i],
                   right = h[i] * derivatives[i + 1];
      c << left + right - 2. * delta[i], 3. * delta[i] - 2. * left - right,
          left, values[i];
      const double before = trend[std::max<Eigen::Index>(0, i - 1)],
                   after = trend[std::min(n - 2, i + 1)];
      if (std::min({before, trend[i], after}) < 0. &&
          std::max({before, trend[i], after}) > 0.)
        continue;
      double roots[2] = {};
      const int count = gsl_poly_solve_quadratic(3. * c[0], 2. * c[1], c[2],
                                                 &roots[0], &roots[1]);
      for (int j = 0; j < count; ++j) {
        const double t = roots[j];
        if (t <= 0. || t >= 1.)
          continue;
        const double value = ((c[0] * t + c[1]) * t + c[2]) * t + c[3];
        if (value < std::min(values[i], values[i + 1]) - tolerance[i] ||
            value > std::max(values[i], values[i + 1]) + tolerance[i]) {
          c << 0., 0., delta[i], values[i];
          break;
        }
      }
    }
  }
  double operator()(double radius) const {
    if (!std::isfinite(radius) || radius < radii_[0] ||
        radius > radii_[radii_.size() - 1])
      throw std::out_of_range("Radius is outside the layer.");
    const auto upper =
        std::upper_bound(radii_.data(), radii_.data() + radii_.size(), radius);
    const auto i =
        std::min<Eigen::Index>(upper - radii_.data() - 1, radii_.size() - 2);
    const double t = (radius - radii_[i]) / (radii_[i + 1] - radii_[i]);
    const auto c = coefficients_.col(i);
    return ((c[0] * t + c[1]) * t + c[2]) * t + c[3];
  }

private:
  RadialArray radii_;
  Eigen::Matrix<double, 4, Eigen::Dynamic> coefficients_;
};
// Four-point Gauss-Legendre quadrature is exact for products of cubic splines
// and for the r^4 density moment on each knot interval.
inline double integrate(const std::function<double(double)> &function,
                        double lower, double upper) {
  static const std::unique_ptr<gsl_integration_glfixed_table,
                               decltype(&gsl_integration_glfixed_table_free)>
      table(gsl_integration_glfixed_table_alloc(4),
            gsl_integration_glfixed_table_free);
  if (!table)
    throw std::bad_alloc();
  // Evaluate C++ functions outside a C callback so exceptions unwind normally.
  double result = 0.;
  for (std::size_t i = 0; i < 4; ++i) {
    double radius, weight;
    gsl_integration_glfixed_point(lower, upper, i, &radius, &weight,
                                  table.get());
    result += weight * function(radius);
  }
  return result;
}
// numpy.gradient(y, x, edge_order=2), with first-order ends for two samples.
inline RadialArray gradient(const RadialArray &values,
                            const RadialArray &coordinates) {
  const auto n = values.size();
  RadialArray result(n);
  if (n == 2) {
    result.setConstant((values[1] - values[0]) /
                       (coordinates[1] - coordinates[0]));
    return result;
  }
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto middle = std::clamp<Eigen::Index>(i, 1, n - 2);
    const double h0 = coordinates[middle] - coordinates[middle - 1],
                 h1 = coordinates[middle + 1] - coordinates[middle];
    double a, b, c;
    if (i == 0) {
      a = -(2. * h0 + h1) / (h0 * (h0 + h1));
      b = (h0 + h1) / (h0 * h1);
      c = -h0 / (h1 * (h0 + h1));
    } else if (i == n - 1) {
      a = h1 / (h0 * (h0 + h1));
      b = -(h0 + h1) / (h0 * h1);
      c = (2. * h1 + h0) / (h1 * (h0 + h1));
    } else {
      a = -h1 / (h0 * (h0 + h1));
      b = (h1 - h0) / (h0 * h1);
      c = h0 / (h1 * (h0 + h1));
    }
    result[i] =
        a * values[middle - 1] + b * values[middle] + c * values[middle + 1];
  }
  return result;
}
} // namespace burnman::detail
