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
#include <gsl/gsl_spline.h>
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
// Native radial integration uses a cubic interpolant (linear for two points).
// Integrating separately over each interval preserves material boundaries.
class RadialSpline {
public:
  RadialSpline(const RadialArray &radii, const RadialArray &values)
      : spline_(gsl_spline_alloc(radii.size() > 2 ? gsl_interp_cspline
                                                  : gsl_interp_linear,
                                 static_cast<std::size_t>(radii.size())),
                gsl_spline_free) {
    check_profile(values, radii.size(), "Density or gravity");
    if (!spline_ || gsl_spline_init(spline_.get(), radii.data(), values.data(),
                                    static_cast<std::size_t>(radii.size())))
      throw std::runtime_error("Unable to construct a radial spline.");
  }
  double operator()(double radius) const {
    return gsl_spline_eval(spline_.get(), radius, nullptr);
  }

private:
  std::unique_ptr<gsl_spline, decltype(&gsl_spline_free)> spline_;
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
// numpy.gradient(y, edge_order=2), with first-order ends for two samples.
inline RadialArray index_gradient(const RadialArray &values) {
  const auto n = values.size();
  RadialArray result(n);
  if (n == 2) {
    result.setConstant(values[1] - values[0]);
    return result;
  }
  result[0] = (-3. * values[0] + 4. * values[1] - values[2]) / 2.;
  result[n - 1] =
      (3. * values[n - 1] - 4. * values[n - 2] + values[n - 3]) / 2.;
  for (Eigen::Index i = 1; i + 1 < n; ++i)
    result[i] = .5 * (values[i + 1] - values[i - 1]);
  return result;
}
} // namespace burnman::detail
