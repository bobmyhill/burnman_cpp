// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "burnman/core/calibrant.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace burnman {
namespace {
void state(double value, double temperature, bool is_volume) {
  if (!std::isfinite(value) || !std::isfinite(temperature) ||
      temperature < 0. || (is_volume && value <= 0.))
    throw std::invalid_argument("Calibrant requires finite pressure, positive "
                                "volume and nonnegative temperature.");
}

double checked(double result) {
  if (!std::isfinite(result))
    throw std::invalid_argument(
        "Calibrant function returned a nonfinite value; "
        "the state may be outside its valid domain.");
  return result;
}

void covariance(const Eigen::Matrix2d &C) {
  if (!C.allFinite() || C(0, 0) < 0. || C(1, 1) < 0.)
    throw std::invalid_argument("Calibrant covariance must be finite and "
                                "positive semidefinite.");
  const double scale = std::sqrt(C(0, 0)) * std::sqrt(C(1, 1));
  const double tolerance =
      1.e-12 * std::max({scale, std::abs(C(0, 1)), std::abs(C(1, 0))});
  if (std::abs(C(0, 1) - C(1, 0)) > tolerance ||
      std::abs(C(0, 1)) > scale * (1. + 1.e-12))
    throw std::invalid_argument("Calibrant covariance must be symmetric and "
                                "positive semidefinite.");
}

// Python BurnMan's downhill bracket search, followed by Brent's method.
// The solver uses dimensionless coordinates and never invokes Python through
// a C callback or changes a process-global GSL error handler.
double invert(const std::function<double(double)> &function, double x0,
              double dx, bool positive = false) {
  constexpr double ratio = 1.618;
  double f0 = checked(function(x0));
  if (f0 == 0.)
    return x0;
  double left = x0 - dx, right = x0 + dx;
  double fl = checked(function(left)), fr = checked(function(right));
  int iterations = 0;
  while ((f0 - fl) * (fr - f0) < 0. && iterations++ < 100) {
    dx /= ratio;
    left = x0 - dx;
    right = x0 + dx;
    fl = checked(function(left));
    fr = checked(function(right));
  }
  if (iterations >= 100)
    throw std::invalid_argument("Cannot bracket the calibrant inversion.");
  const double slope = fr - f0;
  double x1, f1;
  if ((slope > 0. && f0 > 0.) || (slope <= 0. && f0 <= 0.)) {
    dx = -dx;
    x1 = left;
    f1 = fl;
  } else {
    x1 = right;
    f1 = fr;
  }
  iterations = 0;
  while (std::signbit(f0) == std::signbit(f1) && f1 != 0. &&
         iterations++ < 100) {
    dx *= ratio;
    x0 = x1;
    f0 = f1;
    const double next = x1 + dx;
    x1 = positive && next <= 0. ? .5 * x1 : next;
    f1 = checked(function(x1));
  }
  if (f1 == 0.)
    return x1;
  if (iterations >= 100 || std::signbit(f0) == std::signbit(f1))
    throw std::invalid_argument("Cannot bracket the calibrant inversion; "
                                "the state may be outside its valid domain.");
  double a = x0, b = x1, c = x1;
  double fa = f0, fb = f1, fc = f1;
  double d = b - a, e = d;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (int iter = 0; iter < 200; ++iter) {
    if (std::signbit(fb) == std::signbit(fc)) {
      c = a;
      fc = fa;
      d = e = b - a;
    }
    if (std::abs(fc) < std::abs(fb)) {
      a = b;
      fa = fb;
      b = c;
      fb = fc;
      c = a;
      fc = fa;
    }
    const double tolerance = 2. * eps * std::abs(b) + 2.e-15;
    const double midpoint = .5 * (c - b);
    if (std::abs(midpoint) <= tolerance || fb == 0.)
      return b;
    if (std::abs(e) >= tolerance && std::abs(fa) > std::abs(fb)) {
      const double s = fb / fa;
      double p, q;
      if (a == c) {
        p = 2. * midpoint * s;
        q = 1. - s;
      } else {
        q = fa / fc;
        const double r = fb / fc;
        p = s * (2. * midpoint * q * (q - r) - (b - a) * (r - 1.));
        q = (q - 1.) * (r - 1.) * (s - 1.);
      }
      if (p > 0.)
        q = -q;
      p = std::abs(p);
      const double old_e = e;
      e = d;
      if (2. * p < std::min(3. * midpoint * q - std::abs(tolerance * q),
                            std::abs(old_e * q)))
        d = p / q;
      else
        d = e = midpoint;
    } else {
      d = e = midpoint;
    }
    a = b;
    fa = fb;
    b += std::abs(d) > tolerance ? d : std::copysign(tolerance, midpoint);
    fb = checked(function(b));
  }
  throw std::runtime_error("Calibrant inversion did not converge.");
}
} // namespace

Calibrant::Calibrant(Function function, const std::string &return_type,
                     Params parameters)
    : params(std::move(parameters)), function_(std::move(function)),
      return_type_(return_type) {
  if (!function_ || (return_type != "pressure" && return_type != "volume"))
    throw std::invalid_argument(
        "calibrant function return type must either be pressure or volume");
}

double Calibrant::calibrant_function(double value, double temperature,
                                     const Params &parameters) const {
  return checked(function_(value, temperature, parameters));
}

double Calibrant::pressure_function(double v, double t, const Params &p) const {
  state(v, t, true);
  if (return_type_ == "pressure")
    return calibrant_function(v, t, p);
  return 1.e9 *
         invert(
             [&](double x) { return calibrant_function(x * 1.e9, t, p) - v; },
             0., 300.);
}

double Calibrant::volume_function(double pressure_value, double t,
                                  const Params &p) const {
  state(pressure_value, t, false);
  if (return_type_ == "volume")
    return calibrant_function(pressure_value, t, p);
  const auto found = p.find("V_0");
  if (found == p.end() || !std::isfinite(found->second) || found->second <= 0.)
    throw std::invalid_argument("A positive V_0 is required for calibrant "
                                "volume inversion.");
  const double v0 = found->second;
  return v0 * invert(
                  [&](double x) {
                    return pressure_function(x * v0, t, p) - pressure_value;
                  },
                  1., .01, true);
}

double Calibrant::pressure(double v, double t) const {
  return pressure_function(v, t, params);
}
double Calibrant::volume(double p, double t) const {
  return volume_function(p, t, params);
}

Calibrant::CovarianceResult
Calibrant::pressure(double v, double t, const Eigen::Matrix2d &C) const {
  covariance(C);
  state(v, t, true);
  const double dv = v / 1.e7, dt = .01;
  const double vm = pressure(v - dv / 2., t), vp = pressure(v + dv / 2., t);
  const double tm = pressure(v, t - dt / 2.), tp = pressure(v, t + dt / 2.);
  Eigen::Matrix<double, 3, 2> J;
  J << (vp - vm) / dv, (tp - tm) / dt, 1., 0., 0., 1.;
  return {(vm + vp + tm + tp) / 4., J * C * J.transpose()};
}

Calibrant::CovarianceResult Calibrant::volume(double p, double t,
                                              const Eigen::Matrix2d &C) const {
  covariance(C);
  state(p, t, false);
  const double dp = 100., dt = .01;
  const double pm = volume(p - dp / 2., t), pp = volume(p + dp / 2., t);
  const double tm = volume(p, t - dt / 2.), tp = volume(p, t + dt / 2.);
  Eigen::Matrix<double, 3, 2> J;
  J << (pp - pm) / dp, (tp - tm) / dt, 1., 0., 0., 1.;
  return {(pm + pp + tm + tp) / 4., J * C * J.transpose()};
}
} // namespace burnman
