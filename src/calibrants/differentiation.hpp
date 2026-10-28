// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include <cmath>

namespace burnman::calibrants::detail::differentiation {
// One-variable forward differentiation of the published Helmholtz expression.
// This avoids subtracting nearly equal energies to compute thermal pressure,
// which otherwise contaminates pressure inversion and measurement covariance.
struct Derivative {
  double value, derivative;
  Derivative(double v, double d = 0.) : value(v), derivative(d) {}
};
inline Derivative operator+(Derivative a, Derivative b) {
  return {a.value + b.value, a.derivative + b.derivative};
}
inline Derivative operator-(Derivative a, Derivative b) {
  return {a.value - b.value, a.derivative - b.derivative};
}
inline Derivative operator-(Derivative a) { return {-a.value, -a.derivative}; }
inline Derivative operator*(Derivative a, Derivative b) {
  return {a.value * b.value, a.derivative * b.value + a.value * b.derivative};
}
inline Derivative operator/(Derivative a, Derivative b) {
  return {a.value / b.value,
          (a.derivative - a.value / b.value * b.derivative) / b.value};
}
inline Derivative exp(Derivative x) {
  const double value = std::exp(x.value);
  return {value, value * x.derivative};
}
inline Derivative log(Derivative x) {
  return {std::log(x.value), x.derivative / x.value};
}
inline Derivative pow(Derivative x, Derivative y) {
  const double value = std::pow(x.value, y.value);
  return {value, value * (y.derivative * std::log(x.value) +
                          y.value * x.derivative / x.value)};
}
} // namespace burnman::calibrants::detail::differentiation
