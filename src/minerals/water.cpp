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

/* Pitzer & Sterner, JCP 101 (1994), 3111–3116,
 * doi:10.1063/1.467624, water coefficient table and Helmholtz EOS.
 * Ideal-gas thermal reference: Holland & Powell (2011), Table 2a,
 * doi:10.1111/j.1525-1314.2010.00923.x. NIST Chase (1998) is an alternative.
 */
#include "burnman/minerals/water.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace burnman::minerals {
namespace {
constexpr double R = 8.314510; // EOS paper's gas constant
constexpr double coefficients[10][6] = {
    {0, 0, .24657688e6, .51359951e2, 0, 0},
    {0, 0, .58638965, -.28646939e-2, .31375577e-4, 0},
    {0, 0, -.62783840e1, .14791599e-1, .35779579e-3, .15432925e-7},
    {0, 0, 0, -.42719875, -.16325155e-4, 0},
    {0, 0, .56654978e4, -.16580167e2, .76560762e-1, 0},
    {0, 0, 0, .10917883, 0, 0},
    {.38878656e13, -.13494878e9, .30916564e6, .75591105e1, 0, 0},
    {0, 0, -.65537898e5, .18810675e3, 0, 0},
    {-.14182435e14, .18165390e9, -.19769068e6, -.23530318e2, 0, 0},
    {0, 0, .92093375e5, .12246777e3, 0, 0}};
std::array<double, 10> parameters(double t) {
  std::array<double, 10> c{};
  for (std::size_t i = 0; i < c.size(); ++i)
    c[i] = coefficients[i][0] / std::pow(t, 4) + coefficients[i][1] / (t * t) +
           coefficients[i][2] / t + coefficients[i][3] +
           coefficients[i][4] * t + coefficients[i][5] * t * t;
  return c;
}
double pressure(double rho, double t,
                const std::array<double, 10> &c) { // mol/cm^3
  double d = c[1] + rho * (c[2] + rho * (c[3] + rho * (c[4] + rho * c[5])));
  double dp = c[2] + rho * (2 * c[3] + rho * (3 * c[4] + 4 * rho * c[5]));
  return R * t * 1.e6 *
         (rho + c[0] * rho * rho - rho * rho * dp / (d * d) +
          c[6] * rho * rho * std::exp(-c[7] * rho) +
          c[8] * rho * rho * std::exp(-c[9] * rho));
}
double log_fugacity(double rho, double p, double t,
                    const std::array<double, 10> &c) {
  double d = c[1] + rho * (c[2] + rho * (c[3] + rho * (c[4] + rho * c[5])));
  double ar = c[0] * rho + 1 / d - 1 / c[1] -
              c[6] / c[7] * std::expm1(-c[7] * rho) -
              c[8] / c[9] * std::expm1(-c[9] * rho);
  return std::log(rho * R * t * 1.e6 / 1.e5) + ar + p / (rho * R * t * 1.e6) -
         1; // f / 1 bar
}
struct FluidState {
  double rho, logf;
};
FluidState state(double p, double t) {
  if (!(p > 0 && p <= 5.e9 && t >= 500 && t <= 1700))
    throw std::invalid_argument(
        "Water fluid requires 0 < P <= 5 GPa and 500 <= T <= 1700 K.");
  auto c = parameters(t);
  FluidState best{0, std::numeric_limits<double>::infinity()};
  // A logarithmic scan captures vapour, unstable and liquid roots below Tc.
  // Only rising-pressure roots are mechanically stable; minimum G selects one.
  double lo = std::max(1.e-14, std::min(p / (R * t * 1.e6) * .01, 1.e-6));
  double plo = pressure(lo, t, c) - p;
  for (int k = 1; k <= 320; ++k) {
    double hi = lo * std::pow(.16 / lo, 1.0 / (321 - k));
    double phi = pressure(hi, t, c) - p;
    if (plo <= 0 && phi >= 0) {
      double a = lo, b = hi;
      for (int j = 0; j < 65; ++j) {
        double m = .5 * (a + b);
        if (pressure(m, t, c) < p)
          a = m;
        else
          b = m;
      }
      double rho = .5 * (a + b), f = log_fugacity(rho, p, t, c);
      if (std::isfinite(f) && f < best.logf)
        best = {rho, f};
    }
    lo = hi;
    plo = phi;
  }
  if (best.rho == 0.)
    throw std::runtime_error("No stable water density root found.");
  return best;
}
WaterIdealGasProperties nist_reference(double t) {
  double x = t / 1000.;
  double h =
      1000 * (30.092 * x + 6.832514 * x * x / 2 + 6.793435 * x * x * x / 3 -
              2.534480 * std::pow(x, 4) / 4 - .082139 / x - 250.8810);
  double s = 30.092 * std::log(x) + 6.832514 * x + 6.793435 * x * x / 2 -
             2.534480 * x * x * x / 3 - .082139 / (2 * x * x) + 223.3967;
  double cp = 30.092 + 6.832514 * x + 6.793435 * x * x - 2.534480 * x * x * x +
              .082139 / (x * x);
  return {h, s, cp, h - t * s};
}
WaterIdealGasProperties hp_reference(double t) {
  constexpr double t0 = 298.15, h0 = -241810., s0 = 188.80;
  // HP2011 Table 2a: Cp = a + b T + c/T^2 + d/sqrt(T), converted to SI.
  constexpr double a = 40.1, b = .008656, c = 487500., d = -251.2;
  double h = h0 + a * (t - t0) + b * (t * t - t0 * t0) / 2 -
             c * (1 / t - 1 / t0) + 2 * d * (std::sqrt(t) - std::sqrt(t0));
  double s = s0 + a * std::log(t / t0) + b * (t - t0) -
             c * (1 / (t * t) - 1 / (t0 * t0)) / 2 -
             2 * d * (1 / std::sqrt(t) - 1 / std::sqrt(t0));
  double cp = a + b * t + c / (t * t) + d / std::sqrt(t);
  return {h, s, cp, h - t * s};
}
class WaterEOS final : public EquationOfState {
public:
  explicit WaterEOS(WaterThermalReference reference) : reference_(reference) {}
  void validate_parameters(types::MineralParams &) override {}
  double compute_volume(double p, double t,
                        const types::MineralParams &) const override {
    return 1.e-6 / state(p, t).rho;
  }
  double
  compute_gibbs_free_energy(double p, double t, double,
                            const types::MineralParams &) const override {
    return gibbs(p, t);
  }
  double compute_entropy(double p, double t, double,
                         const types::MineralParams &) const override {
    double h = std::min(.01, std::min(t - 500., 1700. - t) / 2);
    if (h <= 0)
      throw std::invalid_argument(
          "Water derivatives require 500 < T < 1700 K.");
    return -(gibbs(p, t + h) - gibbs(p, t - h)) / (2 * h);
  }
  double
  compute_molar_heat_capacity_p(double p, double t, double,
                                const types::MineralParams &) const override {
    double h = std::min(.1, std::min(t - 500., 1700. - t) / 2);
    if (h <= 0)
      throw std::invalid_argument(
          "Water derivatives require 500 < T < 1700 K.");
    // Remove the leading step-size error near the critical region without
    // using a very small step, which amplifies Gibbs-energy roundoff.
    const double g = gibbs(p, t);
    const double coarse =
        ((gibbs(p, t + h) - g) + (gibbs(p, t - h) - g)) / (h * h);
    const double half_h = h / 2;
    const double fine =
        ((gibbs(p, t + half_h) - g) + (gibbs(p, t - half_h) - g)) /
        (half_h * half_h);
    return -t * (4 * fine - coarse) / 3;
  }
  double compute_isothermal_bulk_modulus_reuss(
      double p, double t, double, const types::MineralParams &) const override {
    auto c = parameters(t);
    double r = state(p, t).rho, h = r * 1.e-5;
    return r * (pressure(r + h, t, c) - pressure(r - h, t, c)) / (2 * h);
  }
  double
  compute_thermal_expansivity(double p, double t, double v,
                              const types::MineralParams &m) const override {
    return (compute_volume(p, t + .01, m) - compute_volume(p, t - .01, m)) /
           (.02 * v);
  }
  double compute_shear_modulus(double, double, double,
                               const types::MineralParams &) const override {
    return 0.;
  }

private:
  double gibbs(double p, double t) const {
    return water_ideal_gas_reference(t, reference_).gibbs +
           R * t * state(p, t).logf;
  }
  WaterThermalReference reference_;
};
} // namespace
WaterIdealGasProperties
water_ideal_gas_reference(double temperature, WaterThermalReference reference) {
  if (!(temperature > 0 && std::isfinite(temperature)))
    throw std::invalid_argument(
        "Water thermal reference requires finite T > 0.");
  switch (reference) {
  case WaterThermalReference::HollandPowell2011:
    return hp_reference(temperature);
  case WaterThermalReference::NIST:
    return nist_reference(temperature);
  }
  throw std::invalid_argument("Unknown water thermal reference.");
}

std::shared_ptr<Mineral> water_fluid() {
  return water_fluid(WaterThermalReference::HollandPowell2011);
}
std::shared_ptr<Mineral> water_fluid(WaterThermalReference reference) {
  // Validate the reference before creating an EOS with an invalid selector.
  (void)water_ideal_gas_reference(1000., reference);
  auto m = std::make_shared<Mineral>();
  m->params.formula = types::FormulaMap{{"H", 2}, {"O", 1}};
  m->params.molar_mass = .01801528;
  m->params.napfu = 3;
  m->set_name("H2O fluid (PS94)");
  m->set_method(std::make_shared<WaterEOS>(reference));
  return m;
}
} // namespace burnman::minerals
