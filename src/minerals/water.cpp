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
#include "burnman/eos/pitzer_sterner.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace burnman::minerals {
namespace {
constexpr double R = 8.314510; // EOS paper's gas constant
constexpr types::PitzerSternerCoefficients coefficients{
    {{0, 0, .24657688e6, .51359951e2, 0, 0},
     {0, 0, .58638965, -.28646939e-2, .31375577e-4, 0},
     {0, 0, -.62783840e1, .14791599e-1, .35779579e-3, .15432925e-7},
     {0, 0, 0, -.42719875, -.16325155e-4, 0},
     {0, 0, .56654978e4, -.16580167e2, .76560762e-1, 0},
     {0, 0, 0, .10917883, 0, 0},
     {.38878656e13, -.13494878e9, .30916564e6, .75591105e1, 0, 0},
     {0, 0, -.65537898e5, .18810675e3, 0, 0},
     {-.14182435e14, .18165390e9, -.19769068e6, -.23530318e2, 0, 0},
     {0, 0, .92093375e5, .12246777e3, 0, 0}}};
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
// The legacy water factory uses the paper's R and its selected caloric
// reference. All fluid mechanics and derivatives use the general EOS.
class WaterEOS final : public eos::PitzerSterner {
public:
  explicit WaterEOS(WaterThermalReference reference)
      : PitzerSterner(R), reference_(reference) {}

protected:
  void validate_thermal_parameters(types::MineralParams &) override {}
  ThermalState thermal_state(double t,
                             const types::MineralParams &) const override {
    const auto ideal = water_ideal_gas_reference(t, reference_);
    const double log_ref = std::log(R * t * 1.e6 / 1.e5);
    return {ideal.gibbs + R * t * (log_ref - 1.), ideal.entropy - R * log_ref,
            ideal.heat_capacity_p - R};
  }

private:
  WaterThermalReference reference_;
};
types::MineralParams water_parameters() {
  types::MineralParams p;
  p.formula = types::FormulaMap{{"H", 2}, {"O", 1}};
  p.molar_mass = .01801528;
  p.napfu = 3.;
  p.c_coeffs = coefficients;
  return p;
}
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
  m->params = water_parameters();
  m->params.T_min = 500.;
  m->params.T_max = 1700.;
  m->params.P_max = 5.e9;
  m->set_name("H2O fluid (PS94)");
  m->set_method(std::make_shared<WaterEOS>(reference));
  return m;
}
namespace Pitzer_Sterner_1994 {
std::shared_ptr<Mineral> H2O_Pitzer_Sterner() {
  auto m = std::make_shared<Mineral>();
  m->params = water_parameters();
  m->params.equation_of_state = types::EOSType::PitzerSterner;
  m->params.F_0 = -249677.75163034006 + 233.2550 * 298.15;
  m->params.Cv_0 = 8.31446 * 2.25;
  m->params.debye_0 = 3800.;
  m->params.Debye_n = 1.2;
  m->set_name("H2O");
  m->set_method(types::EOSType::Auto);
  return m;
}
} // namespace Pitzer_Sterner_1994
} // namespace burnman::minerals
