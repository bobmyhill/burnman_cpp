// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/brosh_calphad.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
constexpr double R = constants::physics::gas_constant;

// Appendix A. Evaluate Gamma(X)-Gamma(1) directly, avoiding subtraction of
// the large antiderivative constants at low pressure.
double delta_gamma(int n, double a, double log_x) {
  long double sum = 0., binomial = 1.;
  const long double an = a, lx = log_x;
  for (int k = 1; k <= n; ++k) {
    binomial *= static_cast<long double>(n - k + 1) / k;
    const long double d = k == 3 ? -3.L * lx
                                 : static_cast<long double>(k) / (k - 3) *
                                       std::expm1((3 - k) * lx);
    sum += binomial * std::pow(an - 1.L, n - k) * d;
  }
  return static_cast<double>(3.L * std::pow(an, 1 - n) / n * sum);
}

double x_lower_limit(double n, double a, double k) {
  // Both the fractional-power base and X's denominator must stay positive.
  return 3. * a * k / n * (a > 1. ? std::expm1(n * std::log1p(-1. / a)) : -1.);
}
double pressure_lower_limit(const Params &p) {
  double lower = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < 4; ++i)
    lower = std::max(lower, x_lower_limit(static_cast<double>(i + 2),
                                          (*p.brosh_a)[i], *p.K_0));
  lower = std::max(lower, x_lower_limit(2., 1. / (3. * *p.b_0 - 1.),
                                        *p.K_0 / (1. + *p.delta_0)));
  return std::max(lower, -*p.K_0 / (2. * *p.b_1 * (1. + *p.delta_1)));
}
void check_state(double pressure, double t, const Params &p) {
  if (!std::isfinite(pressure) || !std::isfinite(t) || t < 0. ||
      pressure <= pressure_lower_limit(p))
    throw std::invalid_argument("BroshCalphad requires finite T >= 0 and "
                                "pressure inside its elastic domain.");
}
struct Caloric {
  double g = 0., s = 0., cp = 0.;
};
Caloric sgte(double t, const Params &p) {
  const auto *c = &p.gibbs_coefficients->back().second;
  for (const auto &segment : *p.gibbs_coefficients)
    if (t < segment.first) {
      c = &segment.second;
      break;
    }
  const double log_t = std::log(t);
  Caloric out{(*c)[0] + (*c)[1] * t + (*c)[2] * t * log_t + (*c)[12] * log_t,
              -(*c)[1] - (*c)[2] * (1. + log_t) - (*c)[12] / t,
              -(*c)[2] + (*c)[12] / t};
  constexpr std::array<double, 9> powers{-1., -2., -3., -9., 2.,
                                         3.,  4.,  7.,  .5};
  for (std::size_t i = 0; i < powers.size(); ++i) {
    if ((*c)[i + 3] == 0.)
      continue;
    const double exponent = powers[i],
                 term = (*c)[i + 3] * std::pow(t, exponent);
    out.g += term;
    out.s -= exponent * term / t;
    out.cp -= exponent * (exponent - 1.) * term / t;
  }
  return out;
}
struct EinsteinState {
  Caloric caloric;
  double occupation = 0.;
};
EinsteinState quasiharmonic(double t, double theta, double n) {
  if (t == 0.)
    return {};
  const double x = theta / t;
  if (x > 745.)
    return {};
  const double y = std::exp(-x), denominator = -std::expm1(-x);
  const double log_d = std::log(denominator), prefactor = 3. * n * R;
  return {{prefactor * t * log_d, prefactor * (x * y / denominator - log_d),
           prefactor * x * x * y / (denominator * denominator)},
          y / denominator};
}
struct Correction {
  double c, first, second;
};
Correction correction(double t, const Params &p) {
  if (t < *p.T_0) {
    const double second = *p.delta_Cpr / *p.T_0;
    return {.5 * t * t * second, t * second, second};
  }
  const auto ref = sgte(t, p),
             qh = quasiharmonic(t, *p.theta_0, *p.napfu).caloric;
  return {qh.g - ref.g + *p.delta_Gr - (t - *p.T_0) * *p.delta_Sr +
              (t - .5 * *p.T_0) * *p.delta_Cpr,
          ref.s - qh.s - *p.delta_Sr + *p.delta_Cpr, (ref.cp - qh.cp) / t};
}
struct Cold {
  double v = 0., dvdp = 0., g = 0.;
};
template <bool FreeEnergy> Cold cold(double pressure, const Params &p) {
  Cold out;
  for (std::size_t i = 0; i < 4; ++i) {
    const double n = static_cast<double>(i + 2), a = (*p.brosh_a)[i],
                 c = (*p.brosh_c)[i];
    const double log_base = std::log1p(n * pressure / (3. * a * *p.K_0));
    const double log_x = -std::log1p(a * std::expm1(log_base / n));
    out.v += c * std::exp(3. * log_x);
    out.dvdp -= c * std::exp(4. * log_x + (1. / n - 1.) * log_base) / *p.K_0;
    if constexpr (FreeEnergy)
      out.g += c * delta_gamma(static_cast<int>(i + 2), a, log_x);
  }
  out.v *= *p.V_0;
  out.dvdp *= *p.V_0;
  out.g *= *p.K_0 * *p.V_0;
  return out;
}
struct PressureState {
  double theta, logtheta_first, theta_first, theta_second, i, i_first, i_second;
};
PressureState pressure_state(double pressure, const Params &p) {
  const double a = 1. / (3. * *p.b_0 - 1.), k = *p.K_0 / (1. + *p.delta_0);
  const double log_base = std::log1p(2. * pressure / (3. * a * k));
  const double y = std::exp(.5 * log_base);
  const double log_x = -std::log1p(a * std::expm1(.5 * log_base));
  const double x = std::exp(log_x);
  const double theta =
      *p.theta_0 *
      std::exp(*p.grueneisen_0 / (1. + *p.delta_0) * delta_gamma(2, a, log_x));
  const double slope = *p.grueneisen_0 / *p.K_0 * x * x * x;
  const double f =
      std::sqrt(1. + 2. * *p.b_1 * (1. + *p.delta_1) * pressure / *p.K_0);
  const double factor = std::exp((1. - f) / *p.b_1) / (1. + *p.b_1);
  const double i_first = -(1. + *p.delta_1) / *p.K_0 * factor;
  return {theta,
          slope,
          theta * slope,
          theta * (slope * slope - slope * x / (k * y)),
          (*p.b_1 + f) * factor,
          i_first,
          -i_first * (1. + *p.delta_1) / (*p.K_0 * f)};
}
double volume_at(double pressure, double t, double c, const Params &p) {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  const auto qh = quasiharmonic(t, s.theta, *p.napfu);
  return cold<false>(pressure, p).v +
         3. * *p.napfu * R * qh.occupation * s.theta_first - c * s.i_first;
}
struct VolumeResidual {
  const Params &p;
  double t, c, volume;
};
double residual(double scaled_pressure, void *data) {
  const auto &s = *static_cast<VolumeResidual *>(data);
  return (volume_at(scaled_pressure * *s.p.K_0, s.t, s.c, s.p) - s.volume) /
         *s.p.V_0;
}
void require_finite(const std::optional<double> &value, const char *name) {
  utils::require_set(value, name);
  if (!std::isfinite(*value))
    throw std::invalid_argument(
        std::string("Nonfinite BroshCalphad parameter: ") + name);
}
} // namespace

void BroshCalphad::validate_parameters(Params &p) {
  require_finite(p.V_0, "V_0");
  require_finite(p.K_0, "K_0");
  require_finite(p.Kprime_0, "Kprime_0");
  require_finite(p.napfu, "napfu");
  require_finite(p.theta_0, "theta_0");
  require_finite(p.grueneisen_0, "grueneisen_0");
  require_finite(p.delta_0, "delta_0");
  require_finite(p.delta_1, "delta_1");
  require_finite(p.b_0, "b_0");
  require_finite(p.b_1, "b_1");
  if (*p.V_0 <= 0. || *p.K_0 <= 0. || *p.Kprime_0 <= 1. / 3. ||
      *p.napfu <= 0. || *p.theta_0 <= 0. || *p.grueneisen_0 < 0. ||
      *p.delta_0 <= -1. || *p.delta_1 <= -1. || *p.b_0 <= 1. / 3. ||
      *p.b_1 <= 0.)
    throw std::invalid_argument(
        "Invalid BroshCalphad reference or thermal parameters.");
  utils::require_set(p.gibbs_coefficients, "gibbs_coefficients");
  if (p.gibbs_coefficients->empty())
    throw std::invalid_argument(
        "BroshCalphad requires at least one SGTE segment.");
  double previous = 0.;
  for (const auto &segment : *p.gibbs_coefficients) {
    if (!(segment.first > previous))
      throw std::invalid_argument(
          "SGTE upper temperatures must be positive and strictly increasing.");
    for (double value : segment.second)
      if (!std::isfinite(value))
        throw std::invalid_argument("SGTE coefficients must be finite.");
    previous = segment.first;
  }
  p.T_0 = 298.15; // Appendix C's SGTE reference temperature.
  utils::fallback_to_default(p.P_0, 1.e5);
  if (!p.brosh_a) {
    std::array<double, 4> a{};
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = static_cast<double>(i + 1) / (3. * *p.Kprime_0 - 1.);
    p.brosh_a = a;
  }
  for (double a : *p.brosh_a)
    if (!std::isfinite(a) || a <= 0.)
      throw std::invalid_argument(
          "Brosh cold-curve a coefficients must be positive and finite.");
  if (!p.brosh_c) {
    utils::require_set(p.formula, "formula (to derive Brosh cold-curve c)");
    Eigen::Vector4d target(1., 8., 0., 0.);
    if (p.formula->empty())
      throw std::invalid_argument("Brosh formula must not be empty.");
    for (const auto &[element, count] : *p.formula) {
      if (!std::isfinite(count) || count < 0.)
        throw std::invalid_argument(
            "Brosh formula amounts must be nonnegative and finite.");
      if (count == 0.)
        continue;
      const auto it = constants::chemistry::atomic_numbers.find(element);
      if (it == constants::chemistry::atomic_numbers.end())
        throw std::invalid_argument(
            "Unknown atomic number for Brosh formula element: " + element);
      const double z = std::log(static_cast<double>(it->second));
      target[2] += count * .02713 * std::exp(.97626 * z - .057848 * z * z) *
                   1.e-6 / *p.V_0;
      target[3] += count * .025692 * std::exp(.97914 * z - .057741 * z * z) *
                   1.e-6 / *p.V_0;
    }
    if (target[2] <= 0.)
      throw std::invalid_argument("Brosh formula must contain atoms.");
    Eigen::Matrix4d matrix;
    matrix.row(0).setOnes();
    matrix.row(1) << 0., 6., 8., 9.;
    for (int j = 0; j < 2; ++j)
      for (std::size_t i = 0; i < 4; ++i) {
        const double a = (*p.brosh_a)[i], n = static_cast<double>(i + 2);
        matrix(j + 2, static_cast<Eigen::Index>(i)) =
            std::pow(1. - a +
                         a * std::pow(1. + n * (300.e12 + 30.e12 * j) /
                                               (3. * a * *p.K_0),
                                      1. / n),
                     -3.);
      }
    const auto lu = matrix.fullPivLu();
    if (!lu.isInvertible())
      throw std::invalid_argument("Singular Brosh cold-curve constraints.");
    const Eigen::Vector4d c = lu.solve(target);
    p.brosh_c = std::array<double, 4>{c[0], c[1], c[2], c[3]};
  }
  for (double c : *p.brosh_c)
    if (!std::isfinite(c))
      throw std::invalid_argument(
          "Brosh cold-curve c coefficients must be finite.");
  const auto ref = sgte(*p.T_0, p),
             qh = quasiharmonic(*p.T_0, *p.theta_0, *p.napfu).caloric;
  p.delta_Gr = ref.g - qh.g;
  p.delta_Sr = ref.s - qh.s;
  p.delta_Cpr = ref.cp - qh.cp;
}

double BroshCalphad::compute_volume(double pressure, double t,
                                    const Params &p) const {
  check_state(pressure, t, p);
  return volume_at(pressure, t, correction(t, p).c, p);
}
double BroshCalphad::compute_pressure(double t, double volume,
                                      const Params &p) const {
  check_state(0., t, p);
  if (!std::isfinite(volume) || volume <= 0.)
    throw std::invalid_argument(
        "BroshCalphad requires positive finite volume.");
  VolumeResidual data{p, t, correction(t, p).c, volume};
  const double f0 = residual(0., &data);
  if (f0 == 0.)
    return 0.;
  double lo = 0., hi = 1.;
  if (f0 > 0.) {
    int step = 0;
    while (residual(hi, &data) > 0. && step++ < 100)
      hi *= 2.;
  } else {
    hi = 0.;
    const double limit = pressure_lower_limit(p) / *p.K_0;
    lo = .5 * limit;
    int step = 0;
    while (residual(lo, &data) < 0. && step++ < 100)
      lo = .5 * (lo + limit);
  }
  const double flo = residual(lo, &data), fhi = residual(hi, &data);
  if (!(std::isfinite(flo) && std::isfinite(fhi)) || flo * fhi > 0.)
    throw std::invalid_argument(
        "Cannot bracket a BroshCalphad pressure for this volume.");
  if (flo == 0.)
    return lo * *p.K_0;
  if (fhi == 0.)
    return hi * *p.K_0;
  return *p.K_0 *
         optim::roots::brent(&residual, data, lo, hi, 1.e-14,
                             4. * std::numeric_limits<double>::epsilon());
}
double BroshCalphad::compute_gibbs_free_energy(double pressure, double t,
                                               double, const Params &p) const {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  const double reference = *p.delta_Gr + *p.T_0 * *p.delta_Sr -
                           .5 * *p.T_0 * *p.delta_Cpr +
                           t * (*p.delta_Cpr - *p.delta_Sr);
  return cold<true>(pressure, p).g +
         quasiharmonic(t, s.theta, *p.napfu).caloric.g -
         correction(t, p).c * s.i + reference;
}
double BroshCalphad::compute_entropy(double pressure, double t, double,
                                     const Params &p) const {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  return quasiharmonic(t, s.theta, *p.napfu).caloric.s +
         correction(t, p).first * s.i + *p.delta_Sr - *p.delta_Cpr;
}
double BroshCalphad::compute_molar_heat_capacity_p(double pressure, double t,
                                                   double,
                                                   const Params &p) const {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  return quasiharmonic(t, s.theta, *p.napfu).caloric.cp +
         t * correction(t, p).second * s.i;
}
double BroshCalphad::compute_thermal_expansivity(double pressure, double t,
                                                 double volume,
                                                 const Params &p) const {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  return (quasiharmonic(t, s.theta, *p.napfu).caloric.cp * s.logtheta_first -
          correction(t, p).first * s.i_first) /
         volume;
}
double BroshCalphad::compute_isothermal_bulk_modulus_reuss(
    double pressure, double t, double volume, const Params &p) const {
  check_state(pressure, t, p);
  const auto s = pressure_state(pressure, p);
  const auto qh = quasiharmonic(t, s.theta, *p.napfu);
  double derivative =
      cold<false>(pressure, p).dvdp - correction(t, p).c * s.i_second;
  if (t > 0.)
    derivative += 3. * *p.napfu * R * qh.occupation *
                  (s.theta_second -
                   (1. + qh.occupation) * s.theta_first * s.theta_first / t);
  return -volume / derivative;
}
double BroshCalphad::compute_molar_heat_capacity_v(double pressure, double t,
                                                   double volume,
                                                   const Params &p) const {
  const double alpha = compute_thermal_expansivity(pressure, t, volume, p);
  return compute_molar_heat_capacity_p(pressure, t, volume, p) -
         t * volume * alpha * alpha *
             compute_isothermal_bulk_modulus_reuss(pressure, t, volume, p);
}
double BroshCalphad::compute_isentropic_bulk_modulus_reuss(
    double pressure, double t, double volume, const Params &p) const {
  const double kt =
      compute_isothermal_bulk_modulus_reuss(pressure, t, volume, p);
  return t == 0. ? kt
                 : kt * compute_molar_heat_capacity_p(pressure, t, volume, p) /
                       compute_molar_heat_capacity_v(pressure, t, volume, p);
}
double BroshCalphad::compute_grueneisen_parameter(double pressure, double t,
                                                  double volume,
                                                  const Params &p) const {
  const double kt =
      compute_isothermal_bulk_modulus_reuss(pressure, t, volume, p);
  const double cv = compute_molar_heat_capacity_v(pressure, t, volume, p);
  if (cv != 0.)
    return compute_thermal_expansivity(pressure, t, volume, p) * kt * volume /
           cv;
  const auto s = pressure_state(pressure, p);
  // T->0: the Appendix C term is linear in Cp and dominates the oscillator.
  return kt *
         (*p.delta_Cpr != 0. && s.i > 0. ? -s.i_first / s.i : s.logtheta_first);
}
double BroshCalphad::compute_shear_modulus(double, double, double,
                                           const Params &) const {
  return 0.;
}
double BroshCalphad::compute_helmholtz_free_energy(double pressure, double t,
                                                   double volume,
                                                   const Params &p) const {
  return compute_gibbs_free_energy(pressure, t, volume, p) - pressure * volume;
}
double BroshCalphad::compute_enthalpy(double pressure, double t, double volume,
                                      const Params &p) const {
  return compute_gibbs_free_energy(pressure, t, volume, p) +
         t * compute_entropy(pressure, t, volume, p);
}
double BroshCalphad::compute_molar_internal_energy(double pressure, double t,
                                                   double volume,
                                                   const Params &p) const {
  return compute_enthalpy(pressure, t, volume, p) - pressure * volume;
}
} // namespace burnman::eos
