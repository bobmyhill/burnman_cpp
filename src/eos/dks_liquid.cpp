// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#include "burnman/eos/dks_liquid.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include "dks_common.hpp"
#include <vector>

namespace burnman::eos {
namespace {
constexpr double R = constants::physics::gas_constant;
// Historical constants used by Python BurnMan's liquid free-energy fits.
// Keep them local, as for the PS1994 gas constant.
constexpr double boltzmann = 1.3806488e-23;
constexpr double dirac = 1.054571726e-34;
// x^i/i!, including at x=0. Derivatives are the preceding basis entries.
std::vector<double> expansion(double x, std::size_t count) {
  std::vector<double> result(count, 1.);
  for (std::size_t i = 1; i < count; ++i)
    result[i] = result[i - 1] * x / static_cast<double>(i);
  return result;
}
struct Spin {
  double a, b, d, dv, dvv, atoms;
};
Spin spin(double t, double v, const types::MineralParams &p) {
  if (!p.spin_a)
    return {0., 0., 1., 0., 0., 0.};
  const auto &a = *p.spin_a;
  const auto &b = *p.spin_b;
  const double x = v / *p.V_0;
  const double sa = a[0] + a[1] * x;
  const double sb = b[0] + b[1] / x + b[2] / (x * x) + b[3] / (x * x * x);
  const double d = 2. * (sa * t + sb) + 1.;
  if (!(d > 0.))
    throw std::domain_error("DKS magnetic spin degeneracy must be positive.");
  const double dv = -2. *
                    (-a[1] * t + b[1] / (x * x) + 2. * b[2] / std::pow(x, 3.) +
                     3. * b[3] / std::pow(x, 4.)) /
                    *p.V_0;
  const double dvv =
      2. *
      (2. * b[1] / std::pow(x, 3.) + 6. * b[2] / std::pow(x, 4.) +
       12. * b[3] / std::pow(x, 5.)) /
      (*p.V_0 * *p.V_0);
  const auto iron = p.formula->find("Fe");
  return {sa, sb, d, dv, dvv, iron == p.formula->end() ? 0. : iron->second};
}
double magnetic_pressure(double t, double v, const types::MineralParams &p) {
  const auto s = spin(t, v, p);
  return s.atoms * R * t * s.dv / s.d;
}
} // namespace

void DKS_L::validate_parameters(types::MineralParams &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.T_0, "T_0");
  utils::require_set(p.order_theta, "O_theta");
  utils::require_set(p.order_f, "O_f");
  utils::require_set(p.m, "m");
  utils::require_set(p.a, "a");
  utils::require_set(p.zeta_0, "zeta_0");
  utils::require_set(p.xi, "xi");
  utils::require_set(p.Tel_0, "Tel_0");
  utils::require_set(p.eta, "eta");
  utils::require_set(p.formula, "formula");
  utils::fallback_to_default(p.el_V_0, *p.V_0);
  dks_detail::check_state(*p.T_0, *p.V_0);
  dks_detail::check_state(*p.Tel_0, *p.el_V_0);
  if (*p.order_f < 0 || *p.order_theta < 0 ||
      p.a->size() != static_cast<std::size_t>(*p.order_f) + 1)
    throw std::invalid_argument("DKS a must have O_f + 1 rows.");
  for (const auto &row : *p.a) {
    if (row.size() != static_cast<std::size_t>(*p.order_theta) + 1)
      throw std::invalid_argument("DKS a must have O_theta + 1 columns.");
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("DKS bonding coefficients must be finite.");
  }
  double atoms = 0.;
  for (const auto &[element, n] : *p.formula) {
    const auto mass = constants::chemistry::atomic_masses.find(element);
    if (!std::isfinite(n) || n < 0. ||
        mass == constants::chemistry::atomic_masses.end() ||
        (n > 1.e-5 && mass->second <= 0.))
      throw std::invalid_argument(
          "DKS requires valid elemental masses and atom counts.");
    atoms += n;
  }
  if (!(atoms > 0.))
    throw std::invalid_argument("DKS formula must contain atoms.");
  if (p.spin_a)
    utils::require_set(p.spin_b, "spin_b");
}

DKS_L::State DKS_L::compute_state(double t, double v,
                                  const types::MineralParams &p) {
  dks_detail::check_state(t, v);
  // Ideal gas, Eq. S6. Masses and constants retain BurnMan's precision.
  double fig = 0., sig = 0., atoms = 0.;
  for (const auto &[element, n] : *p.formula) {
    atoms += n;
    if (n > 1.e-5) {
      const double mass = constants::chemistry::atomic_masses.at(element) /
                          constants::physics::avogadro;
      const double lnq = 1.5 * std::log(t) +
                         1.5 * std::log(mass * boltzmann /
                                        (2. * std::acos(-1.) * dirac * dirac));
      const double term = std::log(v / constants::physics::avogadro) + lnq;
      fig += -n * (term + 1.) + n * std::log(n);
      sig += n * (term + 2.5) - n * std::log(n);
    }
  }
  State s{R * t * fig,       R * sig,         atoms * R * t / v,
          atoms * R * t / v, 1.5 * R * atoms, atoms * R / v};

  // Electronic excitation, Eqs. S4-S5, zero below the conductivity onset.
  const double zeta = *p.zeta_0 * std::pow(v / *p.el_V_0, *p.xi);
  const double tel = *p.Tel_0 * std::pow(v / *p.el_V_0, *p.eta);
  if (t >= tel) {
    const double zv = zeta * *p.xi / v;
    const double zvv = zv * (*p.xi - 1.) / v;
    const double tv = tel * *p.eta / v;
    const double tvv = tv * (*p.eta - 1.) / v;
    const double ln = std::log(t / tel);
    const double gimel = .5 * (t * t - tel * tel) - t * tel * ln;
    const double ge = (t - tel) - t * ln;
    const double gt = (t - tel) - tel * ln;
    s.helmholtz -= zeta * gimel;
    s.entropy += zeta * gt;
    s.pressure += zv * gimel + zeta * tv * ge;
    s.bulk_modulus -= v * (zvv * gimel + 2. * zv * ge * tv +
                           zeta * (tvv * ge + tv * tv * (t / tel - 1.)));
    s.alpha_kt += zv * gt - zeta * ln * tv;
    s.heat_capacity_v += zeta * (t - tel);
  }

  // Excess (bonding) free energy, Eq. S2, with exact polynomial derivatives.
  const double x23 = std::pow(*p.V_0 / v, 2. / 3.);
  const double f = .5 * (x23 - 1.);
  const double fv = -x23 / (3. * v), fvv = 5. * x23 / (9. * v * v);
  const double tm = std::pow(t / *p.T_0, *p.m), theta = tm - 1.;
  const double tt = *p.m * tm / t, ttt = *p.m * (*p.m - 1.) * tm / (t * t);
  const auto fb = expansion(f, p.a->size());
  const auto tb = expansion(theta, p.a->front().size());
  double energy = 0., ef = 0., et = 0., eff = 0., eft = 0., ett = 0.;
  for (std::size_t i = 0; i < fb.size(); ++i) {
    for (std::size_t j = 0; j < tb.size(); ++j) {
      const double a = (*p.a)[i][j];
      energy += a * fb[i] * tb[j];
      if (i > 0)
        ef += a * fb[i - 1] * tb[j];
      if (j > 0)
        et += a * fb[i] * tb[j - 1];
      if (i > 1)
        eff += a * fb[i - 2] * tb[j];
      if (j > 1)
        ett += a * fb[i] * tb[j - 2];
      if (i > 0 && j > 0)
        eft += a * fb[i - 1] * tb[j - 1];
    }
  }
  s.helmholtz += energy;
  s.entropy -= tt * et;
  s.pressure -= fv * ef;
  s.bulk_modulus += v * (fvv * ef + fv * fv * eff);
  s.alpha_kt -= fv * tt * eft;
  s.heat_capacity_v -= t * (ttt * et + tt * tt * ett);

  // Optional Ramo-Stixrude magnetic terms; keep the upstream 1 K central
  // difference for alpha*K_T, including its choice of step.
  if (p.spin_a) {
    const auto mag = spin(t, v, p);
    const double nR = mag.atoms * R;
    s.helmholtz -= nR * t * std::log(mag.d);
    s.entropy += nR * (2. * mag.a * t / mag.d + std::log(mag.d));
    s.pressure += nR * t * mag.dv / mag.d;
    s.bulk_modulus -=
        v * nR * t * (mag.dvv / mag.d - std::pow(mag.dv / mag.d, 2.));
    s.heat_capacity_v +=
        nR * t * 4. * mag.a * (mag.a * t + 2. * mag.b + 1.) / (mag.d * mag.d);
    s.alpha_kt +=
        magnetic_pressure(t + .5, v, p) - magnetic_pressure(t - .5, v, p);
  }
  return s;
}
double DKS_L::compute_volume(double pressure, double temperature,
                             const types::MineralParams &p) const {
  return dks_detail::volume(*this, pressure, temperature, p);
}
double DKS_L::compute_pressure(double t, double v,
                               const types::MineralParams &p) const {
  return compute_state(t, v, p).pressure;
}
double DKS_L::compute_isothermal_bulk_modulus_reuss(
    double, double t, double v, const types::MineralParams &p) const {
  return compute_state(t, v, p).bulk_modulus;
}
double
DKS_L::compute_molar_heat_capacity_v(double, double t, double v,
                                     const types::MineralParams &p) const {
  return compute_state(t, v, p).heat_capacity_v;
}
double DKS_L::compute_thermal_expansivity(double, double t, double v,
                                          const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.alpha_kt / s.bulk_modulus;
}
double
DKS_L::compute_grueneisen_parameter(double, double t, double v,
                                    const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.alpha_kt * v / s.heat_capacity_v;
}
double
DKS_L::compute_molar_heat_capacity_p(double, double t, double v,
                                     const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.heat_capacity_v + t * v * s.alpha_kt * s.alpha_kt / s.bulk_modulus;
}
double DKS_L::compute_isentropic_bulk_modulus_reuss(
    double, double t, double v, const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.bulk_modulus + t * v * s.alpha_kt * s.alpha_kt / s.heat_capacity_v;
}
double DKS_L::compute_shear_modulus(double, double, double,
                                    const types::MineralParams &) const {
  return 0.;
}
double
DKS_L::compute_helmholtz_free_energy(double, double t, double v,
                                     const types::MineralParams &p) const {
  return compute_state(t, v, p).helmholtz;
}
double DKS_L::compute_entropy(double, double t, double v,
                              const types::MineralParams &p) const {
  return compute_state(t, v, p).entropy;
}
double DKS_L::compute_gibbs_free_energy(double pressure, double t, double v,
                                        const types::MineralParams &p) const {
  return compute_state(t, v, p).helmholtz + pressure * v;
}
double
DKS_L::compute_molar_internal_energy(double, double t, double v,
                                     const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.helmholtz + t * s.entropy;
}
double DKS_L::compute_enthalpy(double pressure, double t, double v,
                               const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.helmholtz + t * s.entropy + pressure * v;
}
} // namespace burnman::eos
