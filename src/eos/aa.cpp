// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/aa.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>
#include <limits>
#include <type_traits>

namespace burnman::eos {
namespace {
using Params = types::MineralParams;
void check_state(double t, double v) {
  if (!(std::isfinite(t) && t > 0. && std::isfinite(v) && v > 0.))
    throw std::invalid_argument("AA requires finite T > 0 and V > 0.");
}
double integral_power(double k, double l) {
  return k == 0. ? l : std::expm1(k * l) / k;
}
double logistic(double z) {
  const double e = std::exp(-std::abs(z));
  return z >= 0. ? e / (1. + e) : 1. / (1. + e);
}
struct Reference {
  double r, q, energy, t, g, gamma, pressure, prefactor;
};
struct Coefficients {
  double a, b, theta, xi, lambda, f, h, c;
};
struct Thermal {
  double lt, lp, se, sb, fe, fb, fl, fx, atan;
};
struct Mechanical {
  double pressure = 0., kt = 0., dpdt = 0., cv = 0.;
};
enum class Caloric { Entropy, Energy, Helmholtz };

// Prepared once per evaluation, including once for the entire volume solve.
// No mutable caches: C++ callers may edit a MineralParams between evaluations.
struct Model {
  const Params &p;
  double xi1, xi2, ck;
  std::array<double, 5> coefficients;
  explicit Model(const Params &params)
      : p(params), xi1(.75 * (4. - *p.Kprime_0)),
        xi2(.375 * (*p.K_0 * *p.Kdprime_0 + *p.Kprime_0 * (*p.Kprime_0 - 7.)) +
            143. / 24.),
        ck(1.5 * *p.napfu * constants::physics::gas_constant),
        coefficients{xi2 / 8., -(xi1 + 3. * xi2) / 6.,
                     (1. + 2. * xi1 + 3. * xi2) / 4., -(1. + xi1 + xi2) / 2.,
                     (6. + 4. * xi1 + 3. * xi2) / 24.} {}

  template <bool MechanicalNeeded> Reference reference(double v) const {
    Reference s{};
    s.r = *p.V_0 / v;
    const double l = std::log(s.r);
    s.q = std::expm1(2. * l / 3.);
    s.energy = 4.5 * *p.V_0 * *p.K_0 * s.q * s.q *
                   (.25 - xi1 * s.q / 6. + xi2 * s.q * s.q / 8.) -
               *p.P_0 * *p.V_0 * std::expm1(-l);
    double integral = 0.;
    for (std::size_t i = 0; i < coefficients.size(); ++i)
      integral +=
          coefficients[i] *
          integral_power(
              *p.grueneisen_n + (8. - 2. * static_cast<double>(i)) / 3., l);
    integral *= 4.5 * *p.V_0 * *p.K_0;
    integral += *p.P_0 * *p.V_0 *
                (integral_power(*p.grueneisen_n, l) -
                 integral_power(*p.grueneisen_n - 1., l));
    s.t =
        *p.T_0 * std::exp(*p.grueneisen_0 * l + *p.grueneisen_prime * integral);
    if constexpr (MechanicalNeeded) {
      s.g = *p.grueneisen_prime * std::exp(*p.grueneisen_n * l);
      s.gamma = *p.grueneisen_0 + s.g * s.energy;
      s.prefactor = 1.5 * *p.K_0 * std::exp(5. * l / 3.);
      s.pressure =
          *p.P_0 + s.prefactor * s.q * (1. - xi1 * s.q + xi2 * s.q * s.q);
    }
    return s;
  }
  Coefficients caloric_coefficients(double r) const {
    const auto &a = *p.aa_a, &b = *p.aa_b, &theta = *p.aa_Theta, &f = *p.aa_F;
    const auto &lambda = *p.aa_lmda;
    const double c = lambda[2] * std::pow(r, .4),
                 h = lambda[0] + lambda[1] * r + c;
    const double fraction = logistic((r - f[0]) / f[1]);
    return {a[0] + a[1] / r,
            b[0] + b[1] / (r * r),
            theta[0] * std::pow(r, theta[1]),
            *p.xi_0 * std::pow(r, -.6),
            fraction * h,
            fraction,
            h,
            c};
  }
  double cv(double t, const Coefficients &c) const {
    return ck + c.a * t * t / (c.theta * c.theta + t * t) +
           c.b * std::pow(t, .6) +
           (c.lambda * t + c.xi * *p.theta) / (*p.theta + t);
  }
  template <bool FreeEnergy>
  Thermal thermal(double t, double ts, const Coefficients &c) const {
    Thermal f{};
    const double dt = t - ts;
    f.lt = std::log1p(dt / ts);
    f.lp = std::log1p(dt / (*p.theta + ts));
    f.se = .5 * std::log1p(dt * (t + ts) / (c.theta * c.theta + ts * ts));
    f.sb = (5. / 3.) * std::pow(ts, .6) * std::expm1(.6 * f.lt);
    if constexpr (FreeEnergy) {
      f.atan = std::atan2(c.theta * dt, c.theta * c.theta + t * ts);
      f.fe = dt - c.theta * f.atan - t * f.se;
      f.fb = (5. / 8.) * std::pow(ts, 1.6) * std::expm1(1.6 * f.lt) - t * f.sb;
      f.fl = dt - (*p.theta + t) * f.lp;
      f.fx = (*p.theta + t) * f.lp - t * f.lt;
    }
    return f;
  }
  template <Caloric Kind> double caloric(double t, double v) const {
    check_state(t, v);
    const auto s = reference<false>(v);
    const auto c = caloric_coefficients(s.r);
    const double dt = t - s.t;
    if constexpr (Kind == Caloric::Energy) {
      const double lt = std::log1p(dt / s.t),
                   lp = std::log1p(dt / (*p.theta + s.t));
      const double atan = std::atan2(c.theta * dt, c.theta * c.theta + t * s.t);
      return *p.E_0 + s.energy + ck * dt + c.a * (dt - c.theta * atan) +
             (5. / 8.) * c.b * std::pow(s.t, 1.6) * std::expm1(1.6 * lt) +
             c.lambda * dt + *p.theta * (c.xi - c.lambda) * lp;
    } else {
      const auto f = thermal<Kind == Caloric::Helmholtz>(t, s.t, c);
      if constexpr (Kind == Caloric::Entropy)
        return *p.S_0 + (ck + c.xi) * f.lt + (c.lambda - c.xi) * f.lp +
               c.a * f.se + c.b * f.sb;
      else
        return *p.E_0 + s.energy - t * *p.S_0 + ck * (dt - t * f.lt) +
               c.a * f.fe + c.b * f.fb + c.lambda * f.fl + c.xi * f.fx;
    }
  }
  // Only instantiate the derivatives needed by the requested property.
  // The pressure-only root path never evaluates second volume derivatives.
  template <bool ThermalPressure, bool Bulk, bool HeatCapacity>
  Mechanical mechanical(double t, double v) const {
    check_state(t, v);
    const auto s = reference<true>(v);
    const auto c = caloric_coefficients(s.r);
    const auto f = thermal<true>(t, s.t, c);
    const double dt = t - s.t, den = c.theta * c.theta + s.t * s.t;
    const double al = -(*p.aa_a)[1] / s.r,
                 bl = -2. * (*p.aa_b)[1] / (s.r * s.r);
    const double thetal = (*p.aa_Theta)[1] * c.theta, xil = -.6 * c.xi;
    const double fl = -c.f * (1. - c.f) * s.r / (*p.aa_F)[1];
    const double hl = (*p.aa_lmda)[1] * s.r + .4 * c.c;
    const double lambdal = fl * c.h + c.f * hl;
    const double fe_theta = -f.atan + c.theta * dt / den;
    const double cvs = cv(s.t, c);
    const double f_l = al * f.fe + c.a * thetal * fe_theta + bl * f.fb +
                       lambdal * f.fl + xil * f.fx + s.gamma * cvs * dt;
    Mechanical result;
    result.pressure = s.pressure + f_l / v;
    if constexpr (HeatCapacity)
      result.cv = cv(t, c);
    if constexpr (ThermalPressure) {
      const double sl = al * f.se +
                        c.a * thetal * c.theta *
                            (1. / (c.theta * c.theta + t * t) - 1. / den) +
                        bl * f.sb + (lambdal - xil) * f.lp + xil * f.lt;
      result.dpdt = (s.gamma * cvs - sl) / v;
    }
    if constexpr (Bulk) {
      const double fll =
          c.f * (1. - c.f) * (1. - 2. * c.f) * std::pow(s.r / (*p.aa_F)[1], 2) +
          fl;
      const double hll = (*p.aa_lmda)[1] * s.r + .16 * c.c;
      const double lambdall = fll * c.h + 2. * fl * hl + c.f * hll;
      const double fe_theta2 =
          t / (c.theta * c.theta + t * t) - s.t / den +
          dt * (s.t * s.t - c.theta * c.theta) / (den * den);
      const double cvl = al * s.t * s.t / den -
                         2. * c.a * s.t * s.t * c.theta * thetal / (den * den) +
                         bl * std::pow(s.t, .6) +
                         (lambdal * s.t + xil * *p.theta) / (*p.theta + s.t);
      const double cvt =
          2. * c.a * s.t * c.theta * c.theta / (den * den) +
          .6 * c.b * std::pow(s.t, -.4) +
          *p.theta * (c.lambda - c.xi) / std::pow(*p.theta + s.t, 2);
      const double gl = s.g * (*p.grueneisen_n * s.energy + v * s.pressure);
      const double f_ll =
          -al * f.fe +
          (2. * al * thetal + c.a * (*p.aa_Theta)[1] * thetal) * fe_theta +
          c.a * thetal * thetal * fe_theta2 - 2. * bl * f.fb + lambdall * f.fl +
          .36 * c.xi * f.fx + 2. * s.gamma * dt * cvl +
          s.gamma * s.gamma * s.t * dt * cvt +
          cvs * (dt * gl - s.t * s.gamma * s.gamma);
      const double ks =
          s.prefactor * ((5. / 3.) * s.q * (1. - xi1 * s.q + xi2 * s.q * s.q) +
                         (2. / 3.) * (s.q + 1.) *
                             (1. - 2. * xi1 * s.q + 3. * xi2 * s.q * s.q));
      result.kt = ks + (f_l + f_ll) / v;
    }
    return result;
  }
};
struct VolumeResidual {
  const Model &model;
  double pressure, temperature, v0;
};
double residual(double ratio, void *data) {
  const auto &s = *static_cast<VolumeResidual *>(data);
  return s.model.mechanical<false, false, false>(s.temperature, ratio * s.v0)
             .pressure -
         s.pressure;
}
template <class T>
void require_finite(const std::optional<T> &value, const char *name) {
  utils::require_set(value, name);
  if constexpr (std::is_same_v<T, double>) {
    if (std::isfinite(*value))
      return;
  } else {
    bool finite = true;
    for (double x : *value)
      finite = finite && std::isfinite(x);
    if (finite)
      return;
  }
  throw std::invalid_argument(std::string("AA requires finite ") + name + ".");
}
} // namespace

void AA::validate_parameters(Params &p) {
#define REQUIRED(name) require_finite(p.name, #name)
  REQUIRED(P_0);
  REQUIRED(T_0);
  REQUIRED(E_0);
  REQUIRED(S_0);
  REQUIRED(V_0);
  REQUIRED(K_0);
  REQUIRED(Kprime_0);
  REQUIRED(Kdprime_0);
  REQUIRED(napfu);
  REQUIRED(molar_mass);
  REQUIRED(grueneisen_0);
  REQUIRED(grueneisen_prime);
  REQUIRED(grueneisen_n);
  REQUIRED(theta);
  REQUIRED(xi_0);
  REQUIRED(aa_a);
  REQUIRED(aa_b);
  REQUIRED(aa_Theta);
  REQUIRED(aa_lmda);
  REQUIRED(aa_F);
#undef REQUIRED
  for (double x : {*p.T_0, *p.V_0, *p.K_0, *p.napfu, *p.molar_mass, *p.theta,
                   (*p.aa_Theta)[0], (*p.aa_F)[1]})
    if (!(x > 0.))
      throw std::invalid_argument("AA requires positive reference T, V, K_S, "
                                  "n, mass, theta, Theta[0] and F[1].");
}
double AA::compute_volume(double pressure, double t, const Params &p) const {
  check_state(t, *p.V_0);
  if (!std::isfinite(pressure))
    throw std::invalid_argument("AA requires finite pressure.");
  const Model model(p);
  VolumeResidual data{model, pressure, t, *p.V_0};
  const double lo = residual(.1, &data), hi = residual(2., &data);
  if (!(std::isfinite(lo) && std::isfinite(hi)) ||
      (lo != 0. && hi != 0. && std::signbit(lo) == std::signbit(hi)))
    throw std::invalid_argument(
        "Cannot bracket an AA volume in 0.1 <= V/V_0 <= 2.");
  if (lo == 0.)
    return .1 * *p.V_0;
  if (hi == 0.)
    return 2. * *p.V_0;
  return *p.V_0 *
         optim::roots::brent(&residual, data, .1, 2., 1.e-14,
                             4. * std::numeric_limits<double>::epsilon());
}
double AA::compute_pressure(double t, double v, const Params &p) const {
  return Model(p).mechanical<false, false, false>(t, v).pressure;
}
double AA::compute_isothermal_bulk_modulus_reuss(double, double t, double v,
                                                 const Params &p) const {
  return Model(p).mechanical<false, true, false>(t, v).kt;
}
double AA::compute_thermal_expansivity(double, double t, double v,
                                       const Params &p) const {
  const auto s = Model(p).mechanical<true, true, false>(t, v);
  return s.dpdt / s.kt;
}
double AA::compute_grueneisen_parameter(double, double t, double v,
                                        const Params &p) const {
  const auto s = Model(p).mechanical<true, false, true>(t, v);
  return v * s.dpdt / s.cv;
}
double AA::compute_molar_heat_capacity_v(double, double t, double v,
                                         const Params &p) const {
  check_state(t, v);
  const Model m(p);
  return m.cv(t, m.caloric_coefficients(*p.V_0 / v));
}
double AA::compute_molar_heat_capacity_p(double, double t, double v,
                                         const Params &p) const {
  const auto s = Model(p).mechanical<true, true, true>(t, v);
  return s.cv + t * v * s.dpdt * s.dpdt / s.kt;
}
double AA::compute_isentropic_bulk_modulus_reuss(double, double t, double v,
                                                 const Params &p) const {
  const auto s = Model(p).mechanical<true, true, true>(t, v);
  return s.kt + t * v * s.dpdt * s.dpdt / s.cv;
}
double AA::compute_shear_modulus(double, double, double, const Params &) const {
  return 0.;
}
double AA::compute_entropy(double, double t, double v, const Params &p) const {
  return Model(p).caloric<Caloric::Entropy>(t, v);
}
double AA::compute_helmholtz_free_energy(double, double t, double v,
                                         const Params &p) const {
  return Model(p).caloric<Caloric::Helmholtz>(t, v);
}
double AA::compute_molar_internal_energy(double, double t, double v,
                                         const Params &p) const {
  return Model(p).caloric<Caloric::Energy>(t, v);
}
double AA::compute_gibbs_free_energy(double pressure, double t, double v,
                                     const Params &p) const {
  return Model(p).caloric<Caloric::Helmholtz>(t, v) + pressure * v;
}
double AA::compute_enthalpy(double pressure, double t, double v,
                            const Params &p) const {
  return Model(p).caloric<Caloric::Energy>(t, v) + pressure * v;
}
} // namespace burnman::eos
