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

#include "burnman/eos/modular_mie_grueneisen_debye.hpp"
#include "burnman/eos/birch_murnaghan.hpp"
#include "burnman/eos/components/anharmonic_debye.hpp"
#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/bukowinski_electronic.hpp"
#include "burnman/eos/components/debye.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/eos/modified_tait.hpp"
#include "burnman/eos/vinet.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burnman::eos {
namespace {
void require_positive(const std::optional<double> &x, const char *name) {
  utils::require_set(x, name);
  if (!(std::isfinite(*x) && *x > 0.))
    throw std::invalid_argument(std::string(name) +
                                " must be positive and finite.");
}
bool uses_internal_energy_reference(const EquationOfState *reference) {
  return dynamic_cast<const BM3 *>(reference) ||
         dynamic_cast<const Vinet *>(reference) ||
         dynamic_cast<const MT *>(reference);
}
struct VolumeSolverParams {
  const ModularMGD &eos;
  const types::MineralParams &params;
  double pressure, temperature;
};
double volume_residual(double v, void *p) {
  const auto &data = *static_cast<const VolumeSolverParams *>(p);
  return data.eos.compute_pressure(data.temperature, v, data.params) -
         data.pressure;
}
} // namespace

void ModularMGD::validate_parameters(types::MineralParams &p) {
  if (!p.reference_eos)
    throw std::invalid_argument("Missing required parameter: reference_eos");
  if (dynamic_cast<ModularMGD *>(p.reference_eos.get()))
    throw std::invalid_argument(
        "A modular MGD reference_eos cannot itself be modular MGD.");
  if (!p.debye_temperature_model)
    throw std::invalid_argument(
        "Missing required parameter: debye_temperature_model");
  require_positive(p.V_0, "V_0");
  require_positive(p.napfu, "napfu");
  require_positive(p.molar_mass, "molar_mass");
  utils::require_set(p.T_0, "T_0");
  if (!(std::isfinite(*p.T_0) && *p.T_0 >= 0.))
    throw std::invalid_argument("T_0 must be finite and nonnegative.");
  utils::fallback_to_default(p.F_0, p.E_0.value_or(0.));
  utils::fallback_to_default(p.P_0, 1.e5);
  p.reference_eos->validate_parameters(p);
  p.debye_temperature_model->validate_parameters(p);
  if (p.bel_0) {
    utils::require_set(p.gel, "gel");
    if (!std::isfinite(*p.bel_0) || !std::isfinite(*p.gel))
      throw std::invalid_argument("bel_0 and gel must be finite.");
  }
  if (p.anharmonic_thermal_model) {
    AnharmonicDebye::validate_parameters(p);
  } else if (p.anharmonic_prefactor_model) {
    throw std::invalid_argument(
        "anharmonic_prefactor_model requires anharmonic_thermal_model.");
  }
}

ModularMGD::State
ModularMGD::compute_state(double t, double v,
                          const types::MineralParams &p) const {
  if (!(std::isfinite(t) && t >= 0. && std::isfinite(v) && v > 0.))
    throw std::domain_error(
        "Modular MGD requires T >= 0 and V > 0, both finite.");
  const double v0 = *p.V_0, t0 = *p.T_0, n = *p.napfu, x = v / v0;
  const auto &model = *p.debye_temperature_model;
  const double theta = model.value(x, p);
  const double theta_v = model.dVrel(x, p) / v0;
  const double theta_vv = model.dVrel2(x, p) / (v0 * v0);
  if (!(std::isfinite(theta) && theta > 0. && std::isfinite(theta_v) &&
        std::isfinite(theta_vv)))
    throw std::domain_error("Debye temperature and its derivatives must be "
                            "finite, with theta > 0.");

  // Legacy native isothermal EOSs use E_0 where Python BurnMan uses F_0.
  // Bridge that convention locally, leaving the reference implementation
  // intact.
  auto ref_params = p;
  if (uses_internal_energy_reference(p.reference_eos.get()))
    ref_params.E_0 = *p.F_0;
  double pref = p.reference_eos->compute_pressure(t0, v, ref_params);
  // Native BM3 omits P_0 in pressure; Python BM3 includes it.
  if (dynamic_cast<const BM3 *>(p.reference_eos.get()))
    pref += *p.P_0;
  const double fref =
      p.reference_eos->compute_gibbs_free_energy(pref, t0, v, ref_params) -
      pref * v;
  const double kref = p.reference_eos->compute_isothermal_bulk_modulus_reuss(
      pref, t0, v, ref_params);
  const double cv = debye::compute_molar_heat_capacity_v(t, theta, n);
  const double df = debye::compute_dhelmholtz_dTheta(t, theta, n) -
                    debye::compute_dhelmholtz_dTheta(t0, theta, n);
  const double ddf = debye::compute_d2helmholtz_dTheta2(t, theta, n) -
                     debye::compute_d2helmholtz_dTheta2(t0, theta, n);
  State state{fref + debye::compute_helmholtz_free_energy(t, theta, n) -
                  debye::compute_helmholtz_free_energy(t0, theta, n),
              pref - theta_v * df,
              kref + v * (theta_vv * df + theta_v * theta_v * ddf),
              debye::compute_entropy(t, theta, n),
              cv,
              debye::compute_dentropy_dTheta(t, theta, n) * theta_v};
  if (p.bel_0) {
    state.helmholtz += bukowinski::compute_helmholtz_el(t, v, p);
    state.pressure += bukowinski::compute_pressure_el(t, v, p);
    state.bulk_modulus += v * bukowinski::compute_KT_over_V(t, v, p);
    state.entropy += bukowinski::compute_entropy_el(t, v, p);
    state.heat_capacity_v += t * bukowinski::compute_CV_over_T(v, p);
    state.alpha_kt += bukowinski::compute_alpha_KT(t, v, p);
  }
  if (p.anharmonic_thermal_model) {
    const auto anharmonic = AnharmonicDebye::evaluate(t, v, p);
    state.helmholtz += anharmonic.helmholtz;
    state.pressure += anharmonic.pressure;
    state.bulk_modulus += anharmonic.bulk_modulus;
    state.entropy += anharmonic.entropy;
    state.heat_capacity_v += anharmonic.heat_capacity_v;
    state.alpha_kt += anharmonic.alpha_kt;
  }
  return state;
}

double ModularMGD::compute_volume(double pressure, double temperature,
                                  const types::MineralParams &p) const {
  if (!std::isfinite(pressure) || !std::isfinite(temperature) ||
      temperature < 0.)
    throw std::invalid_argument(
        "Modular MGD requires finite pressure and T >= 0.");
  const double v0 = *p.V_0;
  auto state_at = [&](double x) {
    return compute_state(temperature, x * v0, p);
  };
  auto admissible = [](const State &s) {
    return std::isfinite(s.pressure) && std::isfinite(s.bulk_modulus) &&
           s.bulk_modulus > 0.;
  };
  auto fail = []() -> double {
    throw std::domain_error(
        "Cannot find a stable modular MGD volume within the EOS domain.");
  };
  // Search outward from V_0, staying on its connected positive-K_T branch.
  // This avoids evaluating SLB Debye models at invalid fixed endpoints.
  double x = 1.;
  State current{};
  bool found = false;
  for (int i = 0; i < 200; ++i) {
    try {
      current = state_at(x);
      if (admissible(current)) {
        found = true;
        break;
      }
    } catch (const std::domain_error &) {
    }
    x /= 1.025;
  }
  if (!found)
    return fail();
  if (current.pressure == pressure)
    return x * v0;
  const bool expand = current.pressure > pressure;
  VolumeSolverParams solver{*this, p, pressure, temperature};
  for (int i = 0; i < 500; ++i) {
    double next = x * (expand ? 1.025 : 1. / 1.025);
    State trial{};
    bool valid = false;
    try {
      trial = state_at(next);
      valid = admissible(trial);
    } catch (const std::domain_error &) {
    }
    if (!valid) {
      // Resolve the last admissible point before the domain edge/spinodal.
      double stable = x, invalid = next;
      for (int j = 0; j < 60; ++j) {
        const double middle = .5 * (stable + invalid);
        bool ok = false;
        try {
          ok = admissible(state_at(middle));
        } catch (const std::domain_error &) {
        }
        if (ok)
          stable = middle;
        else
          invalid = middle;
      }
      next = stable;
      trial = state_at(next);
    }
    if ((current.pressure - pressure) * (trial.pressure - pressure) <= 0.) {
      return optim::roots::brent(&volume_residual, solver,
                                 std::min(x, next) * v0, std::max(x, next) * v0,
                                 1.e-18 * v0);
    }
    if (!valid)
      return fail();
    x = next;
    current = trial;
  }
  return fail();
}

double ModularMGD::compute_pressure(double t, double v,
                                    const types::MineralParams &p) const {
  return compute_state(t, v, p).pressure;
}

double ModularMGD::compute_isothermal_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double t, double v,
    const types::MineralParams &p) const {
  return compute_state(t, v, p).bulk_modulus;
}

double
ModularMGD::compute_molar_heat_capacity_v(double pressure [[maybe_unused]],
                                          double t, double v,
                                          const types::MineralParams &p) const {
  return compute_state(t, v, p).heat_capacity_v;
}

double
ModularMGD::compute_helmholtz_free_energy(double pressure [[maybe_unused]],
                                          double t, double v,
                                          const types::MineralParams &p) const {
  return compute_state(t, v, p).helmholtz;
}

double ModularMGD::compute_entropy(double pressure [[maybe_unused]], double t,
                                   double v,
                                   const types::MineralParams &p) const {
  return compute_state(t, v, p).entropy;
}

double
ModularMGD::compute_thermal_expansivity(double pressure [[maybe_unused]],
                                        double t, double v,
                                        const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.alpha_kt / s.bulk_modulus;
}

double
ModularMGD::compute_molar_heat_capacity_p(double pressure [[maybe_unused]],
                                          double t, double v,
                                          const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.heat_capacity_v + s.alpha_kt * s.alpha_kt * v * t / s.bulk_modulus;
}

double ModularMGD::compute_isentropic_bulk_modulus_reuss(
    double pressure [[maybe_unused]], double t, double v,
    const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return t <= constants::precision::double_eps
             ? s.bulk_modulus
             : s.bulk_modulus +
                   t * v * s.alpha_kt * s.alpha_kt / s.heat_capacity_v;
}

double
ModularMGD::compute_grueneisen_parameter(double pressure [[maybe_unused]],
                                         double t, double v,
                                         const types::MineralParams &p) const {
  const auto s = compute_state(std::max(t, 1.e-6), v, p);
  return s.alpha_kt * v / s.heat_capacity_v;
}

double
ModularMGD::compute_molar_internal_energy(double pressure [[maybe_unused]],
                                          double t, double v,
                                          const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.helmholtz + t * s.entropy;
}

double
ModularMGD::compute_gibbs_free_energy(double pressure, double t, double v,
                                      const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.helmholtz + pressure * v;
}

double ModularMGD::compute_enthalpy(double pressure, double t, double v,
                                    const types::MineralParams &p) const {
  const auto s = compute_state(t, v, p);
  return s.helmholtz + t * s.entropy + pressure * v;
}

void ModularMGDWithAnharmonicity::validate_parameters(types::MineralParams &p) {
  ModularMGD::validate_parameters(p);
  if (p.anharmonic_thermal_model)
    throw std::invalid_argument("ModularMGDWithAnharmonicity does not allow a "
                                "separate anharmonic contribution.");
  utils::require_set(p.c_anh, "c_anh");
  if (!std::isfinite(*p.c_anh))
    throw std::invalid_argument("c_anh must be finite.");
}

double ModularMGDWithAnharmonicity::ln_volume_ratio(
    double t, double v, const types::MineralParams &p) const {
  const auto harmonic = ModularMGD::compute_state(t, v, p);
  const auto reference = ModularMGD::compute_state(*p.T_0, v, p);
  return harmonic.alpha_kt * (t - *p.T_0) / reference.bulk_modulus;
}

ModularMGD::State ModularMGDWithAnharmonicity::compute_state(
    double t, double v, const types::MineralParams &p) const {
  if (*p.c_anh == 0.)
    return ModularMGD::compute_state(t, v, p);
  // Match the finite-difference steps used by Python BurnMan. At T < dT,
  // use forward differences to keep all evaluations at nonnegative T.
  const double dv = 1.e-5 * *p.V_0, dt = .01, c = *p.c_anh;
  auto l = [&](double temperature, double volume) {
    return ln_volume_ratio(temperature, volume, p);
  };
  const double value = l(t, v);
  const double plus_v = l(t, v + dv), minus_v = l(t, v - dv);
  const double lv = (plus_v - minus_v) / (2. * dv);
  const double lvv = (plus_v - 2. * value + minus_v) / (dv * dv);
  double lt, ltt, lvt;
  if (t >= dt) {
    const double plus_t = l(t + dt, v), minus_t = l(t - dt, v);
    lt = (plus_t - minus_t) / (2. * dt);
    ltt = (plus_t - 2. * value + minus_t) / (dt * dt);
    lvt = (l(t + dt, v + dv) - l(t - dt, v + dv) - l(t + dt, v - dv) +
           l(t - dt, v - dv)) /
          (4. * dv * dt);
  } else {
    const double plus_t = l(t + dt, v), plus_2t = l(t + 2. * dt, v);
    lt = (-3. * value + 4. * plus_t - plus_2t) / (2. * dt);
    ltt = (value - 2. * plus_t + plus_2t) / (dt * dt);
    const double lv_plus = (l(t + dt, v + dv) - l(t + dt, v - dv)) / (2. * dv);
    const double lv_plus2 =
        (l(t + 2. * dt, v + dv) - l(t + 2. * dt, v - dv)) / (2. * dv);
    lvt = (-3. * lv + 4. * lv_plus - lv_plus2) / (2. * dt);
  }
  const double vp = v * std::exp(-c * value);
  const double vp_v = vp * (1. / v - c * lv);
  const double vp_t = -vp * c * lt;
  const double vp_vv =
      vp * (std::pow(1. / v - c * lv, 2) - 1. / (v * v) - c * lvv);
  const double vp_tt = vp * (c * c * lt * lt - c * ltt);
  const double vp_vt = vp * (-c * lt * (1. / v - c * lv) - c * lvt);
  const auto reference = ModularMGD::compute_state(*p.T_0, v, p);
  const auto hot = ModularMGD::compute_state(t, vp, p);
  const auto cold = ModularMGD::compute_state(*p.T_0, vp, p);
  const double dp = hot.pressure - cold.pressure;
  const double dk = hot.bulk_modulus - cold.bulk_modulus;
  return {hot.helmholtz + reference.helmholtz - cold.helmholtz,
          reference.pressure + vp_v * dp,
          reference.bulk_modulus - v * vp_vv * dp + v / vp * vp_v * vp_v * dk,
          hot.entropy + dp * vp_t,
          hot.heat_capacity_v + t * (2. * hot.alpha_kt * vp_t + dp * vp_tt -
                                     dk / vp * vp_t * vp_t),
          hot.alpha_kt * vp_v - dk / vp * vp_v * vp_t + dp * vp_vt};
}

} // namespace burnman::eos
