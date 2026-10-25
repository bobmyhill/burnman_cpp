// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#include "burnman/eos/dks_solid.hpp"
#include "burnman/eos/birch_murnaghan.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include "dks_common.hpp"
#include <gsl/gsl_integration.h>

namespace burnman::eos {
namespace {
struct Strain {
  double f, a1, a2, nu2;
};
Strain strain(double volume, const types::MineralParams &p) {
  const double f = .5 * (std::pow(*p.V_0 / volume, 2. / 3.) - 1.);
  const double g = *p.grueneisen_0;
  const double a1 = 6. * g, a2 = -12. * g + 36. * g * g - 18. * *p.q_0 * g;
  const double nu2 = 1. + a1 * f + .5 * a2 * f * f;
  if (!(std::isfinite(nu2) && nu2 > 0.))
    throw std::domain_error(
        "DKS solid finite-strain frequency must be positive.");
  return {f, a1, a2, nu2};
}
double gamma(double volume, const types::MineralParams &p) {
  if (p.power_law_grueneisen.value_or(true))
    return *p.grueneisen_0 * std::pow(volume / *p.V_0, *p.q_0);
  const auto s = strain(volume, p);
  return (2. * s.f + 1.) * (s.a1 + s.a2 * s.f) / (6. * s.nu2);
}
double integral_kernel(double ratio, void *data) {
  const auto &p = *static_cast<const types::MineralParams *>(data);
  return gamma(ratio * *p.V_0, p) / ratio;
}
} // namespace

void DKS_S::validate_parameters(types::MineralParams &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.K_0, "K_0");
  utils::require_set(p.Kprime_0, "Kprime_0");
  utils::require_set(p.Cv, "Cv");
  utils::require_set(p.grueneisen_0, "grueneisen_0");
  utils::require_set(p.q_0, "q_0");
  utils::require_set(p.S_0, "S_0");
  utils::fallback_to_default(p.power_law_grueneisen, true);
  utils::fallback_to_default(p.T_0, 300.);
  utils::fallback_to_default(p.E_0, std::nan(""));
  utils::fallback_to_default(p.eta_s_0, std::nan(""));
  utils::fallback_to_default(p.G_0, std::nan(""));
  utils::fallback_to_default(p.Gprime_0, std::nan(""));
  dks_detail::check_state(*p.T_0, *p.V_0);
  utils::check_in_range(p.Cv, 0., 1000., "Cv");
  utils::check_in_range(p.grueneisen_0, -.005, 10., "grueneisen_0");
  utils::check_in_range(p.q_0, -10., 10., "q_0");
  if (!std::isnan(*p.eta_s_0))
    utils::check_in_range(p.eta_s_0, -10., 10., "eta_s_0");
}
double DKS_S::integrated_grueneisen(double volume,
                                    const types::MineralParams &p) {
  if (p.power_law_grueneisen.value_or(true)) {
    // Archived aktintvto_220.f: integral gamma(V)/V dV.
    const double ln = std::log(volume / *p.V_0);
    return *p.grueneisen_0 *
           (*p.q_0 == 0. ? ln : std::expm1(*p.q_0 * ln) / *p.q_0);
  }
  if (volume == *p.V_0)
    return 0.;
  // Python integrates gamma(V)/V with scipy.integrate.quad.
  auto copy = p;
  gsl_function fn{&integral_kernel, &copy};
  double result, error;
  std::size_t neval;
  const double ratio = volume / *p.V_0;
  const int status =
      gsl_integration_qng(&fn, std::min(1., ratio), std::max(1., ratio), 1.e-12,
                          1.e-12, &result, &error, &neval);
  if (status != GSL_SUCCESS)
    throw std::runtime_error("DKS solid Gruneisen integral failed.");
  return ratio < 1. ? -result : result;
}
double DKS_S::volume_dependent_q(double volume, const types::MineralParams &p) {
  if (p.power_law_grueneisen.value_or(true))
    return *p.q_0;
  const auto s = strain(volume, p);
  const double gr = gamma(volume, p);
  if (std::abs(*p.grueneisen_0) < 1.e-10)
    return (18. * gr - 6.) / 9.;
  return (18. * gr - 6. -
          .5 / s.nu2 * std::pow(2. * s.f + 1., 2.) * s.a2 / gr) /
         9.;
}
double DKS_S::compute_volume(double pressure, double temperature,
                             const types::MineralParams &p) const {
  return dks_detail::volume(*this, pressure, temperature, p);
}
double DKS_S::compute_pressure(double temperature, double volume,
                               const types::MineralParams &p) const {
  dks_detail::check_state(temperature, volume);
  return BM3::compute_birch_murnaghan(*p.V_0 / volume, p) +
         gamma(volume, p) * *p.Cv * (temperature - *p.T_0) / volume;
}
double
DKS_S::compute_grueneisen_parameter(double, double, double volume,
                                    const types::MineralParams &p) const {
  return gamma(volume, p);
}
double DKS_S::compute_isothermal_bulk_modulus_reuss(
    double, double temperature, double volume,
    const types::MineralParams &p) const {
  const double gr = gamma(volume, p), q = volume_dependent_q(volume, p);
  const double eth = *p.Cv * (temperature - *p.T_0);
  return BM3::compute_bm_bulk_modulus(volume, p) +
         (gr + 1. - q) * gr / volume * eth - gr * gr / volume * eth;
}
double DKS_S::compute_shear_modulus(double, double temperature, double volume,
                                    const types::MineralParams &p) const {
  // The archived relation supplies no shear model.
  if (p.power_law_grueneisen.value_or(true))
    return std::nan("");
  const auto s = strain(volume, p);
  const double a2s = -2. * *p.grueneisen_0 - 2. * *p.eta_s_0;
  const double eta_s =
      -gamma(volume, p) - .5 / s.nu2 * std::pow(2. * s.f + 1., 2.) * a2s;
  return BM3::compute_third_order_shear_modulus(volume, p) -
         eta_s * *p.Cv * (temperature - *p.T_0) / volume;
}
double
DKS_S::compute_molar_heat_capacity_v(double, double, double,
                                     const types::MineralParams &p) const {
  return *p.Cv;
}
double DKS_S::compute_thermal_expansivity(double pressure, double temperature,
                                          double volume,
                                          const types::MineralParams &p) const {
  return gamma(volume, p) * *p.Cv / volume /
         compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               p);
}
double
DKS_S::compute_molar_heat_capacity_p(double pressure, double temperature,
                                     double volume,
                                     const types::MineralParams &p) const {
  return *p.Cv * (1. + gamma(volume, p) * temperature *
                           compute_thermal_expansivity(pressure, temperature,
                                                       volume, p));
}
double DKS_S::compute_isentropic_bulk_modulus_reuss(
    double pressure, double temperature, double volume,
    const types::MineralParams &p) const {
  return compute_isothermal_bulk_modulus_reuss(pressure, temperature, volume,
                                               p) *
         compute_molar_heat_capacity_p(pressure, temperature, volume, p) /
         *p.Cv;
}
double DKS_S::compute_entropy(double, double temperature, double volume,
                              const types::MineralParams &p) const {
  dks_detail::check_state(temperature, volume);
  return *p.S_0 + *p.Cv * (std::log(temperature / *p.T_0) +
                           integrated_grueneisen(volume, p));
}
double
DKS_S::compute_helmholtz_free_energy(double, double temperature, double volume,
                                     const types::MineralParams &p) const {
  dks_detail::check_state(temperature, volume);
  const double f = p.power_law_grueneisen.value_or(true)
                       ? .5 * (std::pow(*p.V_0 / volume, 2. / 3.) - 1.)
                       : strain(volume, p).f;
  const double b2 = 9. * *p.K_0, b3 = 27. * *p.K_0 * (*p.Kprime_0 - 4.);
  const double dt = temperature - *p.T_0;
  const double fcmp = .5 * b2 * f * f * *p.V_0 + *p.V_0 * b3 * f * f * f / 6.;
  const double fth =
      -*p.S_0 * dt -
      *p.Cv * (temperature * std::log(temperature / *p.T_0) - dt) -
      *p.Cv * dt * integrated_grueneisen(volume, p);
  return *p.E_0 - *p.T_0 * *p.S_0 + fcmp + fth;
}
double DKS_S::compute_gibbs_free_energy(double pressure, double temperature,
                                        double volume,
                                        const types::MineralParams &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p) +
         pressure * volume;
}
double
DKS_S::compute_molar_internal_energy(double pressure, double temperature,
                                     double volume,
                                     const types::MineralParams &p) const {
  return compute_helmholtz_free_energy(pressure, temperature, volume, p) +
         temperature * compute_entropy(pressure, temperature, volume, p);
}
double DKS_S::compute_enthalpy(double pressure, double temperature,
                               double volume,
                               const types::MineralParams &p) const {
  return compute_molar_internal_energy(pressure, temperature, volume, p) +
         pressure * volume;
}
} // namespace burnman::eos
