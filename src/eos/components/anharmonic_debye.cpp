// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/eos/components/anharmonic_debye.hpp"
#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/utils/validate_optionals.hpp"
#include <cmath>

namespace burnman::eos {
void AnharmonicDebye::validate_parameters(types::MineralParams &p) {
  utils::require_set(p.V_0, "V_0");
  utils::require_set(p.T_0, "T_0");
  if (!std::isfinite(*p.V_0) || *p.V_0 <= 0. || !std::isfinite(*p.T_0) ||
      *p.T_0 < 0.)
    throw std::invalid_argument(
        "AnharmonicDebye requires finite V_0>0 and T_0>=0.");
  if (!p.debye_temperature_model || !p.anharmonic_prefactor_model ||
      !p.anharmonic_thermal_model)
    throw std::invalid_argument(
        "AnharmonicDebye requires debye_temperature_model, "
        "anharmonic_prefactor_model and anharmonic_thermal_model.");
  p.debye_temperature_model->validate_parameters(p);
  p.anharmonic_prefactor_model->validate_parameters(p);
  p.anharmonic_thermal_model->validate_parameters(p);
}
AnharmonicDebye::State
AnharmonicDebye::evaluate(double t, double v, const types::MineralParams &p) {
  if (!std::isfinite(t) || t < 0. || !std::isfinite(v) || v <= 0.)
    throw std::domain_error("AnharmonicDebye requires finite T>=0 and V>0.");
  const double v0 = *p.V_0, x = v / v0;
  const auto &theta_model = *p.debye_temperature_model;
  const auto &prefactor = *p.anharmonic_prefactor_model;
  const double theta = theta_model.value(x, p);
  const double theta_v = theta_model.dVrel(x, p) / v0;
  const double theta_vv = theta_model.dVrel2(x, p) / (v0 * v0);
  if (!(std::isfinite(theta) && theta > 0. && std::isfinite(theta_v) &&
        std::isfinite(theta_vv)))
    throw std::domain_error("AnharmonicDebye characteristic temperature must "
                            "be positive and its volume derivatives finite.");
  const double a = prefactor.value(x, p), av = prefactor.dVrel(x, p) / v0;
  const double avv = prefactor.dVrel2(x, p) / (v0 * v0);
  const auto thermal = p.anharmonic_thermal_model->evaluate(t, theta, p);
  const auto reference = p.anharmonic_thermal_model->evaluate(*p.T_0, theta, p);
  const double f = thermal.helmholtz - reference.helmholtz;
  const double ft = thermal.dhelmholtz_dTheta - reference.dhelmholtz_dTheta;
  const double ftt =
      thermal.d2helmholtz_dTheta2 - reference.d2helmholtz_dTheta2;
  return {a * f,
          -av * f - a * theta_v * ft,
          v * (avv * f + 2. * av * theta_v * ft +
               a * (theta_vv * ft + theta_v * theta_v * ftt)),
          a * thermal.entropy,
          a * thermal.heat_capacity,
          av * thermal.entropy + a * theta_v * thermal.dentropy_dTheta};
}
double AnharmonicDebye::helmholtz_energy(double t, double v,
                                         const types::MineralParams &p) {
  return evaluate(t, v, p).helmholtz;
}
double AnharmonicDebye::pressure(double t, double v,
                                 const types::MineralParams &p) {
  return evaluate(t, v, p).pressure;
}
double AnharmonicDebye::isothermal_bulk_modulus(double t, double v,
                                                const types::MineralParams &p) {
  return evaluate(t, v, p).bulk_modulus;
}
double AnharmonicDebye::entropy(double t, double v,
                                const types::MineralParams &p) {
  return evaluate(t, v, p).entropy;
}
double AnharmonicDebye::heat_capacity_v(double t, double v,
                                        const types::MineralParams &p) {
  return evaluate(t, v, p).heat_capacity_v;
}
double AnharmonicDebye::dSdV(double t, double v,
                             const types::MineralParams &p) {
  return evaluate(t, v, p).alpha_kt;
}
} // namespace burnman::eos
