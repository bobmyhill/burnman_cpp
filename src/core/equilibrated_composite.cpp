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

#include "burnman/core/equilibrated_composite.hpp"
#include "burnman/core/relaxed_solution.hpp"
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace burnman {
namespace pd = phase_diagrams;

struct EquilibratedComposite::Impl {
  pd::detail::Engine engine;
  bool relaxed, warm_start;
  types::AveragingType scheme;
  std::optional<pd::detail::WorkState> accepted;
  std::optional<pd::State> attempted;

  Impl(const types::FormulaMap &composition,
       const std::vector<std::shared_ptr<Material>> &candidates,
       const pd::Settings &settings, bool relax, types::AveragingType averaging,
       bool warm)
      : engine(composition, candidates, settings), relaxed(relax),
        warm_start(warm), scheme(averaging) {}
};

EquilibratedComposite::EquilibratedComposite(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const pd::Settings &settings, bool relaxed, types::AveragingType scheme,
    bool warm_start) {
  for (const auto &phase : candidates)
    if (std::dynamic_pointer_cast<RelaxedSolution>(phase))
      throw std::invalid_argument(
          "EquilibratedComposite requires ordinary Solution candidates; "
          "equilibrium determines their internal populations.");
  impl_ = std::make_unique<Impl>(composition, candidates, settings, relaxed,
                                 scheme, warm_start);
  set_name("EquilibratedComposite");
  Assemblage::set_averaging_scheme(scheme);
}
EquilibratedComposite::~EquilibratedComposite() = default;

void EquilibratedComposite::require_equilibrium() const {
  if (!impl_->accepted || !has_state())
    throw std::runtime_error("Call set_state(pressure, temperature) before "
                             "querying the equilibrated composite.");
}

void EquilibratedComposite::set_state(double new_pressure,
                                      double new_temperature) {
  if (!std::isfinite(new_pressure) || !std::isfinite(new_temperature) ||
      new_pressure < 0.0 || new_temperature < 0.0 ||
      (new_temperature == 0.0 && !impl_->engine.settings.active_solution_faces))
    throw std::invalid_argument("Equilibrium requires finite P,T >= 0; T=0 "
                                "requires active_solution_faces.");
  pd::detail::WorkState next;
  try {
    // The previous equilibrium supplies estimates, never prescribed phases.
    // Continuation checks competing-phase affinities, curvature and distinct
    // solution copies, falling back to phase discovery when necessary.
    next = impl_->engine.stable_at(
        {new_pressure, new_temperature},
        impl_->warm_start && impl_->accepted ? &*impl_->accepted : nullptr);
  } catch (const std::exception &error) {
    pd::State failure;
    failure.pressure = new_pressure;
    failure.temperature = new_temperature;
    failure.message = error.what();
    impl_->attempted = failure;
    throw;
  }
  impl_->attempted = next.state;
  if (!next.state.success || next.state.outside_model_domain ||
      !next.assemblage)
    throw std::runtime_error("EquilibratedComposite equilibrium failed: " +
                             next.state.message);
  const auto material_name = get_name();
  // Copy the entire active basis, including distinct solvus copies. Clear
  // scalar caches because the accepted ordinary Assemblage may have cached
  // frozen responses, whereas this material may request relaxed responses.
  static_cast<Assemblage &>(*this) = *next.assemblage;
  Assemblage::reset_cache();
  Assemblage::set_averaging_scheme(impl_->scheme);
  set_name(material_name);
  impl_->accepted = std::move(next);
}

void EquilibratedComposite::set_method(types::EOSType) {
  throw std::logic_error("EquilibratedComposite uses the equations of state in "
                         "its thermodynamic dataset.");
}
void EquilibratedComposite::set_method(std::shared_ptr<EquationOfState>) {
  throw std::logic_error("EquilibratedComposite uses the equations of state in "
                         "its thermodynamic dataset.");
}
void EquilibratedComposite::set_averaging_scheme(types::AveragingType scheme) {
  Assemblage::set_averaging_scheme(scheme);
  impl_->scheme = scheme;
  Assemblage::reset_cache();
}
void EquilibratedComposite::set_relaxed(bool relaxed) {
  impl_->relaxed = relaxed;
  Assemblage::reset_cache();
}
bool EquilibratedComposite::get_relaxed() const { return impl_->relaxed; }
void EquilibratedComposite::set_warm_start(bool warm_start) {
  impl_->warm_start = warm_start;
}
bool EquilibratedComposite::get_warm_start() const { return impl_->warm_start; }
types::AveragingType EquilibratedComposite::get_averaging_scheme() const {
  return impl_->scheme;
}
pd::Settings EquilibratedComposite::get_settings() const {
  return impl_->engine.settings;
}
std::optional<pd::State> EquilibratedComposite::get_equilibrium_state() const {
  return impl_->accepted ? std::optional<pd::State>(impl_->accepted->state)
                         : std::nullopt;
}
std::optional<pd::State>
EquilibratedComposite::get_last_equilibrium_attempt() const {
  return impl_->attempted;
}
std::vector<std::shared_ptr<Material>>
EquilibratedComposite::get_candidate_phases() const {
  std::vector<std::shared_ptr<Material>> result;
  for (const auto &phase : impl_->engine.phases)
    result.push_back(pd::detail::clone(phase.material));
  return result;
}

std::shared_ptr<Assemblage> EquilibratedComposite::snapshot() const {
  require_equilibrium();
  auto result = impl_->engine.copy_assemblage(*this);
  result->set_name(get_name());
  result->set_averaging_scheme(impl_->scheme);
  result->reset_cache();
  return result;
}
void EquilibratedComposite::clear_state() {
  const auto material_name = get_name();
  static_cast<Assemblage &>(*this) = Assemblage{};
  Assemblage::set_averaging_scheme(impl_->scheme);
  set_name(material_name);
  impl_->accepted.reset();
}

double EquilibratedComposite::property(const std::string &property_name) const {
  require_equilibrium();
  // A plain Assemblage avoids recursive virtual dispatch and double relaxation
  // through this material's cached properties. Keep its actual active phases.
  Assemblage frozen = static_cast<const Assemblage &>(*this);
  frozen.reset_cache();
  return pd::assemblage_properties(frozen, {property_name}, impl_->relaxed)
      .at(property_name);
}
#define EQUILIBRATED_PROPERTY(name)                                            \
  double EquilibratedComposite::compute_##name() const {                       \
    return property(#name);                                                    \
  }
EQUILIBRATED_PROPERTY(molar_internal_energy)
EQUILIBRATED_PROPERTY(molar_gibbs)
EQUILIBRATED_PROPERTY(molar_helmholtz)
EQUILIBRATED_PROPERTY(molar_volume)
EQUILIBRATED_PROPERTY(molar_mass)
EQUILIBRATED_PROPERTY(density)
EQUILIBRATED_PROPERTY(molar_entropy)
EQUILIBRATED_PROPERTY(molar_enthalpy)
EQUILIBRATED_PROPERTY(isothermal_bulk_modulus_reuss)
EQUILIBRATED_PROPERTY(isentropic_bulk_modulus_reuss)
EQUILIBRATED_PROPERTY(isothermal_compressibility_reuss)
EQUILIBRATED_PROPERTY(isentropic_compressibility_reuss)
EQUILIBRATED_PROPERTY(shear_modulus)
EQUILIBRATED_PROPERTY(p_wave_velocity)
EQUILIBRATED_PROPERTY(bulk_sound_velocity)
EQUILIBRATED_PROPERTY(shear_wave_velocity)
EQUILIBRATED_PROPERTY(grueneisen_parameter)
EQUILIBRATED_PROPERTY(thermal_expansivity)
EQUILIBRATED_PROPERTY(molar_heat_capacity_v)
EQUILIBRATED_PROPERTY(molar_heat_capacity_p)
EQUILIBRATED_PROPERTY(isentropic_thermal_gradient)
#undef EQUILIBRATED_PROPERTY
types::FormulaMap EquilibratedComposite::compute_formula() const {
  require_equilibrium();
  return Assemblage::compute_formula();
}
} // namespace burnman
