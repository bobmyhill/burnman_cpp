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
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
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

std::shared_ptr<Material> EquilibratedComposite::copy() const {
  auto result = std::make_shared<EquilibratedComposite>(
      impl_->engine.bulk, get_candidate_phases(), get_settings(), get_relaxed(),
      get_averaging_scheme(), get_warm_start());
  result->impl_->attempted = impl_->attempted;
  result->impl_->engine.potential_seed = impl_->engine.potential_seed;
  if (impl_->accepted) {
    auto accepted = *impl_->accepted;
    accepted.assemblage = impl_->engine.copy_assemblage(*accepted.assemblage);
    static_cast<Assemblage &>(*result) = *accepted.assemblage;
    result->Assemblage::set_averaging_scheme(impl_->scheme);
    result->Assemblage::reset_cache();
    result->impl_->accepted = std::move(accepted);
  }
  result->set_name(get_name());
  return result;
}

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
  accept_state(std::move(next));
}

void EquilibratedComposite::accept_state(pd::detail::WorkState next) {
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

void EquilibratedComposite::set_state_with_hugoniot(double new_pressure,
                                                    double reference_pressure,
                                                    double reference_energy,
                                                    double reference_volume,
                                                    double temperature_guess) {
  using namespace equilibration;
  if (!std::isfinite(new_pressure) || new_pressure < 0. ||
      !std::isfinite(temperature_guess) || temperature_guess <= 0.)
    throw std::invalid_argument("Supply finite P >= 0 and a positive finite "
                                "Hugoniot temperature estimate.");
  const HugoniotConstraint energy(reference_pressure, reference_energy,
                                  reference_volume);
  auto &engine = impl_->engine;
  const auto &settings = engine.settings;
  pd::detail::WorkState next;
  next.state.pressure = new_pressure;
  next.state.temperature = temperature_guess;
  try {
    std::vector<pd::detail::WorkState> trials;
    auto enqueue = [&](pd::detail::WorkState trial) {
      for (const auto &old : trials) {
        if (old.ids.size() != trial.ids.size())
          continue;
        bool same = true;
        for (const auto &phase : trial.state.phases) {
          const auto match = pd::detail::find_phase(old.state.phases, phase.id);
          if (match == old.state.phases.end() ||
              (match->composition - phase.composition).norm() >
                  settings.composition_tolerance) {
            same = false;
            break;
          }
        }
        if (same)
          return;
      }
      trial.assemblage.reset();
      trials.push_back(std::move(trial));
    };
    if (impl_->warm_start && impl_->accepted) {
      next = *impl_->accepted;
      next.state.temperature = temperature_guess;
    } else {
      next = engine.stable(new_pressure, temperature_guess);
      if (!next.state.success)
        throw std::runtime_error(next.state.message);
    }
    enqueue(next);
    bool tried_cold = false;
    for (std::size_t iteration = 0; iteration < trials.size() && iteration < 60;
         ++iteration) {
      next = trials[iteration];
      next.assemblage =
          engine.make_assemblage(next.ids, next.state.phases, new_pressure,
                                 next.state.temperature, 1.e-7, true);
      auto constraints = pd::detail::constraints(
          std::make_unique<PressureConstraint>(new_pressure), energy.clone());
      pd::detail::EquilibriumSolve solved;
      try {
        solved = engine.solve(*next.assemblage, constraints, false, .01);
      } catch (const std::exception &error) {
        solved.success = false;
        solved.message = error.what();
        // A failed Newton estimate may be outside the allowed phase bounds.
        // Restore the feasible estimate before selecting another assemblage.
        next.assemblage =
            engine.make_assemblage(next.ids, next.state.phases, new_pressure,
                                   next.state.temperature, 1.e-7, true);
      }
      engine.record_coordinates(next.state, *next.assemblage);
      next.state.phases = engine.snapshot(*next.assemblage, next.ids);
      next.state.success = false;
      next.state.message = solved.message;
      if (settings.verbose) {
        std::cerr << "Hugoniot trial at " << next.state.pressure << " Pa, "
                  << next.state.temperature << " K: " << solved.message << '\n';
        for (const auto &phase : next.state.phases)
          std::cerr << "  " << phase.name << ": " << phase.amount << " mol\n";
      }
      const auto present = pd::detail::present_phase_ids(
          next.state.phases, settings.amount_tolerance);
      // At an edge of a coexistence interval the vanishing phase no longer
      // imposes a chemical-equilibrium equation. Continue with the survivors.
      if (!present.empty() && present.size() < next.ids.size()) {
        auto surviving = next;
        surviving.ids = present;
        pd::detail::erase_if(surviving.state.phases, [&](const auto &phase) {
          return std::find(present.begin(), present.end(), phase.id) ==
                 present.end();
        });
        enqueue(std::move(surviving));
        continue;
      }
      if (!solved.success) {
        if (!tried_cold) {
          tried_cold = true;
          auto cold = engine.stable(new_pressure, temperature_guess);
          if (cold.state.success)
            enqueue(std::move(cold));
        }
        continue;
      }
      next.state.equilibrium_error =
          pd::detail::reaction_residual(*next.assemblage);
      next.state.mass_balance_error = engine.mass_error(*next.assemblage);
      next.state.minimum_affinity = engine.stability(*next.assemblage);
      std::optional<pd::detail::WorkState> discovered;
      if (next.state.minimum_affinity < -settings.affinity_tolerance) {
        // An assemblage fixes only chemical-potential combinations in the
        // span of its formulae. Gibbs minimisation supplies an admissible
        // supporting plane for the stability test; its remaining coefficients
        // are auxiliary, not uniquely defined component chemical potentials.
        discovered = engine.stable(new_pressure, next.state.temperature);
        if (!discovered->state.success)
          throw std::runtime_error(discovered->state.message);
        next.state.minimum_affinity = engine.stability(*next.assemblage);
      }
      const auto x = get_parameter_vector(*next.assemblage);
      const double energy_error =
          std::abs(energy.evaluate(x, *next.assemblage));
      if (next.state.minimum_affinity >= -settings.affinity_tolerance &&
          engine.locally_stable(*next.assemblage) &&
          engine.distinct(*next.assemblage, next.ids) &&
          energy_error <=
              1.e-6 * std::max(1., next.assemblage->get_n_moles())) {
        next.state.gibbs =
            next.assemblage->get_n_moles() * next.assemblage->get_molar_gibbs();
        next.state.success = true;
        next.state.message = "Stable equilibrium with Hugoniot energy balance";
        accept_state(std::move(next));
        return;
      }
      if (next.state.minimum_affinity >= -settings.affinity_tolerance &&
          energy_error > 1.e-6 * std::max(1., next.assemblage->get_n_moles()))
        throw std::runtime_error("Hugoniot energy residual exceeds tolerance.");

      // A stable assemblage at this P,T supplies phases that can lower Gibbs
      // energy. Mix the two bulk-balanced estimates, then allow both their
      // amounts and temperature to change under the energy constraint.
      const auto stable =
          discovered ? *discovered
                     : engine.stable(new_pressure, next.state.temperature);
      if (!stable.state.success)
        throw std::runtime_error(stable.state.message);
      enqueue(stable);
      auto phase_states = next.state.phases;
      for (auto &phase : phase_states)
        phase.amount *= .5;
      for (auto phase : stable.state.phases) {
        phase.amount *= .5;
        auto match = std::find_if(
            phase_states.begin(), phase_states.end(), [&](auto &old) {
              return old.candidate_index == phase.candidate_index &&
                     (old.composition - phase.composition).norm() <
                         settings.composition_tolerance;
            });
        if (match != phase_states.end()) {
          match->amount += phase.amount;
          continue;
        }
        int copy = 0;
        while (copy < settings.max_phase_instances &&
               pd::detail::find_phase(phase_states,
                                      phase.candidate_index *
                                              settings.max_phase_instances +
                                          copy) != phase_states.end())
          ++copy;
        if (copy == settings.max_phase_instances)
          throw std::runtime_error("Hugoniot exceeds the allowed number of "
                                   "coexisting compositions of one phase.");
        phase.id = phase.candidate_index * settings.max_phase_instances + copy;
        phase_states.push_back(std::move(phase));
      }
      const auto ids = pd::detail::phase_ids(phase_states);
      next.ids = ids;
      next.state.phases = std::move(phase_states);
      enqueue(next);
    }
    throw std::runtime_error("Hugoniot equilibrium did not converge: " +
                             next.state.message);
  } catch (const std::exception &error) {
    next.state.success = false;
    next.state.message = error.what();
    impl_->attempted = next.state;
    throw;
  }
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
