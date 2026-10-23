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

#pragma once

#include "burnman/core/assemblage.hpp"
#include "burnman/tools/phase_diagram/types.hpp"
#include <memory>
#include <optional>

namespace burnman {

/** A composite whose phases and compositions are discovered at every state.
 * Minimises Gibbs energy at fixed bulk composition, pressure and temperature,
 * allowing phases to enter or leave and distinct compositions of a solution
 * to coexist. Candidate Material class objects are copied. A failed solve
 * preserves the last accepted material state and records the attempted
 * equilibrium separately.
 *
 * warm_start=true reuses the last equilibrium's phases, amounts and
 * compositions as initial estimates, checking competing phases and local
 * stability before acceptance. An unsuccessful estimate triggers a new phase
 * search. false starts each search independently.
 *
 * relaxed=true includes equilibrium changes in phase amounts and compositions
 * in thermal derivatives; false holds the calculated fractions/compositions
 * fixed when taking these derivatives. Elastic averaging
 * follows Assemblage. n_moles retains the supplied bulk's elemental mole scale.
 */
class EquilibratedComposite : public Assemblage {
public:
  EquilibratedComposite(
      const types::FormulaMap &composition,
      const std::vector<std::shared_ptr<Material>> &candidates,
      const phase_diagrams::Settings &settings = {}, bool relaxed = true,
      types::AveragingType averaging_scheme = types::AveragingType::VRH,
      bool warm_start = true);
  ~EquilibratedComposite() override;

  void set_state(double new_pressure, double new_temperature) override;
  void set_method(types::EOSType method) override;
  void set_method(std::shared_ptr<EquationOfState> method) override;
  void set_averaging_scheme(types::AveragingType scheme);
  void set_relaxed(bool relaxed);
  bool get_relaxed() const;
  void set_warm_start(bool warm_start);
  bool get_warm_start() const;
  types::AveragingType get_averaging_scheme() const;
  phase_diagrams::Settings get_settings() const;
  std::optional<phase_diagrams::State> get_equilibrium_state() const;
  std::optional<phase_diagrams::State> get_last_equilibrium_attempt() const;
  std::vector<std::shared_ptr<Material>> get_candidate_phases() const;

  /// A frozen, independent phase snapshot for ordinary Composite workflows.
  std::shared_ptr<Assemblage> snapshot() const;
  /// Restore the uninitialised state after a Python evaluate() call.
  void clear_state();

protected:
#define BURNMAN_EQUILIBRATED_PROPERTY(name)                                    \
  double compute_##name() const override
  BURNMAN_EQUILIBRATED_PROPERTY(molar_internal_energy);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_gibbs);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_helmholtz);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_volume);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_mass);
  BURNMAN_EQUILIBRATED_PROPERTY(density);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_entropy);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_enthalpy);
  BURNMAN_EQUILIBRATED_PROPERTY(isothermal_bulk_modulus_reuss);
  BURNMAN_EQUILIBRATED_PROPERTY(isentropic_bulk_modulus_reuss);
  BURNMAN_EQUILIBRATED_PROPERTY(isothermal_compressibility_reuss);
  BURNMAN_EQUILIBRATED_PROPERTY(isentropic_compressibility_reuss);
  BURNMAN_EQUILIBRATED_PROPERTY(shear_modulus);
  BURNMAN_EQUILIBRATED_PROPERTY(p_wave_velocity);
  BURNMAN_EQUILIBRATED_PROPERTY(bulk_sound_velocity);
  BURNMAN_EQUILIBRATED_PROPERTY(shear_wave_velocity);
  BURNMAN_EQUILIBRATED_PROPERTY(grueneisen_parameter);
  BURNMAN_EQUILIBRATED_PROPERTY(thermal_expansivity);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_heat_capacity_v);
  BURNMAN_EQUILIBRATED_PROPERTY(molar_heat_capacity_p);
  BURNMAN_EQUILIBRATED_PROPERTY(isentropic_thermal_gradient);
#undef BURNMAN_EQUILIBRATED_PROPERTY
  types::FormulaMap compute_formula() const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void require_equilibrium() const;
  double property(const std::string &property_name) const;
};
} // namespace burnman
