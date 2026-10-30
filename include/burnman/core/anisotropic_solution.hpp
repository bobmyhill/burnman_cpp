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

#ifndef BURNMAN_CORE_ANISOTROPIC_SOLUTION_HPP_INCLUDED
#define BURNMAN_CORE_ANISOTROPIC_SOLUTION_HPP_INCLUDED

#include "burnman/core/anisotropic_mineral.hpp"
#include "burnman/core/solution.hpp"

namespace burnman {
using CompositionTensors = std::vector<Eigen::Matrix3d>;
/// Psi excess, its f/Pth derivatives, and d(Psi:I)/dp at fixed V,T.
using PsiExcessDerivatives = std::tuple<types::Matrix6d, types::Matrix6d,
                                        types::Matrix6d, CompositionTensors>;
using PsiExcessFunction = std::function<PsiExcessDerivatives(
    double log_volume, double thermal_pressure,
    const Eigen::ArrayXd &fractions)>;

/// Myhill (2024) anisotropic mixing at a common molar volume and temperature.
class AnisotropicSolution : public Solution, public AnisotropicMineral {
public:
  /// Model polymorphic_endmembers must contain only AnisotropicMineral objects.
  AnisotropicSolution(
      const std::string &name,
      std::shared_ptr<solution_models::SolutionModel> solution_model,
      PsiExcessFunction psi_excess_function,
      std::optional<Eigen::ArrayXd> molar_fractions = std::nullopt);
  double T_0 = 298.15;

  std::shared_ptr<Material> copy() const override;
  void reset_cache() override;
  void set_composition(const Eigen::ArrayXd &fractions) override;
  void set_state(double pressure, double temperature) override;
  void set_state_with_volume(double volume, double temperature,
                             const Eigen::Vector2d &pressure_guesses =
                                 Eigen::Vector2d(0., 1.e10)) override;
  void set_method(types::EOSType method) override;
  void set_method(std::shared_ptr<EquationOfState> method) override;
  void compute_base_properties();

  const std::vector<std::shared_ptr<AnisotropicMineral>> &
  get_endmembers() const;
  const std::shared_ptr<Solution> &get_scalar_solution() const;
  const CompositionTensors &get_logm_M0_mbr() const;
  const CompositionTensors &get_PsiI_mbr() const;
  const CompositionTensors &get_dPsiIdp_excess() const;
  CompositionTensors get_dM0dp_fixed_VT() const;
  CompositionTensors get_unrotated_dFdp_fixed_VT() const;
  CompositionTensors get_unrotated_dMdn_fixed_VT() const;
  CompositionTensors get_dMdn_fixed_VT() const;
  CompositionTensors get_depsdn_fixed_VT() const;

protected:
  PsiState compute_psi_state() const override;
  std::pair<double, double> compute_deformation_alpha_beta() const override;
  double compute_molar_internal_energy() const override;
  double compute_molar_gibbs() const override;
  double compute_molar_helmholtz() const override;
  double compute_molar_mass() const override;
  double compute_molar_volume() const override;
  double compute_molar_volume_unmodified() const override;
  double compute_density() const override;
  double compute_molar_entropy() const override;
  double compute_molar_enthalpy() const override;
  double compute_isothermal_bulk_modulus_reuss() const override;
  double compute_isentropic_bulk_modulus_reuss() const override;
  double compute_isothermal_compressibility_reuss() const override;
  double compute_isentropic_compressibility_reuss() const override;
  double compute_shear_modulus() const override;
  double compute_p_wave_velocity() const override;
  double compute_bulk_sound_velocity() const override;
  double compute_shear_wave_velocity() const override;
  double compute_grueneisen_parameter() const override;
  double compute_thermal_expansivity() const override;
  double compute_molar_heat_capacity_v() const override;
  double compute_molar_heat_capacity_p() const override;
  double compute_isentropic_thermal_gradient() const override;
  types::FormulaMap compute_formula() const override;

private:
  void require_composition() const;
  std::vector<std::shared_ptr<AnisotropicMineral>> endmembers_;
  std::shared_ptr<Solution> scalar_solution_;
  PsiExcessFunction psi_excess_function_;
  CompositionTensors logm_M0_mbr_;
  mutable CompositionTensors PsiI_mbr_, dPsiIdp_excess_;
};
} // namespace burnman

#endif
