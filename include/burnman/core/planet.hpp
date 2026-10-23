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

#include "burnman/core/layer.hpp"

namespace burnman {
/** A spherical planet assembled from adjoining material layers.
 * Layers are sorted from the centre outwards. Pressure and gravity are
 * continuous across interfaces; repeated boundary radii retain both material
 * states. Adiabatic temperatures inherit the temperature of the overlying
 * layer unless their upper-boundary temperature is explicitly prescribed.
 */
class Planet {
public:
  using Array = Eigen::ArrayXd;
  Planet(std::string name, std::vector<std::shared_ptr<Layer>> layers,
         int n_max_iterations = 50, double max_delta = 1.e-5,
         bool verbose = false);
  std::string name;
  bool verbose;
  const std::vector<std::shared_ptr<Layer>> &get_layers() const;
  const Array &get_radii() const;
  Eigen::Index get_n_slices() const;
  double get_radius_planet() const;
  double get_volume() const;
  std::string to_string() const;
  void reset();
  std::shared_ptr<Layer> get_layer(const std::string &name) const;
  /// At a shared interface, return the deeper layer.
  std::shared_ptr<Layer> get_layer_by_radius(double radius) const;
  void set_pressure_mode(const std::string &pressure_mode = "self-consistent",
                         const Array &pressures = Array{},
                         double pressure_top = 0., double gravity_bottom = 0.,
                         int n_max_iterations = 50, double max_delta = 1.e-5);
  void make();
  Eigen::MatrixXd
  evaluate(const std::vector<std::string> &properties,
           const std::optional<Array> &radlist = std::nullopt) const;
  Array get_property(const std::string &property_name) const;
  const Array &get_pressure() const { return get_pressures(); }
  const Array &get_temperature() const { return get_temperatures(); }
#define BURNMAN_PLANET_PROPERTY(name)                                          \
  Array get_##name() const { return get_property(#name); }
  BURNMAN_PLANET_PROPERTY(molar_internal_energy);
  BURNMAN_PLANET_PROPERTY(molar_gibbs);
  BURNMAN_PLANET_PROPERTY(molar_helmholtz);
  BURNMAN_PLANET_PROPERTY(molar_mass);
  BURNMAN_PLANET_PROPERTY(molar_volume);
  BURNMAN_PLANET_PROPERTY(density);
  BURNMAN_PLANET_PROPERTY(molar_entropy);
  BURNMAN_PLANET_PROPERTY(molar_enthalpy);
  BURNMAN_PLANET_PROPERTY(isothermal_bulk_modulus_reuss);
  BURNMAN_PLANET_PROPERTY(isentropic_bulk_modulus_reuss);
  BURNMAN_PLANET_PROPERTY(isothermal_compressibility_reuss);
  BURNMAN_PLANET_PROPERTY(isentropic_compressibility_reuss);
  BURNMAN_PLANET_PROPERTY(shear_modulus);
  BURNMAN_PLANET_PROPERTY(p_wave_velocity);
  BURNMAN_PLANET_PROPERTY(bulk_sound_velocity);
  BURNMAN_PLANET_PROPERTY(shear_wave_velocity);
  BURNMAN_PLANET_PROPERTY(grueneisen_parameter);
  BURNMAN_PLANET_PROPERTY(thermal_expansivity);
  BURNMAN_PLANET_PROPERTY(molar_heat_capacity_v);
  BURNMAN_PLANET_PROPERTY(molar_heat_capacity_p);
  BURNMAN_PLANET_PROPERTY(isentropic_thermal_gradient);
#undef BURNMAN_PLANET_PROPERTY

  const Array &get_pressures() const;
  const Array &get_temperatures() const;
  const Array &get_gravity() const;
  const std::string &get_pressure_mode() const;
  double get_pressure_top() const;
  double get_gravity_bottom() const;
  bool is_made() const;
  bool converged() const;
  int get_n_iterations() const;
  int get_n_max_iterations() const { return n_max_iterations_; }
  double get_max_delta() const { return max_delta_; }
  double get_mass() const;
  double get_average_density() const;
  double get_moment_of_inertia() const;
  double get_moment_of_inertia_factor() const;
  Array get_depth() const;
  Array evaluate_density(const Array &pressures,
                         const Array &temperatures) const;
  Array evaluate_temperature(const Array &pressures) const;
  Array compute_gravity(const Array &density, double gravity_bottom) const;
  Array compute_pressure(const Array &density, const Array &gravity,
                         double pressure_top) const;

private:
  std::vector<std::shared_ptr<Layer>> layers_;
  Array radii_, pressures_, temperatures_, gravity_, userpressures_;
  std::string pressure_mode_ = "self-consistent";
  double pressure_top_ = 0., gravity_bottom_ = 0., max_delta_ = 1.e-5;
  int n_max_iterations_ = 50, n_iterations_ = 0;
  bool converged_ = false;
  mutable std::map<std::string, Array> cached_;
  void require_made() const;
};
} // namespace burnman
