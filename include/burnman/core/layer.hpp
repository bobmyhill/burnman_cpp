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

#include "burnman/core/material.hpp"
#include <Eigen/Core>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace burnman {

/** A spherical material layer on strictly increasing radii in metres.
 * Set a Material class object, a temperature mode and pressure boundary
 * conditions, then call make() before querying thermodynamic properties.
 * Each radius retains an independent material state. Pressures use Pa,
 * temperatures K and gravity m/s².
 * Radial integrals use not-a-knot cubic interpolation with local linear
 * fallback for overshoot on monotone stencils. This fallback suppresses
 * ringing but does not locate interfaces.
 */
class Layer {
public:
  using Array = Eigen::ArrayXd;
  Layer(std::string name, const Array &radii, bool verbose = false);
  virtual ~Layer() = default;

  std::string name;
  const Array radii;
  const double inner_radius, outer_radius, thickness;
  const Eigen::Index n_slices;
  bool verbose;
  Eigen::Index n_start = 0, n_end = 0;

  void reset();
  void set_material(std::shared_ptr<Material> material);
  std::shared_ptr<Material> get_material() const;
  void
  set_temperature_mode(const std::string &temperature_mode = "adiabatic",
                       const Array &temperatures = Array{},
                       std::optional<double> temperature_top = std::nullopt);
  void set_pressure_mode(const std::string &pressure_mode = "self-consistent",
                         const Array &pressures = Array{},
                         std::optional<double> gravity_bottom = std::nullopt,
                         std::optional<double> pressure_top = std::nullopt,
                         int n_max_iterations = 50, double max_delta = 1.e-5);
  void make();
  std::string to_string() const;

  /// One row per requested property, preserving the order of queried radii.
  /// An absent radlist uses the internal grid and retains its material states.
  Eigen::MatrixXd
  evaluate(const std::vector<std::string> &properties,
           const std::optional<Array> &radlist = std::nullopt,
           std::optional<double> radius_planet = std::nullopt) const;
  Array get_property(const std::string &property_name) const;
  const Array &get_pressure() const { return get_pressures(); }
  const Array &get_temperature() const { return get_temperatures(); }
#define BURNMAN_LAYER_PROPERTY(name)                                           \
  Array get_##name() const { return get_property(#name); }
  BURNMAN_LAYER_PROPERTY(molar_internal_energy);
  BURNMAN_LAYER_PROPERTY(molar_gibbs);
  BURNMAN_LAYER_PROPERTY(molar_helmholtz);
  BURNMAN_LAYER_PROPERTY(molar_mass);
  BURNMAN_LAYER_PROPERTY(molar_volume);
  BURNMAN_LAYER_PROPERTY(density);
  BURNMAN_LAYER_PROPERTY(molar_entropy);
  BURNMAN_LAYER_PROPERTY(molar_enthalpy);
  BURNMAN_LAYER_PROPERTY(isothermal_bulk_modulus_reuss);
  BURNMAN_LAYER_PROPERTY(isentropic_bulk_modulus_reuss);
  BURNMAN_LAYER_PROPERTY(isothermal_compressibility_reuss);
  BURNMAN_LAYER_PROPERTY(isentropic_compressibility_reuss);
  BURNMAN_LAYER_PROPERTY(shear_modulus);
  BURNMAN_LAYER_PROPERTY(p_wave_velocity);
  BURNMAN_LAYER_PROPERTY(bulk_sound_velocity);
  BURNMAN_LAYER_PROPERTY(shear_wave_velocity);
  BURNMAN_LAYER_PROPERTY(grueneisen_parameter);
  BURNMAN_LAYER_PROPERTY(thermal_expansivity);
  BURNMAN_LAYER_PROPERTY(molar_heat_capacity_v);
  BURNMAN_LAYER_PROPERTY(molar_heat_capacity_p);
  BURNMAN_LAYER_PROPERTY(isentropic_thermal_gradient);
#undef BURNMAN_LAYER_PROPERTY

  const Array &get_pressures() const;
  const Array &get_temperatures() const;
  const std::vector<std::shared_ptr<Material>> &get_sublayers() const;
  const std::string &get_pressure_mode() const;
  const std::string &get_temperature_mode() const;
  std::optional<double> get_temperature_top() const;
  std::optional<double> get_pressure_top() const;
  std::optional<double> get_gravity_bottom() const;
  const Array &get_usertemperatures() const;
  bool is_made() const;
  bool converged() const;
  int get_n_iterations() const;
  int get_n_max_iterations() const { return n_max_iterations_; }
  double get_max_delta() const { return max_delta_; }

  double get_mass() const;
  double get_moment_of_inertia() const;
  Array get_gravity() const;
  Array get_bullen() const;
  /// Squared buoyancy frequency, g²*(bullen-1)/v_phi², in s⁻².
  Array get_brunt_vasala() const;

  Array evaluate_temperature(
      const Array &pressures,
      std::optional<double> temperature_top = std::nullopt) const;
  Array evaluate_density(const Array &pressures,
                         const Array &temperatures) const;
  /// Integrate d(g*r²)/dr = 4*pi*G*rho*r² from the lower boundary.
  virtual Array compute_gravity(const Array &density,
                                double gravity_bottom) const;
  /// Integrate dP/dr = -rho*g from the upper boundary.
  virtual Array compute_pressure(const Array &density, const Array &gravity,
                                 double pressure_top) const;
  /// Integral of rho(r)*r^power, used for mass and moment of inertia.
  virtual double density_integral(const Array &density, int power) const;

private:
  friend class Planet;
  std::shared_ptr<Material> material_;
  std::string pressure_mode_ = "self-consistent", temperature_mode_;
  Array pressures_, temperatures_, userpressures_, usertemperatures_;
  std::optional<double> pressure_top_, temperature_top_, gravity_bottom_;
  int n_max_iterations_ = 50, n_iterations_ = 0;
  double max_delta_ = 1.e-5;
  bool converged_ = false;
  std::vector<std::shared_ptr<Material>> sublayers_;
  mutable std::map<std::string, Array> cached_;

  void require_material() const;
  void require_made() const;
  void set_profile(const Array &pressures, const Array &temperatures,
                   double gravity_bottom);
};

/** Exponential temperature perturbations at the two boundaries of a layer.
 * The scale c = Ra^(1/4), total temperature change and boundary-layer ratio
 * determine the bottom and top perturbations. Gradients use K/m.
 */
class BoundaryLayerPerturbation {
public:
  BoundaryLayerPerturbation(double radius_bottom, double radius_top,
                            double rayleigh_number, double temperature_change,
                            double boundary_layer_ratio);
  double r0, r1, Ra, c;
  double temperature(double radius) const;
  Eigen::ArrayXd temperature(const Eigen::ArrayXd &radii) const;
  double dTdr(double radius) const;
  Eigen::ArrayXd dTdr(const Eigen::ArrayXd &radii) const;
  void set_model_thermal_gradients(double dTdr_bottom, double dTdr_top);
  double get_a() const;
  double get_b() const;

private:
  double bottom_amplitude_, top_amplitude_;
};
} // namespace burnman
