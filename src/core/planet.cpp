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

#include "burnman/core/planet.hpp"
#include "burnman/utils/warnings.hpp"
#include "radial_profile.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>

namespace burnman {
namespace {
constexpr double pi = 3.14159265358979323846;
}
Planet::Planet(std::string planet_name,
               std::vector<std::shared_ptr<Layer>> layers, int iterations,
               double delta, bool verbose_output)
    : name(std::move(planet_name)), verbose(verbose_output),
      layers_(std::move(layers)) {
  if (layers_.empty())
    throw std::invalid_argument("A planet requires at least one Layer.");
  for (const auto &layer : layers_)
    if (!layer)
      throw std::invalid_argument("Planet layers cannot be null.");
  std::sort(layers_.begin(), layers_.end(), [](const auto &a, const auto &b) {
    return a->inner_radius < b->inner_radius;
  });
  Eigen::Index count = 0;
  for (std::size_t i = 0; i < layers_.size(); ++i) {
    if (i && layers_[i]->inner_radius != layers_[i - 1]->outer_radius)
      throw std::invalid_argument(
          "Planet layers must adjoin without gaps or overlaps.");
    layers_[i]->n_start = count;
    count += layers_[i]->n_slices;
    layers_[i]->n_end = count;
  }
  radii_.resize(count);
  for (const auto &layer : layers_)
    radii_.segment(layer->n_start, layer->n_slices) = layer->radii;
  set_pressure_mode("self-consistent", {}, 0., 0., iterations, delta);
}
const std::vector<std::shared_ptr<Layer>> &Planet::get_layers() const {
  return layers_;
}
const Planet::Array &Planet::get_radii() const { return radii_; }
Eigen::Index Planet::get_n_slices() const { return radii_.size(); }
double Planet::get_radius_planet() const { return radii_[radii_.size() - 1]; }
double Planet::get_volume() const {
  return 4. * pi / 3. * std::pow(get_radius_planet(), 3);
}
const std::string &Planet::get_pressure_mode() const { return pressure_mode_; }
double Planet::get_pressure_top() const { return pressure_top_; }
double Planet::get_gravity_bottom() const { return gravity_bottom_; }
bool Planet::converged() const { return converged_; }
int Planet::get_n_iterations() const { return n_iterations_; }
bool Planet::is_made() const { return pressures_.size() == radii_.size(); }
void Planet::require_made() const {
  if (!is_made())
    throw std::runtime_error("Call make() before evaluating the planet.");
  for (const auto &layer : layers_)
    layer->require_made();
}
void Planet::reset() { cached_.clear(); }
std::shared_ptr<Layer> Planet::get_layer(const std::string &layer_name) const {
  for (const auto &layer : layers_)
    if (layer->name == layer_name)
      return layer;
  throw std::out_of_range("No layer named " + layer_name + ".");
}
std::shared_ptr<Layer> Planet::get_layer_by_radius(double radius) const {
  if (!std::isfinite(radius) || radius < radii_[0] ||
      radius > get_radius_planet())
    throw std::out_of_range("Radius is outside the planet.");
  for (const auto &layer : layers_)
    if (radius <= layer->outer_radius)
      return layer;
  throw std::out_of_range("Radius is outside the planet.");
}
void Planet::set_pressure_mode(const std::string &mode, const Array &pressures,
                               double top, double bottom, int iterations,
                               double delta) {
  if (mode != "user-defined" && mode != "self-consistent")
    throw std::invalid_argument("Unknown planet pressure mode: " + mode);
  if (!std::isfinite(top) || !std::isfinite(bottom) || bottom < 0.)
    throw std::invalid_argument(
        "Supply finite pressure_top and nonnegative gravity_bottom.");
  if (radii_[0] == 0. && bottom != 0.)
    throw std::invalid_argument("Gravity must be zero at the centre.");
  if (mode == "user-defined")
    detail::check_profile(pressures, radii_.size(), "Pressures");
  detail::check_iterations(iterations, delta);
  // Repeated interface radii describe the same physical pressure.
  if (mode == "user-defined")
    for (std::size_t i = 1; i < layers_.size(); ++i) {
      const auto offset = layers_[i]->n_start;
      if (pressures[offset] != pressures[offset - 1])
        throw std::invalid_argument(
            "User-defined pressures must be continuous at layer interfaces.");
    }
  pressure_mode_ = mode;
  userpressures_ = pressures;
  pressure_top_ = top;
  gravity_bottom_ = bottom;
  n_max_iterations_ = iterations;
  max_delta_ = delta;
  pressures_.resize(0);
  temperatures_.resize(0);
  gravity_.resize(0);
  converged_ = false;
  n_iterations_ = 0;
  reset();
}
Planet::Array Planet::evaluate_density(const Array &pressures,
                                       const Array &temperatures) const {
  detail::check_profile(pressures, radii_.size(), "Pressures");
  detail::check_profile(temperatures, radii_.size(), "Temperatures");
  Array density(radii_.size());
  for (const auto &layer : layers_)
    density.segment(layer->n_start, layer->n_slices) = layer->evaluate_density(
        pressures.segment(layer->n_start, layer->n_slices),
        temperatures.segment(layer->n_start, layer->n_slices));
  return density;
}
Planet::Array Planet::evaluate_temperature(const Array &pressures) const {
  detail::check_profile(pressures, radii_.size(), "Pressures");
  Array temperatures(radii_.size());
  std::optional<double> top;
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    const auto &layer = *it;
    if (layer->temperature_top_)
      top = layer->temperature_top_;
    const Array values = layer->evaluate_temperature(
        pressures.segment(layer->n_start, layer->n_slices), top);
    temperatures.segment(layer->n_start, layer->n_slices) = values;
    top = values[0];
  }
  return temperatures;
}
Planet::Array Planet::compute_gravity(const Array &density,
                                      double bottom) const {
  detail::check_profile(density, radii_.size(), "Density");
  Array gravity(radii_.size());
  for (const auto &layer : layers_) {
    const Array values = layer->compute_gravity(
        density.segment(layer->n_start, layer->n_slices), bottom);
    detail::check_profile(values, layer->n_slices, "Gravity");
    gravity.segment(layer->n_start, layer->n_slices) = values;
    bottom = values[values.size() - 1];
  }
  return gravity;
}
Planet::Array Planet::compute_pressure(const Array &density,
                                       const Array &gravity, double top) const {
  detail::check_profile(density, radii_.size(), "Density");
  detail::check_profile(gravity, radii_.size(), "Gravity");
  Array pressures(radii_.size());
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    const auto &layer = *it;
    const Array values = layer->compute_pressure(
        density.segment(layer->n_start, layer->n_slices),
        gravity.segment(layer->n_start, layer->n_slices), top);
    detail::check_profile(values, layer->n_slices, "Pressures");
    pressures.segment(layer->n_start, layer->n_slices) = values;
    top = values[0];
  }
  return pressures;
}
void Planet::make() {
  reset();
  converged_ = false;
  n_iterations_ = 0;
  Array pressures, temperatures, gravity;
  if (pressure_mode_ == "user-defined") {
    pressures = userpressures_;
    temperatures = evaluate_temperature(pressures);
    converged_ = true;
  } else {
    pressures = pressure_top_ + (get_radius_planet() - radii_) * 1.e3;
    temperatures = evaluate_temperature(pressures);
    for (int i = 0; i < n_max_iterations_; ++i) {
      const Array previous = pressures;
      const auto density = evaluate_density(pressures, temperatures);
      gravity = compute_gravity(density, gravity_bottom_);
      pressures = compute_pressure(density, gravity, pressure_top_);
      temperatures = evaluate_temperature(pressures);
      const double change = (pressures - previous).abs().maxCoeff() /
                            std::max(1., pressures.abs().maxCoeff());
      n_iterations_ = i + 1;
      if (verbose)
        std::cout << "Iteration " << n_iterations_
                  << " maximum relative pressure error: " << change << '\n';
      if (change < max_delta_) {
        converged_ = true;
        break;
      }
    }
    if (!converged_)
      utils::warn("Planet pressure iteration did not converge: " + name);
  }
  // Report gravity from the final state, consistently with the mass integral.
  gravity = compute_gravity(evaluate_density(pressures, temperatures),
                            gravity_bottom_);
  // Store both sides of each interface, with the actual lower-boundary gravity.
  for (const auto &layer : layers_) {
    layer->reset();
    layer->set_profile(pressures.segment(layer->n_start, layer->n_slices),
                       temperatures.segment(layer->n_start, layer->n_slices),
                       gravity[layer->n_start]);
    layer->pressure_mode_ = "set-in-planet";
    layer->converged_ = converged_;
    layer->n_iterations_ = n_iterations_;
  }
  pressures_ = std::move(pressures);
  temperatures_ = std::move(temperatures);
  gravity_ = std::move(gravity);
}
const Planet::Array &Planet::get_pressures() const {
  require_made();
  return pressures_;
}
const Planet::Array &Planet::get_temperatures() const {
  require_made();
  return temperatures_;
}
const Planet::Array &Planet::get_gravity() const {
  require_made();
  return gravity_;
}
Planet::Array Planet::get_depth() const { return get_radius_planet() - radii_; }
double Planet::get_mass() const {
  require_made();
  double mass = 0.;
  for (const auto &layer : layers_)
    mass += layer->get_mass();
  return mass;
}
double Planet::get_average_density() const { return get_mass() / get_volume(); }
double Planet::get_moment_of_inertia() const {
  require_made();
  double moment = 0.;
  for (const auto &layer : layers_)
    moment += layer->get_moment_of_inertia();
  return moment;
}
double Planet::get_moment_of_inertia_factor() const {
  return get_moment_of_inertia() /
         (get_mass() * get_radius_planet() * get_radius_planet());
}
Planet::Array Planet::get_property(const std::string &property_name) const {
  if (property_name == "radii" || property_name == "radius")
    return radii_;
  if (property_name == "depth")
    return get_depth();
  require_made();
  if (property_name == "pressure" || property_name == "pressures" ||
      property_name == "P")
    return pressures_;
  if (property_name == "temperature" || property_name == "temperatures" ||
      property_name == "T")
    return temperatures_;
  if (property_name == "gravity")
    return gravity_;
  const auto found = cached_.find(property_name);
  if (found != cached_.end())
    return found->second;
  Array values(radii_.size());
  for (const auto &layer : layers_)
    values.segment(layer->n_start, layer->n_slices) =
        layer->get_property(property_name);
  cached_[property_name] = values;
  return values;
}
Eigen::MatrixXd Planet::evaluate(const std::vector<std::string> &properties,
                                 const std::optional<Array> &radlist) const {
  if (!radlist) {
    Eigen::MatrixXd values(static_cast<Eigen::Index>(properties.size()),
                           radii_.size());
    for (std::size_t i = 0; i < properties.size(); ++i)
      values.row(static_cast<Eigen::Index>(i)) =
          get_property(properties[i]).matrix().transpose();
    return values;
  }
  Eigen::MatrixXd values(static_cast<Eigen::Index>(properties.size()),
                         radlist->size());
  for (Eigen::Index i = 0; i < radlist->size(); ++i) {
    const auto layer = get_layer_by_radius((*radlist)[i]);
    const Array radius = Array::Constant(1, (*radlist)[i]);
    values.col(i) =
        layer->evaluate(properties, radius, get_radius_planet()).col(0);
  }
  return values;
}
std::string Planet::to_string() const {
  std::string result =
      name + " consists of " + std::to_string(layers_.size()) + " layers:\n";
  for (const auto &layer : layers_)
    result += layer->to_string();
  return result;
}
} // namespace burnman
