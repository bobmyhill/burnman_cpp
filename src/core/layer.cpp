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

#include "burnman/core/layer.hpp"
#include "burnman/core/geotherm.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/warnings.hpp"
#include "radial_profile.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <utility>

namespace burnman {
namespace {
constexpr double pi = 3.14159265358979323846;
bool is_pressure(const std::string &name) {
  return name == "pressure" || name == "pressures" || name == "P";
}
bool is_temperature(const std::string &name) {
  return name == "temperature" || name == "temperatures" || name == "T";
}
bool is_radial(const std::string &name) {
  return is_pressure(name) || is_temperature(name) || name == "radii" ||
         name == "radius" || name == "gravity" || name == "bullen" ||
         name == "brunt_vasala";
}
} // namespace

Layer::Layer(std::string layer_name, const Array &radius_grid,
             bool verbose_output)
    : name(std::move(layer_name)), radii(detail::checked_radii(radius_grid)),
      inner_radius(radii[0]), outer_radius(radii[radii.size() - 1]),
      thickness(outer_radius - inner_radius), n_slices(radii.size()),
      verbose(verbose_output), usertemperatures_(Array::Zero(n_slices)) {}

void Layer::reset() {
  pressures_.resize(0);
  temperatures_.resize(0);
  sublayers_.clear();
  cached_.clear();
  converged_ = false;
  n_iterations_ = 0;
}
void Layer::set_material(std::shared_ptr<Material> material) {
  if (!material)
    throw std::invalid_argument("Supply a Material class object.");
  material_ = std::move(material);
  reset();
}
std::shared_ptr<Material> Layer::get_material() const { return material_; }
void Layer::require_material() const {
  if (!material_)
    throw std::runtime_error("Call set_material() before making the layer.");
}
void Layer::require_made() const {
  if (!is_made())
    throw std::runtime_error("Call make() before evaluating the layer.");
}
bool Layer::is_made() const {
  return sublayers_.size() == static_cast<std::size_t>(n_slices);
}
bool Layer::converged() const { return converged_; }
int Layer::get_n_iterations() const { return n_iterations_; }
const std::string &Layer::get_pressure_mode() const { return pressure_mode_; }
const std::string &Layer::get_temperature_mode() const {
  return temperature_mode_;
}
std::optional<double> Layer::get_temperature_top() const {
  return temperature_top_;
}
std::optional<double> Layer::get_pressure_top() const { return pressure_top_; }
std::optional<double> Layer::get_gravity_bottom() const {
  return gravity_bottom_;
}
const Layer::Array &Layer::get_usertemperatures() const {
  return usertemperatures_;
}
const Layer::Array &Layer::get_pressures() const {
  require_made();
  return pressures_;
}
const Layer::Array &Layer::get_temperatures() const {
  require_made();
  return temperatures_;
}
const std::vector<std::shared_ptr<Material>> &Layer::get_sublayers() const {
  require_made();
  return sublayers_;
}
void Layer::set_temperature_mode(const std::string &mode,
                                 const Array &temperatures,
                                 std::optional<double> top) {
  if (mode != "adiabatic" && mode != "perturbed-adiabatic" &&
      mode != "user-defined")
    throw std::invalid_argument("Unknown layer temperature mode: " + mode);
  if (mode != "adiabatic")
    detail::check_profile(temperatures, n_slices, "Temperatures");
  if (mode == "user-defined" && (temperatures < 0.).any())
    throw std::invalid_argument("Temperatures must be nonnegative.");
  if (top && (!std::isfinite(*top) || *top <= 0.))
    throw std::invalid_argument(
        "The upper-boundary temperature must be positive and finite.");
  reset();
  temperature_mode_ = mode;
  usertemperatures_ =
      mode == "adiabatic" ? Array::Zero(n_slices) : temperatures;
  temperature_top_ = mode == "user-defined" ? std::nullopt : top;
}
void Layer::set_pressure_mode(const std::string &mode, const Array &pressures,
                              std::optional<double> bottom_gravity,
                              std::optional<double> top_pressure,
                              int iterations, double delta) {
  if (mode != "user-defined" && mode != "self-consistent")
    throw std::invalid_argument("Unknown layer pressure mode: " + mode);
  if (!bottom_gravity || !std::isfinite(*bottom_gravity) ||
      *bottom_gravity < 0.)
    throw std::invalid_argument("Supply a finite, nonnegative gravity_bottom.");
  if (mode == "user-defined")
    detail::check_profile(pressures, n_slices, "Pressures");
  if (mode == "self-consistent" &&
      (!top_pressure || !std::isfinite(*top_pressure)))
    throw std::invalid_argument("Supply a finite pressure_top.");
  detail::check_iterations(iterations, delta);
  reset();
  pressure_mode_ = mode;
  userpressures_ = pressures;
  gravity_bottom_ = bottom_gravity;
  pressure_top_ = top_pressure;
  n_max_iterations_ = iterations;
  max_delta_ = delta;
}
Layer::Array Layer::evaluate_temperature(const Array &pressures,
                                         std::optional<double> top) const {
  require_material();
  detail::check_profile(pressures, n_slices, "Pressures");
  if (temperature_mode_.empty())
    throw std::runtime_error(
        "Call set_temperature_mode() before making the layer.");
  if (temperature_mode_ == "user-defined")
    return usertemperatures_;
  if (!top)
    top = temperature_top_;
  if (!top)
    throw std::runtime_error(
        "Supply temperature_top for the outermost adiabat.");
  const double anchor = *top - usertemperatures_[n_slices - 1];
  Array reversed_pressures = pressures.reverse();
  Array temperatures =
      geotherm::adiabatic_profile(reversed_pressures, *material_, anchor)
          .reverse();
  temperatures += usertemperatures_;
  if (!temperatures.isFinite().all() || (temperatures < 0.).any())
    throw std::runtime_error(
        "The temperature profile contains invalid temperatures.");
  return temperatures;
}
Layer::Array Layer::evaluate_density(const Array &pressures,
                                     const Array &temperatures) const {
  require_material();
  detail::check_profile(pressures, n_slices, "Pressures");
  detail::check_profile(temperatures, n_slices, "Temperatures");
  auto working = material_->copy();
  Array density(n_slices);
  for (Eigen::Index i = 0; i < n_slices; ++i) {
    working->set_state(pressures[i], temperatures[i]);
    density[i] = working->get_density();
  }
  if (!density.isFinite().all() || (density <= 0.).any())
    throw std::runtime_error(
        "The layer material must have finite, positive density.");
  return density;
}
Layer::Array Layer::compute_gravity(const Array &density, double bottom) const {
  const detail::RadialSpline rho(radii, density);
  Array gravity(n_slices);
  double enclosed = bottom * inner_radius * inner_radius;
  gravity[0] = inner_radius == 0. ? 0. : bottom;
  for (Eigen::Index i = 1; i < n_slices; ++i) {
    enclosed +=
        4. * pi * constants::physics::gravitation *
        detail::integrate(
            [&](double radius) { return rho(radius) * radius * radius; },
            radii[i - 1], radii[i]);
    gravity[i] = enclosed / (radii[i] * radii[i]);
  }
  return gravity;
}
Layer::Array Layer::compute_pressure(const Array &density, const Array &gravity,
                                     double top) const {
  const detail::RadialSpline rho(radii, density), g(radii, gravity);
  Array pressures(n_slices);
  pressures[n_slices - 1] = top;
  for (Eigen::Index i = n_slices - 1; i > 0; --i)
    pressures[i - 1] = pressures[i] + detail::integrate(
                                          [&](double radius) {
                                            return rho(radius) * g(radius);
                                          },
                                          radii[i - 1], radii[i]);
  return pressures;
}
double Layer::density_integral(const Array &density, int power) const {
  const detail::RadialSpline rho(radii, density);
  double result = 0.;
  for (Eigen::Index i = 1; i < n_slices; ++i)
    result += detail::integrate(
        [&](double radius) { return rho(radius) * std::pow(radius, power); },
        radii[i - 1], radii[i]);
  return result;
}
void Layer::set_profile(const Array &pressures, const Array &temperatures,
                        double bottom) {
  require_material();
  detail::check_profile(pressures, n_slices, "Pressures");
  detail::check_profile(temperatures, n_slices, "Temperatures");
  if ((temperatures < 0.).any())
    throw std::invalid_argument("Temperatures must be nonnegative.");
  std::vector<std::shared_ptr<Material>> states;
  states.reserve(static_cast<std::size_t>(n_slices));
  auto working = material_->copy();
  for (Eigen::Index i = 0; i < n_slices; ++i) {
    working->set_state(pressures[i], temperatures[i]);
    states.push_back(working->copy());
  }
  pressures_ = pressures;
  temperatures_ = temperatures;
  gravity_bottom_ = bottom;
  sublayers_ = std::move(states);
  cached_.clear();
}
void Layer::make() {
  reset();
  require_material();
  if (!gravity_bottom_)
    throw std::runtime_error(
        "Call set_pressure_mode() before making the layer.");
  Array pressures, temperatures;
  if (pressure_mode_ == "user-defined") {
    pressures = userpressures_;
    temperatures = evaluate_temperature(pressures);
    converged_ = true;
  } else if (pressure_mode_ == "self-consistent") {
    if (!pressure_top_)
      throw std::runtime_error("Supply pressure_top before making the layer.");
    pressures = *pressure_top_ + (outer_radius - radii) * 1.e3;
    temperatures = evaluate_temperature(pressures);
    for (int i = 0; i < n_max_iterations_; ++i) {
      const double previous = pressures.maxCoeff();
      const auto density = evaluate_density(pressures, temperatures);
      const auto gravity = compute_gravity(density, *gravity_bottom_);
      pressures = compute_pressure(density, gravity, *pressure_top_);
      temperatures = evaluate_temperature(pressures);
      const double change = std::abs(previous - pressures.maxCoeff()) /
                            std::max(1., std::abs(pressures.maxCoeff()));
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
      utils::warn("Layer pressure iteration did not converge: " + name);
  } else
    throw std::runtime_error(
        "Layer pressures are set by a Planet; call Planet.make().");
  set_profile(pressures, temperatures, *gravity_bottom_);
}
Layer::Array Layer::get_gravity() const {
  require_made();
  return compute_gravity(get_property("density"), *gravity_bottom_);
}
Layer::Array Layer::get_bullen() const {
  const Array density = get_property("density");
  const Array velocity = get_property("bulk_sound_velocity");
  const Array phi = velocity.square(), kappa = phi * density;
  return detail::index_gradient(kappa) / detail::index_gradient(pressures_) +
         detail::index_gradient(phi) / detail::index_gradient(radii) /
             get_property("gravity");
}
Layer::Array Layer::get_brunt_vasala() const {
  const Array velocity = get_property("bulk_sound_velocity");
  return get_property("gravity").square() * (get_property("bullen") - 1.) /
         velocity.square();
}
double Layer::get_mass() const {
  return 4. * pi * density_integral(get_property("density"), 2);
}
double Layer::get_moment_of_inertia() const {
  return (8. * pi / 3.) * density_integral(get_property("density"), 4);
}
Layer::Array Layer::get_property(const std::string &property_name) const {
  if (property_name == "radii" || property_name == "radius")
    return radii;
  require_made();
  if (is_pressure(property_name))
    return pressures_;
  if (is_temperature(property_name))
    return temperatures_;
  const auto found = cached_.find(property_name);
  if (found != cached_.end())
    return found->second;
  Array values(n_slices);
  if (property_name == "gravity")
    values = get_gravity();
  else if (property_name == "bullen")
    values = get_bullen();
  else if (property_name == "brunt_vasala")
    values = get_brunt_vasala();
  else
    for (Eigen::Index i = 0; i < n_slices; ++i)
      values[i] =
          sublayers_[static_cast<std::size_t>(i)]->get_property(property_name);
  cached_[property_name] = values;
  return values;
}
Eigen::MatrixXd Layer::evaluate(const std::vector<std::string> &properties,
                                const std::optional<Array> &radlist,
                                std::optional<double> radius_planet) const {
  const auto &query = radlist ? *radlist : radii;
  Eigen::MatrixXd result(static_cast<Eigen::Index>(properties.size()),
                         query.size());
  if (radlist)
    for (double radius : query)
      detail::interpolate(radius, radii, radii);
  std::shared_ptr<Material> working;
  for (std::size_t row = 0; row < properties.size(); ++row) {
    const auto &property_name = properties[row];
    if (property_name == "depth") {
      if (!radius_planet || !std::isfinite(*radius_planet) ||
          *radius_planet < outer_radius)
        throw std::invalid_argument("Supply radius_planet >= the layer's outer "
                                    "radius to evaluate depth.");
      result.row(static_cast<Eigen::Index>(row)) =
          (*radius_planet - query).matrix().transpose();
    } else if (!radlist) {
      result.row(static_cast<Eigen::Index>(row)) =
          get_property(property_name).matrix().transpose();
    } else if (is_radial(property_name)) {
      const auto values = get_property(property_name);
      for (Eigen::Index j = 0; j < query.size(); ++j)
        result(static_cast<Eigen::Index>(row), j) =
            detail::interpolate(query[j], radii, values);
    } else {
      require_made();
      if (!working)
        working = material_->copy();
      for (Eigen::Index j = 0; j < query.size(); ++j) {
        working->set_state(detail::interpolate(query[j], radii, pressures_),
                           detail::interpolate(query[j], radii, temperatures_));
        result(static_cast<Eigen::Index>(row), j) =
            working->get_property(property_name);
      }
    }
  }
  return result;
}
std::string Layer::to_string() const {
  return "The " + name + " is made of " +
         (material_ ? material_->get_name() : "an unset material") + " with " +
         temperature_mode_ + " temperatures and " + pressure_mode_ +
         " pressures\n";
}

BoundaryLayerPerturbation::BoundaryLayerPerturbation(double bottom, double top,
                                                     double rayleigh,
                                                     double change,
                                                     double ratio)
    : r0(bottom), r1(top), Ra(rayleigh), c(std::pow(rayleigh, .25)) {
  if (!std::isfinite(bottom) || !std::isfinite(top) || bottom < 0. ||
      top <= bottom || !std::isfinite(rayleigh) || rayleigh < 0. ||
      !std::isfinite(change) || !std::isfinite(ratio) || ratio < 0.)
    throw std::invalid_argument("Supply finite increasing layer radii, Ra >= 0 "
                                "and boundary_layer_ratio >= 0.");
  bottom_amplitude_ = change / (1. + ratio);
  top_amplitude_ = -ratio * bottom_amplitude_;
}
double BoundaryLayerPerturbation::temperature(double radius) const {
  if (!std::isfinite(radius))
    throw std::invalid_argument("Radii must be finite.");
  return bottom_amplitude_ * std::exp(-c * (radius - r0) / (r1 - r0)) +
         top_amplitude_ * std::exp(-c * (r1 - radius) / (r1 - r0));
}
Eigen::ArrayXd
BoundaryLayerPerturbation::temperature(const Eigen::ArrayXd &query) const {
  return query.unaryExpr([&](double radius) { return temperature(radius); });
}
double BoundaryLayerPerturbation::dTdr(double radius) const {
  if (!std::isfinite(radius))
    throw std::invalid_argument("Radii must be finite.");
  return c / (r1 - r0) *
         (-bottom_amplitude_ * std::exp(-c * (radius - r0) / (r1 - r0)) +
          top_amplitude_ * std::exp(-c * (r1 - radius) / (r1 - r0)));
}
Eigen::ArrayXd
BoundaryLayerPerturbation::dTdr(const Eigen::ArrayXd &query) const {
  return query.unaryExpr([&](double radius) { return dTdr(radius); });
}
void BoundaryLayerPerturbation::set_model_thermal_gradients(double bottom,
                                                            double top) {
  if (!std::isfinite(bottom) || !std::isfinite(top) || c == 0.)
    throw std::invalid_argument(
        "Thermal gradients must be finite and Ra must be positive.");
  const double e = std::exp(-c), divisor = -std::expm1(-2. * c);
  const double u = -bottom * (r1 - r0) / c, v = -top * (r1 - r0) / c;
  bottom_amplitude_ = (u - e * v) / divisor;
  top_amplitude_ = (e * u - v) / divisor;
}
double BoundaryLayerPerturbation::get_a() const {
  return bottom_amplitude_ * std::exp(-c);
}
double BoundaryLayerPerturbation::get_b() const {
  return top_amplitude_ * std::exp(-c);
}
} // namespace burnman
