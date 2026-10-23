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
#include "burnman/core/planet.hpp"
#include "burnman/utils/constants.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

namespace {
using namespace burnman;
constexpr double pi = 3.14159265358979323846;
class UniformMaterial : public Material {
public:
  explicit UniformMaterial(double value) : value_(value) {}
  std::shared_ptr<Material> copy() const override {
    return std::make_shared<UniformMaterial>(*this);
  }

protected:
  double compute_density() const override { return value_; }

private:
  double value_;
};
std::shared_ptr<Layer> uniform_layer(const std::string &name, double lower,
                                     double upper, double density,
                                     int count = 41) {
  auto layer = std::make_shared<Layer>(
      name, Eigen::ArrayXd::LinSpaced(count, lower, upper));
  layer->set_material(std::make_shared<UniformMaterial>(density));
  layer->set_temperature_mode("user-defined",
                              Eigen::ArrayXd::Constant(count, 300.));
  return layer;
}
} // namespace
TEST_CASE("Uniform spherical layer recovers analytical hydrostatic equilibrium",
          "[layer]") {
  using namespace burnman;
  const double radius = 1.e6, rho = 4000., surface = 1.e5;
  auto layer = uniform_layer("sphere", 0., radius, rho);
  layer->set_pressure_mode("self-consistent", {}, 0., surface);
  layer->make();
  REQUIRE(layer->converged());
  const auto gravity = layer->get_gravity(), pressure = layer->get_pressures();
  for (Eigen::Index i = 0; i < layer->n_slices; ++i) {
    const double r = layer->radii[i];
    CHECK_THAT(
        gravity[i],
        Catch::Matchers::WithinAbs(
            4. * pi * constants::physics::gravitation * rho * r / 3., 1.e-12));
    CHECK_THAT(pressure[i],
               Catch::Matchers::WithinAbs(
                   surface + 2. * pi * constants::physics::gravitation * rho *
                                 rho * (radius * radius - r * r) / 3.,
                   1.e-4));
  }
  const double mass = 4. * pi * rho * std::pow(radius, 3) / 3.;
  CHECK_THAT(layer->get_mass(), Catch::Matchers::WithinRel(mass, 1.e-13));
  CHECK_THAT(layer->get_moment_of_inertia(),
             Catch::Matchers::WithinRel(.4 * mass * radius * radius, 1.e-13));
  REQUIRE(layer->get_sublayers()[0] != layer->get_sublayers()[1]);
  layer->get_sublayers()[0]->set_state(42., 100.);
  CHECK(layer->get_sublayers()[1]->get_temperature() == 300.);
}
TEST_CASE("Layered planet preserves mass, interfaces and central gravity",
          "[planet]") {
  using namespace burnman;
  const double radius = 2.e6, core_radius = 1.e6;
  auto core = uniform_layer("core", 0., core_radius, 8000.);
  auto mantle = uniform_layer("mantle", core_radius, radius, 4000., 201);
  Planet planet("test", {mantle, core});
  planet.make();
  REQUIRE(planet.converged());
  REQUIRE(planet.get_layers()[0] == core);
  const auto pressure = planet.get_pressures(), gravity = planet.get_gravity();
  CHECK(gravity[0] == 0.);
  CHECK(core->get_gravity()[0] == 0.);
  CHECK(pressure[core->n_end - 1] == pressure[mantle->n_start]);
  CHECK(gravity[core->n_end - 1] == gravity[mantle->n_start]);
  const double expected_mass =
      4. * pi / 3. *
      (8000. * std::pow(core_radius, 3) +
       4000. * (std::pow(radius, 3) - std::pow(core_radius, 3)));
  const double expected_moment =
      8. * pi / 15. *
      (8000. * std::pow(core_radius, 5) +
       4000. * (std::pow(radius, 5) - std::pow(core_radius, 5)));
  CHECK_THAT(planet.get_mass(),
             Catch::Matchers::WithinRel(expected_mass, 1.e-13));
  CHECK_THAT(planet.get_moment_of_inertia(),
             Catch::Matchers::WithinRel(expected_moment, 1.e-13));
  const double a = 4. * pi * constants::physics::gravitation * 4000. / 3.;
  const double b = a * std::pow(core_radius, 3);
  const double interface_pressure =
      4000. * (.5 * a * (radius * radius - core_radius * core_radius) +
               b * (1. / core_radius - 1. / radius));
  const double central_pressure =
      interface_pressure + 2. * pi * constants::physics::gravitation * 8000. *
                               8000. * core_radius * core_radius / 3.;
  CHECK_THAT(pressure[0], Catch::Matchers::WithinRel(central_pressure, 1.e-6));
  CHECK_THAT(gravity[gravity.size() - 1],
             Catch::Matchers::WithinRel(constants::physics::gravitation *
                                            expected_mass / (radius * radius),
                                        1.e-13));
  CHECK(planet.get_layer_by_radius(core_radius) == core);
  const Eigen::Array3d query(1.8e6, .2e6, core_radius);
  const auto values = planet.evaluate({"density", "depth"}, query);
  CHECK(values(0, 0) == 4000.);
  CHECK(values(0, 1) == 8000.);
  CHECK(values(0, 2) == 8000.);
  CHECK(values(1, 0) == radius - query[0]);
  CHECK_THROWS_AS(planet.get_layer_by_radius(-1.), std::out_of_range);
  CHECK_THROWS_AS(
      Planet("gap", {core, uniform_layer("outer", 1.1e6, 2.e6, 4000.)}),
      std::invalid_argument);
}
TEST_CASE("Radial gravity and density moments converge for a graded sphere",
          "[layer]") {
  using namespace burnman;
  const double radius = 1.e6, rho = 4000.;
  const double surface_gravity = 4. * pi * constants::physics::gravitation *
                                 rho * radius * (1. / 3. + .25 / 5.);
  const double fourth_moment = rho * std::pow(radius, 5) * (1. / 5. + .25 / 7.);
  double previous = 1.;
  for (int count : {9, 17, 33}) {
    Layer layer("graded", Eigen::ArrayXd::LinSpaced(count, 0., radius));
    const Eigen::ArrayXd density =
        rho * (1. + .25 * (layer.radii / radius).square());
    const auto gravity = layer.compute_gravity(density, 0.);
    const double error = std::abs(gravity[count - 1] / surface_gravity - 1.);
    CHECK(error < previous);
    previous = error;
    if (count == 33) {
      CHECK(error < 1.e-5);
      CHECK_THAT(layer.density_integral(density, 4),
                 Catch::Matchers::WithinRel(fourth_moment, 1.e-5));
    }
  }
}
TEST_CASE("Boundary perturbations retain specified gradients at large Rayleigh "
          "numbers",
          "[layer]") {
  using namespace burnman;
  BoundaryLayerPerturbation perturbation(3480.e3, 5711.e3, 1.e7, 1000., .25);
  CHECK_THAT(perturbation.temperature(3480.e3),
             Catch::Matchers::WithinAbs(800., 1.e-10));
  CHECK_THAT(perturbation.temperature(5711.e3),
             Catch::Matchers::WithinAbs(-200., 1.e-10));
  perturbation.set_model_thermal_gradients(-.018, -.0006);
  CHECK_THAT(perturbation.dTdr(3480.e3),
             Catch::Matchers::WithinAbs(-.018, 1.e-14));
  CHECK_THAT(perturbation.dTdr(5711.e3),
             Catch::Matchers::WithinAbs(-.0006, 1.e-14));
  BoundaryLayerPerturbation thin(0., 1., 1.e16, 100., 0.);
  CHECK(thin.temperature(0.) == 100.);
  CHECK(std::isfinite(thin.temperature(.5)));
}
