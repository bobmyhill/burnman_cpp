// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of tests/test_radial_structure.py; reference:
// tests/reference/burnman.json.

#include "../../src/core/radial_profile.hpp"
#include "burnman/core/layer.hpp"
#include "burnman/core/planet.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/utils/constants.hpp"
#include "original_test_helpers.hpp"
#include <array>
#include <limits>
#include <utility>

namespace {
using namespace burnman;
using Array = Eigen::ArrayXd;
constexpr double pi = 3.14159265358979323846;
constexpr double G = constants::physics::gravitation;

class UniformMaterial : public Material {
public:
  explicit UniformMaterial(double rho) : density_(rho) {}
  std::shared_ptr<Material> copy() const override {
    return std::make_shared<UniformMaterial>(*this);
  }

protected:
  double compute_density() const override { return density_; }

private:
  double density_;
};

class PressureStepMaterial : public Material {
public:
  std::shared_ptr<Material> copy() const override {
    return std::make_shared<PressureStepMaterial>(*this);
  }

protected:
  double compute_density() const override {
    return get_pressure() >= 1.e9 ? 8000. : 4000.;
  }
};

std::shared_ptr<Layer> uniform_layer(const std::string &name, double inner,
                                     double outer, double density, int n = 65) {
  auto layer = std::make_shared<Layer>(name, Array::LinSpaced(n, inner, outer));
  layer->set_material(std::make_shared<UniformMaterial>(density));
  layer->set_temperature_mode("user-defined", Array::Constant(n, 300.));
  return layer;
}

void check_array(const Array &actual, const Array &expected,
                 double rtol = 1.e-5, double atol = 1.e-16) {
  REQUIRE(actual.size() == expected.size());
  for (Eigen::Index i = 0; i < actual.size(); ++i)
    original::check_close(actual[i], expected[i], rtol, atol);
}
} // namespace

TEST_CASE("Original radial spline bounds density jumps", "[original][radial]") {
  Array uneven(9);
  uneven << 0., .05, .2, .3, .45, .47, .6, .85, 1.;
  for (const Array &x : {Array(Array::LinSpaced(17, 0., 1.)), uneven})
    for (double interface : {0., .45, .99})
      for (double sign : {-1., 1.})
        for (const auto &scale : {std::pair{1., 1.e-6}, std::pair{1.e6, 1.}}) {
          const double radius_scale = scale.first, density_scale = scale.second;
          const Array r = x * radius_scale;
          const Array rho = x.unaryExpr([&](double v) {
            return density_scale *
                   (6000. + sign * (v <= interface ? 2000. : -2000.));
          });
          const detail::RadialSpline spline(r, rho);
          for (Eigen::Index i = 0; i < r.size(); ++i)
            original::check_close(spline(r[i]), rho[i], 1.e-14);
          const double tolerance =
              128. * std::numeric_limits<double>::epsilon() * rho.maxCoeff();
          for (Eigen::Index i = 0; i + 1 < r.size(); ++i)
            for (int j = 0; j <= 256; ++j) {
              const double value = spline(r[i] + (r[i + 1] - r[i]) * j / 256.);
              CHECK(value >= std::min(rho[i], rho[i + 1]) - tolerance);
              CHECK(value <= std::max(rho[i], rho[i + 1]) + tolerance);
            }
        }
}

TEST_CASE("Original radial spline preserves smooth profiles and extrema",
          "[original][radial]") {
  Array uneven(6);
  uneven << 0., .1, .25, .4, .8, 1.;
  for (const Array &x : {Array(Array::LinSpaced(9, 0., 1.)), uneven})
    for (int profile = 0; profile < 5; ++profile) {
      const auto value = [profile](double r) {
        switch (profile) {
        case 0:
          return 4000.;
        case 1:
          return 8000. - 4000. * r * r;
        case 2:
          return 8000. - 4000. * r * r * r;
        case 3:
          return 8000. - 4000. * (r - .53) * (r - .53);
        default:
          return 4000. + 4000. * (r - .53) * (r - .53);
        }
      };
      const detail::RadialSpline spline(x * 1.e6, x.unaryExpr(value));
      for (int j = 0; j <= 1000; ++j) {
        const double r = j / 1000.;
        original::check_close(spline(r * 1.e6), value(r), 1.e-14);
      }
    }
}

TEST_CASE("Original density jump uses consistent radial integrals",
          "[original][radial]") {
  Layer layer("density jump", Array::LinSpaced(65, 1.e6, 2.e6));
  layer.set_material(std::make_shared<PressureStepMaterial>());
  layer.set_temperature_mode("user-defined", Array::Constant(65, 300.));
  layer.set_pressure_mode("user-defined", Array::LinSpaced(65, 2.e9, 0.), 5.);
  layer.make();
  const detail::RadialSpline rho(layer.radii, layer.get_density());
  const Array gravity = layer.get_gravity();
  const detail::RadialSpline g(layer.radii, gravity);
  const auto integral = [&](const auto &function) {
    double sum = 0.;
    for (Eigen::Index i = 0; i + 1 < layer.radii.size(); ++i)
      sum += original::quad(function, layer.radii[i], layer.radii[i + 1], 0.,
                            1.e-11);
    return sum;
  };
  original::check_close(
      layer.get_mass(),
      4. * pi * integral([&](double r) { return rho(r) * r * r; }), 1.e-13);
  original::check_close(layer.get_moment_of_inertia(),
                        8. * pi / 3. * integral([&](double r) {
                          return rho(r) * std::pow(r, 4.);
                        }),
                        1.e-13);
  original::check_close(
      gravity.tail(1)[0] * layer.outer_radius * layer.outer_radius,
      5. * layer.inner_radius * layer.inner_radius + G * layer.get_mass(),
      1.e-14);
  const Array pressure =
      layer.compute_pressure(layer.get_density(), gravity, 1.e5);
  original::check_close(
      pressure[0], 1.e5 + integral([&](double r) { return rho(r) * g(r); }),
      1.e-13);
  CHECK((pressure.tail(64) < pressure.head(64)).all());
}

TEST_CASE("Original uniform sphere", "[original][radial]") {
  const double R = 1.e6, rho = 4000., surface = 1.e5;
  Array three(3);
  three << 0., .2 * R, R;
  for (const Array &r : {Array(Array::LinSpaced(2, 0., R)), three,
                         Array(Array::LinSpaced(9, 0., R))}) {
    Layer layer("sphere", r);
    layer.set_material(std::make_shared<UniformMaterial>(rho));
    layer.set_temperature_mode("user-defined", Array::Constant(r.size(), 300.));
    layer.set_pressure_mode("self-consistent", {}, 0., surface);
    layer.make();
    check_array(layer.get_gravity(), 4. * pi * G * rho * r / 3., 1.e-7, 1.e-13);
    check_array(layer.get_pressures(),
                surface + 2. * pi * G * rho * rho * (R * R - r.square()) / 3.,
                1.e-13);
    const double mass = 4. * pi * rho * R * R * R / 3.;
    original::check_close(layer.get_mass(), mass, 1.e-14);
    original::check_close(layer.get_moment_of_inertia(), .4 * mass * R * R,
                          1.e-14);
    CHECK(layer.converged());
  }
}

TEST_CASE("Original hollow shell keeps inner gravity", "[original][radial]") {
  const double inner = 1.e6, outer = 2.e6, rho = 4000., g0 = 5., p0 = 1.e5;
  auto layer = uniform_layer("shell", inner, outer, rho, 257);
  Planet planet("shell", {layer}, 50, 1.e-12);
  planet.set_pressure_mode("self-consistent", {}, p0, g0, 50, 1.e-12);
  planet.make();
  const double a = 4. * pi * G * rho / 3., b = (g0 - a * inner) * inner * inner;
  const Array &r = layer->radii;
  check_array(layer->get_gravity(), a * r + b / r.square(), 1.e-14);
  check_array(layer->get_pressures(),
              p0 + rho * (a * (outer * outer - r.square()) / 2. +
                          b * (1. / r - 1. / outer)),
              1.e-8);
  CHECK(*layer->get_gravity_bottom() == g0);
  CHECK(planet.get_gravity()[0] == g0);
}

TEST_CASE("Original layer interfaces and unordered queries",
          "[original][radial]") {
  auto core = uniform_layer("core", 0., 1.e6, 8000.);
  auto mantle = uniform_layer("mantle", 1.e6, 2.e6, 4000., 257);
  Planet planet("test", {mantle, core}, 50, 1.e-12);
  planet.make();
  CHECK(*core->get_gravity_bottom() == 0.);
  CHECK(*mantle->get_gravity_bottom() == core->get_gravity().tail(1)[0]);
  CHECK(core->get_pressures().tail(1)[0] == mantle->get_pressures()[0]);
  Array queries(4), expected(4);
  queries << 1.8e6, .2e6, 1.e6, 1.8e6;
  expected << 4000., 8000., 8000., 4000.;
  check_array(planet.evaluate({"density"}, queries).row(0).transpose(),
              expected, 1.e-7);
  CHECK(planet.evaluate({"density"}, Array{}).size() == 0);
  for (double radius :
       {-1., 2.e6 + 1., std::numeric_limits<double>::quiet_NaN()})
    CHECK_THROWS_AS(planet.get_layer_by_radius(radius), std::out_of_range);
  const double mass = 4. * pi / 3. * (8000. * 1.e18 + 4000. * (8.e18 - 1.e18));
  original::check_close(planet.get_mass(), mass, 1.e-14);
  original::check_close(planet.get_gravity().tail(1)[0],
                        G * mass / (2.e6 * 2.e6), 1.e-14);
}

TEST_CASE("Original prescribed planet pressures are used on each make",
          "[original][radial]") {
  auto layer = uniform_layer("sphere", 0., 1.e6, 4000.);
  Planet planet("test", {layer});
  for (double maximum : {1.e9, 2.e9}) {
    const Array pressure = Array::LinSpaced(planet.get_n_slices(), maximum, 0.);
    planet.set_pressure_mode("user-defined", pressure);
    planet.make();
    check_array(planet.get_pressures(), pressure, 1.e-7);
    check_array(layer->get_pressures(), pressure, 1.e-7);
    check_array(planet.get_gravity(),
                4. * pi * G * 4000. * planet.get_radii() / 3., 1.e-7);
    CHECK(planet.converged());
  }
}

TEST_CASE("Original repeated planet make refreshes material properties",
          "[original][radial]") {
  auto layer = std::make_shared<Layer>("MgO", Array::LinSpaced(9, 0., 1.e6));
  layer->set_material(minerals::SLB_2011::periclase());
  layer->set_temperature_mode("user-defined", Array::Constant(9, 1000.));
  Planet planet("test", {layer});
  std::array<double, 2> values{};
  int i = 0;
  for (double maximum : {1.e9, 2.e9}) {
    planet.set_pressure_mode("user-defined", Array::LinSpaced(9, maximum, 0.));
    planet.make();
    values[static_cast<std::size_t>(i++)] = planet.get_density()[0];
    auto check = minerals::SLB_2011::periclase();
    check->set_state(maximum, 1000.);
    original::check_close(planet.get_density()[0], check->get_density(),
                          1.e-14);
  }
  CHECK(values[0] != values[1]);
}
