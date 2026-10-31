// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Analytical and upstream radial regressions; reference:
// tests/reference/burnman.json.

#include "../../src/core/radial_profile.hpp"
#include "burnman/core/layer.hpp"
#include "burnman/core/planet.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/warnings.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <sstream>
#include <utility>

namespace {
using namespace burnman;
using Array = Eigen::ArrayXd;
constexpr double pi = 3.14159265358979323846;
class ProfileMaterial : public Material {
public:
  explicit ProfileMaterial(bool compressible = false)
      : compressible_(compressible) {}
  std::shared_ptr<Material> copy() const override {
    return std::make_shared<ProfileMaterial>(*this);
  }

protected:
  double compute_density() const override {
    return 4000. * (1. + (compressible_ ? get_pressure() / 1.e11 : 0.));
  }
  double compute_bulk_sound_velocity() const override {
    const double p = get_pressure();
    return std::sqrt(1.e6 + 1.e-4 * p + 1.e-16 * p * p);
  }

private:
  bool compressible_;
};
std::shared_ptr<Layer> profile_layer(bool compressible = false,
                                     double inner = 0., int count = 9) {
  auto layer =
      std::make_shared<Layer>("profile", Array::LinSpaced(count, inner, 2.e6));
  layer->set_material(std::make_shared<ProfileMaterial>(compressible));
  layer->set_temperature_mode("user-defined", Array::Constant(count, 300.));
  return layer;
}
class MovingProfileLayer : public Layer {
public:
  MovingProfileLayer() : Layer("moving", Array::LinSpaced(3, 0., 1.e6)) {
    set_material(std::make_shared<ProfileMaterial>());
    set_temperature_mode("user-defined", Array::Constant(3, 300.));
  }
  Array compute_pressure(const Array &, const Array &,
                         double top) const override {
    Array result(3);
    result << 1.e9, ++updates_ == 1 ? 8.e8 : 2.e8, top;
    return result;
  }

private:
  mutable int updates_ = 0;
};
struct CaptureWarnings {
  std::ostringstream output;
  std::streambuf *previous = std::cerr.rdbuf(output.rdbuf());
  bool suppressed = std::exchange(utils::suppress_warnings, false);
  ~CaptureWarnings() {
    std::cerr.rdbuf(previous);
    utils::suppress_warnings = suppressed;
  }
};
} // namespace

TEST_CASE("Radial spline preserves smooth polynomials and supported extrema",
          "[layer][radial]") {
  for (const int count : {2, 3, 4, 9}) {
    Array x = Array::LinSpaced(count, 0., 1.);
    if (count > 3)
      x = x.square();
    for (int profile = 0; profile < 4; ++profile) {
      if (count < 4 && profile == 1)
        continue;
      if (count < 3 && profile > 1)
        continue;
      const auto value = [profile](double r) {
        if (profile == 0)
          return 4000. + 1000. * r;
        if (profile == 1)
          return 8000. - 4000. * r * r * r;
        const double offset = r - .53;
        return 6000. + (profile == 2 ? 1. : -1.) * 2000. * offset * offset;
      };
      const Array density = x.unaryExpr(value);
      const detail::RadialSpline spline(x * 1.e6, density);
      for (int j = 0; j <= 100; ++j) {
        const double r = static_cast<double>(j) / 100.;
        CHECK_THAT(spline(r * 1.e6),
                   Catch::Matchers::WithinAbs(value(r), 1.e-10));
      }
    }
  }
}

TEST_CASE("Radial spline bounds density steps on uniform and uneven grids",
          "[layer][radial]") {
  Array uneven(9);
  uneven << 0., .05, .2, .3, .45, .47, .6, .85, 1.;
  for (const Array &x : {Array(Array::LinSpaced(17, 0., 1.)), uneven})
    for (const double interface : {0., .45, .99})
      for (const double direction : {-1., 1.})
        for (const double scale : {1.e-6, 1.}) {
          const Array density = x.unaryExpr([&](double r) {
            return scale *
                   (6000. + direction * (r <= interface ? 2000. : -2000.));
          });
          const Array radii = x * (scale == 1. ? 1.e6 : 1.);
          const detail::RadialSpline spline(radii, density);
          const double tolerance = 128. *
                                   std::numeric_limits<double>::epsilon() *
                                   density.maxCoeff();
          for (Eigen::Index i = 0; i + 1 < x.size(); ++i)
            for (int j = 0; j <= 32; ++j) {
              const double r = radii[i] + (radii[i + 1] - radii[i]) *
                                              static_cast<double>(j) / 32.;
              const double value = spline(r);
              CHECK(value >= std::min(density[i], density[i + 1]) - tolerance);
              CHECK(value <= std::max(density[i], density[i + 1]) + tolerance);
            }
        }
}

TEST_CASE("Radial convergence checks changes away from maximum pressure",
          "[layer][planet][radial]") {
  auto layer = std::make_shared<MovingProfileLayer>();
  SECTION("Layer") {
    layer->set_pressure_mode("self-consistent", {}, 0., 0., 3, 1.e-12);
    layer->make();
    CHECK(layer->converged());
    CHECK(layer->get_n_iterations() == 3);
    CHECK(layer->get_pressures()[1] == 2.e8);
  }
  SECTION("Planet") {
    Planet planet("moving", {layer}, 3, 1.e-12);
    planet.make();
    CHECK(planet.converged());
    CHECK(planet.get_n_iterations() == 3);
    CHECK(layer->get_n_iterations() == 3);
    CHECK(planet.get_pressures()[1] == 2.e8);
  }
}

TEST_CASE("Final planetary gravity matches mass even without convergence",
          "[planet][radial]") {
  auto layer = profile_layer(true);
  Planet planet("one iteration", {layer}, 1, 1.e-14);
  CaptureWarnings warnings;
  planet.make();
  CHECK_FALSE(planet.converged());
  CHECK_FALSE(layer->converged());
  CHECK(planet.get_n_iterations() == 1);
  CHECK(warnings.output.str().find("did not converge") != std::string::npos);
  const double radius = planet.get_radius_planet();
  CHECK_THAT(planet.get_gravity().tail(1)[0],
             Catch::Matchers::WithinRel(constants::physics::gravitation *
                                            planet.get_mass() /
                                            (radius * radius),
                                        1.e-13));
  CHECK((planet.get_gravity() - layer->get_gravity()).abs().maxCoeff() <
        1.e-12);
}

TEST_CASE("Hollow planet preserves its prescribed inner gravity",
          "[planet][radial]") {
  const double inner = 1.e6, outer = 2.e6, rho = 4000., g0 = 5., p0 = 1.e5;
  auto layer = profile_layer(false, inner, 257);
  Planet planet("shell", {layer});
  planet.set_pressure_mode("self-consistent", {}, p0, g0, 50, 1.e-12);
  planet.make();
  const double a = 4. * pi * constants::physics::gravitation * rho / 3.;
  const double b = (g0 - a * inner) * inner * inner;
  const auto gravity = planet.get_gravity(), pressure = planet.get_pressures();
  for (Eigen::Index i = 0; i < layer->n_slices; ++i) {
    const double r = layer->radii[i];
    CHECK_THAT(gravity[i],
               Catch::Matchers::WithinRel(a * r + b / (r * r), 1.e-13));
    CHECK_THAT(pressure[i], Catch::Matchers::WithinRel(
                                p0 + rho * (.5 * a * (outer * outer - r * r) +
                                            b * (1. / r - 1. / outer)),
                                1.e-8));
  }
  CHECK(*layer->get_gravity_bottom() == g0);
  CHECK(gravity[0] == g0);
}

TEST_CASE("Repeated prescribed planets refresh cached material states",
          "[planet][radial]") {
  auto layer = profile_layer(true);
  Planet planet("prescribed", {layer});
  for (double maximum : {1.e9, 2.e9}) {
    const Array pressure = Array::LinSpaced(layer->n_slices, maximum, 0.);
    planet.set_pressure_mode("user-defined", pressure);
    planet.make();
    const auto density = planet.get_density();
    CHECK(planet.converged());
    CHECK(density[0] == 4000. * (1. + maximum / 1.e11));
    CHECK((planet.get_pressures() - pressure).abs().maxCoeff() == 0.);
    CHECK((layer->get_pressures() - pressure).abs().maxCoeff() == 0.);
  }
  CHECK_THROWS_AS(layer->set_pressure_mode("self-consistent", {}, 1., 0.),
                  std::invalid_argument);
  CHECK_THROWS_AS(planet.set_pressure_mode("self-consistent", {}, 0., 1.),
                  std::invalid_argument);
}

TEST_CASE("Bullen derivatives use uneven physical coordinates",
          "[layer][radial]") {
  Array radii(5);
  radii << 0., .1e6, .4e6, .8e6, 1.e6;
  Layer layer("derivatives", radii);
  layer.set_material(std::make_shared<ProfileMaterial>());
  layer.set_temperature_mode("user-defined", Array::Constant(5, 300.));
  const Array pressure = 4.e9 - 3000. * radii;
  layer.set_pressure_mode("user-defined", pressure, 0.);
  layer.make();
  const Array bullen = layer.get_bullen(), gravity = layer.get_gravity();
  CHECK(std::isnan(bullen[0]));
  CHECK(std::isnan(layer.get_brunt_vasala()[0]));
  for (Eigen::Index i = 1; i < radii.size(); ++i) {
    const double derivative = 1.e-4 + 2.e-16 * pressure[i];
    CHECK_THAT(bullen[i],
               Catch::Matchers::WithinAbs(4000. * derivative -
                                              3000. * derivative / gravity[i],
                                          1.e-13));
  }
}
