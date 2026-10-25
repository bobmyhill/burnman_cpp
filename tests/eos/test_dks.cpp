// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#include "burnman/eos/dks_liquid.hpp"
#include "burnman/eos/dks_solid.hpp"
#include "burnman/eos/make_eos.hpp"
#include "burnman/minerals/datasets.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace burnman;
using namespace Catch::Matchers;

TEST_CASE("DKS solid defaults to the Fortran power-law thermal relation",
          "[dks][eos]") {
  auto p = minerals::DKS_2013_solids::periclase()->params;
  CHECK(p.power_law_grueneisen == true);
  eos::DKS_S model;
  const double v = .7 * *p.V_0, t = 5000., dv = v * 1.e-5;
  const double pressure = model.compute_pressure(t, v, p);
  CHECK_THAT(model.compute_grueneisen_parameter(pressure, t, v, p),
             WithinRel(*p.grueneisen_0 * std::pow(.7, *p.q_0), 1.e-14));
  const double dfdv =
      (model.compute_helmholtz_free_energy(pressure, t, v + dv, p) -
       model.compute_helmholtz_free_energy(pressure, t, v - dv, p)) /
      (2. * dv);
  CHECK_THAT(pressure, WithinRel(-dfdv, 2.e-8));
  const double kt = -v *
                    (model.compute_pressure(t, v + dv, p) -
                     model.compute_pressure(t, v - dv, p)) /
                    (2. * dv);
  CHECK_THAT(model.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
             WithinRel(kt, 2.e-8));
  CHECK_THAT(model.compute_volume(pressure, t, p), WithinRel(v, 1.e-12));
  p.q_0 = 0.;
  CHECK_THAT(model.compute_entropy(pressure, t, v, p),
             WithinRel(*p.S_0 + *p.Cv * (std::log(t / *p.T_0) +
                                         *p.grueneisen_0 * std::log(.7)),
                       1.e-14));
}

TEST_CASE("DKS factory and reference states", "[dks][eos]") {
  auto solid = minerals::DKS_2013_solids::periclase();
  const auto &p = solid->params;
  solid->set_state(0., *p.T_0);
  CHECK_THAT(solid->get_molar_volume(), WithinRel(*p.V_0, 1.e-13));
  CHECK_THAT(solid->get_molar_internal_energy(), WithinRel(*p.E_0, 1.e-13));
  CHECK_THAT(solid->get_molar_entropy(), WithinRel(*p.S_0, 1.e-13));
  CHECK_THAT(solid->get_molar_heat_capacity_v(), WithinRel(*p.Cv, 1.e-13));
  CHECK(std::isnan(solid->get_shear_modulus()));
  CHECK(dynamic_cast<eos::DKS_S *>(eos::make_eos(types::EOSType::DKS_S).get()));
  CHECK(dynamic_cast<eos::DKS_L *>(eos::make_eos(types::EOSType::DKS_L).get()));
  auto liquid = minerals::DKS_2013_liquids::MgO_liquid();
  liquid->set_state(50.e9, 4000.);
  CHECK(liquid->get_shear_modulus() == 0.);
}

TEST_CASE("DKS derivatives and pressure-volume inversion", "[dks][eos]") {
  for (auto mineral : {minerals::DKS_2013_solids::perovskite(),
                       minerals::DKS_2013_liquids::SiO2_liquid()}) {
    const auto &p = mineral->params;
    auto model = eos::make_eos(*p.equation_of_state);
    const double t = 5000., v = .7 * *p.V_0;
    const double pressure = model->compute_pressure(t, v, p);
    const double dv = 1.e-5 * v, dt = .1;
    const auto f = [&](double temperature, double volume) {
      return model->compute_helmholtz_free_energy(pressure, temperature, volume,
                                                  p);
    };
    const double pressure_fd = -(f(t, v + dv) - f(t, v - dv)) / (2. * dv);
    const double entropy_fd = -(f(t + dt, v) - f(t - dt, v)) / (2. * dt);
    const double kt_fd = -v *
                         (model->compute_pressure(t, v + dv, p) -
                          model->compute_pressure(t, v - dv, p)) /
                         (2. * dv);
    const double cv_fd = t *
                         (model->compute_entropy(pressure, t + dt, v, p) -
                          model->compute_entropy(pressure, t - dt, v, p)) /
                         (2. * dt);
    CHECK_THAT(pressure, WithinRel(pressure_fd, 2.e-7));
    CHECK_THAT(model->compute_entropy(pressure, t, v, p),
               WithinRel(entropy_fd, 2.e-8));
    CHECK_THAT(model->compute_isothermal_bulk_modulus_reuss(pressure, t, v, p),
               WithinRel(kt_fd, 2.e-7));
    CHECK_THAT(model->compute_molar_heat_capacity_v(pressure, t, v, p),
               WithinRel(cv_fd, 2.e-8));
    CHECK_THAT(model->compute_volume(pressure, t, p), WithinRel(v, 1.e-12));
    CHECK_THROWS_AS(model->compute_pressure(0., v, p), std::domain_error);
    CHECK_THROWS_AS(model->compute_volume(pressure, -1., p), std::domain_error);
  }
}

TEST_CASE("DKS coefficient shapes and electronic reference volume",
          "[dks][eos]") {
  auto p = minerals::DKS_2013_liquids::MgO_liquid()->params;
  eos::DKS_L model;
  SECTION("Default electronic volume") {
    p.el_V_0.reset();
    model.validate_parameters(p);
    CHECK(p.el_V_0 == p.V_0);
  }
  SECTION("Missing coefficients") {
    p.a.reset();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Ragged coefficients") {
    p.a->front().pop_back();
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
  SECTION("Magnetic parameters must be paired") {
    p.spin_a = std::array<double, 2>{0., 0.};
    CHECK_THROWS_AS(model.validate_parameters(p), std::invalid_argument);
  }
}
