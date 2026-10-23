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

#include "burnman/core/geotherm.hpp"
#include "burnman/core/material.hpp"
#include "burnman/core/mineral.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using Catch::Approx;
using namespace burnman::geotherm;
using Array = Eigen::ArrayXd;

namespace {
class AnalyticMaterial : public burnman::Material {
protected:
  double compute_molar_entropy() const override {
    if (get_pressure() < -0.5e9)
      throw std::runtime_error("Pressure outside analytic material domain.");
    return 100.0 * std::log(get_temperature() / 1000.0) -
           10.0 * std::log((get_pressure() + 1.0e9) / 1.0e9);
  }
  double compute_grueneisen_parameter() const override { return 1.0; }
  double compute_isentropic_bulk_modulus_reuss() const override {
    return 10.0 * (get_pressure() + 1.0e9);
  }
};
} // namespace

TEST_CASE("Tabulated geotherms match the upstream reference temperatures",
          "[geotherm]") {
  CHECK(BrownShankland().temperatures(600.0e3) == Approx(1825.0));
  CHECK(Anderson().temperatures(621.0e3) == Approx(1805.0));
  CHECK(StaceyContinental().temperatures(11.0e3) == Approx(540.0));
  CHECK(StaceyContinental().temperatures(620.0e3) == Approx(2225.0));
  CHECK(StaceyOceanic().temperatures(11.0e3) == Approx(550.0));
  CHECK(StaceyOceanic().temperatures(620.0e3) == Approx(2225.0));
  CHECK(Katsura2022().temperatures(100.0e3) == Approx(1672.0));
  CHECK(Katsura2022().temperatures(400.0e3) == Approx(1796.0));
  CHECK(Plesa2022Mars6cm3().temperatures(173.0e3) == Approx(1028.0));
  CHECK(Plesa2022Mars6cm3().temperatures(320.0e3) == Approx(1489.0));
  CHECK(Anzellini2013().temperatures(100.0e3) ==
        Approx(1603.017).margin(0.001));
  CHECK(Anzellini2013().temperatures(700.0e3) ==
        Approx(1952.184).margin(0.001));
}

TEST_CASE("Geotherm interpolation retains the shallower discontinuity value",
          "[geotherm]") {
  Geotherm model((Array(4) << 0.0, 10.0, 10.0, 20.0).finished(),
                 (Array(4) << 100.0, 200.0, 400.0, 600.0).finished());
  CHECK(model.minimum_depth == 0.0);
  CHECK(model.maximum_depth == 20.0);
  CHECK(model.temperatures(0.0) == 100.0);
  CHECK(model.temperatures(5.0) == 150.0);
  CHECK(model.temperatures(10.0) == 200.0);
  CHECK(model.temperatures(15.0) == 500.0);
  CHECK(model.temperatures(20.0) == 600.0);
  CHECK(model.temperatures(Array(0)).size() == 0);
  CHECK_THROWS_AS(model.temperatures(-1.0), std::invalid_argument);
  CHECK_THROWS_AS(model.temperatures(21.0), std::invalid_argument);
  CHECK_THROWS_AS(model.temperatures(std::numeric_limits<double>::quiet_NaN()),
                  std::invalid_argument);
  CHECK_THROWS_AS(Geotherm(Array(0), Array(0)), std::invalid_argument);
  CHECK_THROWS_AS(Geotherm(Array::Ones(3), Array::Ones(2)),
                  std::invalid_argument);
  CHECK_THROWS_AS(Geotherm(Array::LinSpaced(3, 2.0, 0.0), Array::Ones(3)),
                  std::invalid_argument);
}

TEST_CASE("Pressure geotherms use a selectable seismic reference",
          "[geotherm]") {
  auto seismic = std::make_shared<burnman::seismic::SeismicTable>();
  seismic->table_depth = (Array(2) << 0.0, 1000.0).finished();
  seismic->table_pressure = (Array(2) << 0.0, 1.0e9).finished();
  GeothermFromPressures model((Array(2) << 0.0, 1.0e9).finished(),
                              (Array(2) << 300.0, 1300.0).finished(), seismic);
  CHECK(model.temperatures(500.0) == 800.0);
  CHECK(model.temperatures_from_pressures(0.5e9) == 800.0);
  CHECK_THROWS_AS(model.temperatures_from_pressures(1.1e9),
                  std::invalid_argument);
  Anzellini2013 prem;
  CHECK(prem.depth_to_pressure_model.get() == &burnman::seismic::prem_model());
}

TEST_CASE("Material adiabats conserve entropy on unordered pressure queries",
          "[geotherm]") {
  auto material = std::make_shared<AnalyticMaterial>();
  const double anchor = 2.0e9;
  const double T_0 = 1500.0;
  const Array pressures =
      (Array(6) << 2.0e9, 8.0e9, 0.0, 3.0e9, 3.0e9, 2.0e9).finished();
  const auto temperatures =
      adiabatic_profile(pressures, *material, T_0, anchor);
  for (Eigen::Index i = 0; i < pressures.size(); ++i)
    CHECK(temperatures[i] ==
          Approx(T_0 * std::pow((pressures[i] + 1.0e9) / (anchor + 1.0e9), 0.1))
              .epsilon(1.0e-12));
  CHECK(material->get_pressure() == anchor);
  CHECK(material->get_temperature() == T_0);
  const auto default_anchor = adiabatic_profile(pressures, *material, T_0);
  CHECK((temperatures - default_anchor).abs().maxCoeff() < 1.0e-9);
  AdiabaticGeotherm geotherm(material, T_0, anchor);
  CHECK((geotherm.temperatures_from_pressures(pressures) - temperatures)
            .abs()
            .maxCoeff() < 1.0e-9);
  CHECK_THROWS_AS(
      adiabatic_profile(Array::Constant(1, -1.0e9), *material, T_0, anchor),
      std::runtime_error);
  CHECK(material->get_pressure() == anchor);
  CHECK(material->get_temperature() == T_0);
  CHECK(adiabatic_profile(Array(0), *material, T_0).size() == 0);
  CHECK_THROWS_AS(AdiabaticGeotherm(nullptr, T_0, anchor),
                  std::invalid_argument);
  CHECK_THROWS_AS(adiabatic_profile(pressures, *material, -1.0),
                  std::invalid_argument);
}

TEST_CASE("Native periclase adiabat matches the upstream SLB3 regression",
          "[geotherm]") {
  burnman::Mineral rock;
  rock.params.V_0 = 11.24e-6;
  rock.params.K_0 = 161.0e9;
  rock.params.Kprime_0 = 3.8;
  rock.params.G_0 = 131.0e9;
  rock.params.Gprime_0 = 2.1;
  rock.params.molar_mass = 0.0403;
  rock.params.napfu = 2.0;
  rock.params.debye_0 = 773.0;
  rock.params.grueneisen_0 = 1.5;
  rock.params.q_0 = 1.5;
  rock.params.eta_s_0 = 2.8;
  rock.set_method(burnman::types::EOSType::SLB3);
  const Array pressures = (Array(2) << 100.0e9, 150.0e9).finished();
  const auto temperatures = adiabatic_profile(pressures, rock, 1500.0);
  CHECK(temperatures[0] == 1500.0);
  // The historical upstream regression is rounded relative to the current
  // entropy solve; live Python parity is checked separately.
  CHECK(temperatures[1] == Approx(1650.22034002).margin(1.0e-4));
}
