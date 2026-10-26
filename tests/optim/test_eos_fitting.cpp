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

#include "burnman/core/material.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/optim/eos_fitting.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

using namespace burnman;

TEST_CASE("Default MLE tolerances use fixed energy and pressure scales",
          "[eos_fitting]") {
  // No EOS: fixed tolerances must not evaluate energy properties.
  Material material;
  const auto tolerances = optim::default_mle_tolerances(
      material, {"H", "energy", "molar_internal_energy", "helmholtz",
                 "molar_helmholtz", "gibbs", "molar_gibbs", "enthalpy",
                 "molar_enthalpy", "P", "pressure", "T", "temperature"});
  CHECK((tolerances.head(9).array() == 1.).all());
  CHECK((tolerances.segment(9, 2).array() == 100.).all());
  CHECK((tolerances.tail(2).array() == 1.e-5 * 300.).all());
  CHECK(material.get_pressure() == 1.e5);
  CHECK(material.get_temperature() == 300.);
  material.set_state(1.e9, 1000.);
  CHECK(optim::default_mle_tolerances(material, {}).size() == 0);
  CHECK(material.get_pressure() == 1.e5);
  CHECK(material.get_temperature() == 300.);
}

TEST_CASE("Native mineral fitting preserves exact PTV data and applies bounds",
          "[eos_fitting]") {
  auto mineral = minerals::HP_2011_ds62::fo();
  Eigen::MatrixXd data(8, 3);
  for (Eigen::Index i = 0; i < data.rows(); ++i) {
    const double pressure = 1.e9 + static_cast<double>(i) * 1.e9 / 7.;
    mineral->set_state(pressure, *mineral->params.T_0);
    data.row(i) << pressure, *mineral->params.T_0, mineral->get_molar_volume();
  }
  optim::NonlinearFitOptions options;
  options.param_tolerance = 1.e-5;
  auto model = optim::fit_PTV_data(mineral, {"V_0", "K_0", "Kprime_0"}, data,
                                   {}, std::nullopt, std::nullopt, options);
  CHECK(model.m == mineral);
  CHECK(model.popt == model.get_params());
  CHECK(model.pcov.isZero());
  CHECK(model.WSS == 0.);
  Eigen::MatrixXd bounds(3, 2);
  bounds << 0., 1., 0., 1.e15, 3., 4.;
  model.bounds = bounds;
  Eigen::Vector3d trial(*mineral->params.V_0, *mineral->params.K_0, 5.);
  model.set_params(trial);
  CHECK(*mineral->params.Kprime_0 == 4.);
}

TEST_CASE("Default MLE relative tolerances require positive property values",
          "[eos_fitting]") {
  class VolumeMaterial : public Material {
  public:
    double volume = 4.e-5;
    double compute_molar_volume() const override { return volume; }
  } material;
  const auto tolerances =
      optim::default_mle_tolerances(material, {"V", "molar_volume"});
  CHECK((tolerances.array() == 1.e-5 * material.volume).all());
  for (const double invalid :
       {0., -1., std::numeric_limits<double>::quiet_NaN()}) {
    material.volume = invalid;
    CHECK_THROWS_AS(optim::default_mle_tolerances(material, {"V"}),
                    std::invalid_argument);
  }
}

TEST_CASE("Native PT-property fits reproduce exact Gibbs entropy enthalpy and "
          "expansivity data",
          "[eos_fitting]") {
  for (const std::string flag : {"gibbs", "S", "H", "alpha"}) {
    CAPTURE(flag);
    auto mineral = minerals::HP_2011_ds62::fo();
    Eigen::MatrixXd data(8, 3);
    for (Eigen::Index i = 0; i < data.rows(); ++i) {
      const double pressure = 1.e9 + static_cast<double>(i) * 1.e9 / 7.;
      mineral->set_state(pressure, *mineral->params.T_0);
      data.row(i) << pressure, *mineral->params.T_0,
          mineral->get_property(flag);
    }
    optim::NonlinearFitOptions options;
    options.param_tolerance = 1.e-5;
    auto model =
        optim::fit_PTp_data(mineral, {"V_0", "K_0", "Kprime_0"}, flag, data, {},
                            {}, std::nullopt, std::nullopt, options);
    CHECK(model.WSS == 0.);
    CHECK(model.pcov.isZero());
    CHECK(std::any_cast<std::string>(model.flags[0]) == flag);
  }
}
