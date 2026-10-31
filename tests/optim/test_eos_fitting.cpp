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
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace burnman;

namespace {
// Port of the binary/ternary density-normal checks in tests/test_fitting.py
// from the Python reference selected by tests/reference/burnman.json.
void check_solution_fit_density_normal(
    const std::shared_ptr<Solution> &solution,
    const std::vector<Eigen::ArrayXd> &compositions) {
  const auto n = solution->get_n_endmembers();
  for (const auto &fractions : compositions) {
    for (const auto &state :
         {Eigen::Vector2d(5.e9, 1000.), Eigen::Vector2d(25.e9, 1600.)}) {
      CAPTURE(fractions, state);
      solution->set_composition(fractions);
      solution->set_state(state[0], state[1]);
      const double density = solution->get_density();
      Eigen::VectorXd point(n + 3);
      point.head(n) = fractions.matrix();
      point.tail<3>() << state[0], state[1], density;
      Eigen::MatrixXd data(1, point.size());
      data.row(0) = point.transpose();
      Eigen::MatrixXd covariance =
          Eigen::MatrixXd::Zero(point.size(), point.size());
      covariance.diagonal().tail<3>() << 1.e12, 1., 1.;
      optim::SolutionFit model(solution, data, {covariance}, {"rho"},
                               {{"V_0", 0}},
                               Eigen::VectorXd::Constant(1, 1.e-8));
      // d(rho)/dP = rho/K_T and d(rho)/dT = -alpha*rho.
      Eigen::VectorXd expected = Eigen::VectorXd::Zero(point.size());
      expected.tail<3>() << -density /
                                solution->get_isothermal_bulk_modulus_reuss(),
          solution->get_thermal_expansivity() * density, 1.;
      expected.normalize();
      const auto normal = model.normal(point, std::string("rho"));
      REQUIRE(normal.allFinite());
      for (Eigen::Index i = 0; i < point.size(); ++i)
        CHECK_THAT(
            normal[i],
            Catch::Matchers::WithinAbs(
                expected[i],
                std::max(1.e-16, 1.e-6 * std::max(std::abs(normal[i]),
                                                  std::abs(expected[i])))));
    }
  }
}
} // namespace

TEST_CASE("Original binary solution density fitting normal",
          "[eos_fitting][solution_density_normal][original]") {
  check_solution_fit_density_normal(
      minerals::SLB_2011::mg_fe_olivine(),
      {Eigen::Array2d(.7, .3), Eigen::Array2d(.2, .8)});
}

TEST_CASE("Original ternary solution density fitting normal",
          "[eos_fitting][solution_density_normal][original]") {
  check_solution_fit_density_normal(
      minerals::SLB_2011::mg_fe_perovskite(),
      {Eigen::Array3d(.7, .2, .1), Eigen::Array3d(.2, .5, .3)});
}

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

TEST_CASE(
    "Native solution fitting shares parameters and uses pressure weighting",
    "[eos_fitting]") {
  auto solution = minerals::SLB_2011::mg_fe_olivine();
  Eigen::MatrixXd data(12, 5);
  for (Eigen::Index i = 0; i < data.rows(); ++i) {
    const double fraction = (static_cast<double>(i) + 0.5) / 12.;
    const double pressure = static_cast<double>(i % 4) * 1.e9;
    solution->set_composition(Eigen::Array2d(1. - fraction, fraction));
    solution->set_state(pressure, 800.);
    data.row(i) << 1. - fraction, fraction, pressure, 800.,
        solution->get_molar_volume();
  }
  optim::NonlinearFitOptions options;
  options.verbose = false;
  auto model = optim::fit_XPTp_data(
      solution, {{"V_0", 0}, {"V_0", 1}, {"V", 0, 1}}, std::string("V"), data,
      {}, {}, std::nullopt, std::nullopt, options);
  CHECK(model.m == solution);
  CHECK(model.popt == model.get_params());
  CHECK(model.WSS == 0.);
  CHECK(model.pcov.isZero());
  CHECK(model.data_covariances[0](2, 2) == 1.);
  CHECK(model.data_covariances[0].sum() == 1.);
  CHECK(model.fit_params_strings ==
        std::vector<std::string>{"V_0 (0)", "V_0 (1)", "V (0,1)"});
  const auto point = data.row(4).transpose().eval();
  CHECK(model.normal(point).head(2).isZero());
  const double old_volume = model.function(point)[4];
  Eigen::VectorXd trial = model.popt;
  trial[0] *= 1.001;
  trial[2] = 1.e-7;
  model.set_params(trial);
  CHECK(model.get_params().isApprox(trial));
  CHECK(solution->get_molar_volume() != old_volume);
  CHECK(model.function(point)[4] != old_volume);
}
