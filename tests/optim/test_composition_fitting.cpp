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

#include "burnman/optim/composition_fitting.hpp"
#include "burnman/utils/exceptions.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using burnman::optim::DummyCompositionSolution;
using burnman::optim::fit_phase_proportions_to_bulk_composition;
using Catch::Approx;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

TEST_CASE(
    "Phase amounts recover the bulk inventory without a unit-sum constraint",
    "[composition_fitting]") {
  Matrix phases(3, 2);
  phases << 1., 0., 0., 1., 1., 1.;
  Vector bulk(3);
  bulk << 1., 3., 4.;
  const auto [amounts, covariance, residual] =
      fit_phase_proportions_to_bulk_composition(phases, bulk);
  CHECK(amounts[0] == Approx(1.));
  CHECK(amounts[1] == Approx(3.));
  CHECK((phases * amounts - bulk).norm() < 1.e-13);
  CHECK(covariance(0, 0) == Approx(2. / 3.));
  CHECK(covariance(0, 1) == Approx(-1. / 3.));
  CHECK(covariance(1, 1) == Approx(2. / 3.));
  CHECK(residual == Approx(0.).margin(1.e-25));
}

TEST_CASE("Phase fitting respects nonnegativity when no exact mixture exists",
          "[composition_fitting]") {
  Matrix phases(3, 2);
  phases << 1., 0., 0., 1., 1., 1.;
  Vector bulk(3);
  bulk << -1., 2., 0.;
  const auto [amounts, covariance, residual] =
      fit_phase_proportions_to_bulk_composition(phases, bulk);
  CHECK(amounts[0] == Approx(0.).margin(1.e-10));
  CHECK(amounts[1] == Approx(1.));
  CHECK(residual == Approx(3.));
  CHECK(covariance(0, 0) == Approx(2. / 3.));
  CHECK_THROWS_AS(
      fit_phase_proportions_to_bulk_composition(phases, Vector::Ones(2)),
      std::invalid_argument);
  phases.col(1) = phases.col(0);
  CHECK_THROWS_AS(fit_phase_proportions_to_bulk_composition(phases, bulk),
                  std::invalid_argument);
}

TEST_CASE("Composition-only solutions provide elemental and site chemistry",
          "[composition_fitting]") {
  DummyCompositionSolution olivine({"Mg2SiO4", "Fe2SiO4"},
                                   {"[Mg]2SiO4", "[Fe]2SiO4"});
  CHECK(olivine.get_n_endmembers() == 2);
  CHECK(olivine.get_n_elements() == 4);
  CHECK(olivine.get_site_names() == std::vector<std::string>{"Mg_A", "Fe_A"});
  CHECK(olivine.get_endmember_formulae()[0].at("Mg") == Approx(2.));
  CHECK(olivine.get_stoichiometric_matrix().cols() == 4);
  CHECK(olivine.get_endmember_occupancies().matrix().isIdentity());
  CHECK(olivine.get_endmember_n_occupancies().matrix().isApprox(
      2. * Matrix::Identity(2, 2)));
  CHECK_THROWS_AS(olivine.get_site_occupancies(), std::runtime_error);
  CHECK_THROWS_AS(olivine.site_formula(), std::runtime_error);

  olivine.set_composition((Eigen::Array2d() << 0.2, 0.8).finished());
  CHECK(olivine.get_site_occupancies().isApprox(
      (Eigen::Array2d() << 0.2, 0.8).finished()));
  CHECK(olivine.get_formula().at("Mg") == Approx(0.4));
  CHECK(olivine.get_formula().at("Fe") == Approx(1.6));
  CHECK(olivine.get_formula().at("Si") == Approx(1.));
  CHECK(olivine.site_formula() == "[Mg0.20Fe0.80]2SiO4");
  CHECK_FALSE(olivine.has_state());

  auto copied =
      std::dynamic_pointer_cast<DummyCompositionSolution>(olivine.copy());
  REQUIRE(copied);
  CHECK(copied->get_solution_model() != olivine.get_solution_model());
  copied->set_composition((Eigen::Array2d() << 1., 0.).finished());
  CHECK(copied->site_formula(3, false) == "[Mg1.000]2SiO4");
  CHECK(copied->get_formula().at("Mg") == Approx(2.));
  CHECK(olivine.site_formula() == "[Mg0.20Fe0.80]2SiO4");
  CHECK(olivine.site_formula(1, false,
                             (Eigen::Array2d() << -1.e-15, 1.).finished()) ==
        "[Fe1.0]2SiO4");
  CHECK_THROWS_AS(olivine.site_formula(-1), std::invalid_argument);
  CHECK_THROWS_AS(olivine.site_formula(2, true, Eigen::ArrayXd::Ones(3)),
                  std::invalid_argument);
}

TEST_CASE("Composition-only solutions keep physical site occupancies",
          "[composition_fitting]") {
  DummyCompositionSolution solution({"MgO", "FeO", "Mg1/2Fe1/2O"},
                                    {"[Mg]O", "[Fe]O", "[Mg1/2Fe1/2]O"});
  // Negative endmember coordinates can represent a physical composition.
  solution.set_composition((Eigen::Array3d() << -0.2, 0.4, 0.8).finished());
  CHECK(solution.site_formula() == "[Mg0.20Fe0.80]O");
  CHECK(solution.get_formula().at("Mg") == Approx(0.2));
  CHECK_THROWS_AS(
      solution.set_composition((Eigen::Array3d() << -0.5, 1.5, 0.).finished()),
      std::invalid_argument);
  CHECK_THROWS_AS(solution.set_composition(Eigen::ArrayXd::Ones(2)),
                  std::invalid_argument);
  CHECK_THROWS_AS(solution.set_composition(Eigen::ArrayXd::Zero(3)),
                  std::invalid_argument);
  CHECK(solution.site_formula() == "[Mg0.20Fe0.80]O");

  solution.set_state(1.e9, 1000.);
  CHECK_THROWS_AS(solution.get_molar_volume(),
                  burnman::exceptions::NotImplementedError);
  CHECK_THROWS_AS(solution.get_molar_gibbs(),
                  burnman::exceptions::NotImplementedError);
  CHECK_THROWS_AS(solution.get_activities(),
                  burnman::exceptions::NotImplementedError);
  CHECK_THROWS_AS(solution.get_gibbs_hessian(),
                  burnman::exceptions::NotImplementedError);
}

TEST_CASE("Composition-only solutions reject inconsistent formulae",
          "[composition_fitting]") {
  CHECK_THROWS_AS(DummyCompositionSolution({}, {}), std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"[Mg]O", "[Fe]O"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"XxO"}, {"[Xx]O"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"MgO"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"[MgO"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"[]O"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"[[Mg]]O"}),
                  std::invalid_argument);
  CHECK_THROWS_AS(DummyCompositionSolution({"MgO"}, {"[Mg0.5]O"}),
                  std::invalid_argument);
  CHECK_THROWS(DummyCompositionSolution({"MgO", "FeO"}, {"[Mg]O", "[Fe][O]"}));
}
