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
#include <cmath>
#include <stdexcept>

using burnman::optim::DummyCompositionSolution;
using burnman::optim::fit_composition_to_solution;
using burnman::optim::fit_phase_proportions_to_bulk_composition;
using Catch::Approx;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

TEST_CASE("Solution fitting normalizes molar amounts and projects covariance",
          "[composition_fitting]") {
  DummyCompositionSolution solution({"MgO", "FeO"}, {"[Mg]O", "[Fe]O"});
  const Vector observations = (Vector(3) << 2., 6., 9.).finished();
  const auto [amounts, raw_covariance, raw_residual] =
      fit_composition_to_solution(solution, {"Mg", "Fe", "O"}, observations,
                                  std::nullopt, {}, false);
  CHECK(amounts[0] == Approx(7. / 3.));
  CHECK(amounts[1] == Approx(19. / 3.));
  CHECK(raw_residual == Approx(1. / 3.));
  const Matrix amount_covariance =
      (Matrix(2, 2) << 2., -1., -1., 2.).finished() / 3.;
  const Matrix raw_projection =
      Matrix::Identity(2, 2) - amounts * Eigen::RowVector2d::Ones();
  CHECK(raw_covariance.isApprox(raw_projection * amount_covariance *
                                raw_projection.transpose()));

  const auto [fractions, covariance, residual] = fit_composition_to_solution(
      solution, {"Mg", "Fe", "O"}, observations, std::nullopt);
  CHECK(fractions[0] == Approx(7. / 26.));
  CHECK(fractions[1] == Approx(19. / 26.));
  CHECK(residual == Approx(1. / 26.));
  const Matrix projection =
      Matrix::Identity(2, 2) - fractions * Eigen::RowVector2d::Ones();
  CHECK(covariance.isApprox(projection * amount_covariance *
                            projection.transpose() /
                            std::pow(amounts.sum(), 2)));
  CHECK(covariance.rowwise().sum().norm() < 1.e-14);
  CHECK(solution.get_molar_fractions().size() == 0);
  CHECK_FALSE(solution.has_state());
}

TEST_CASE("Solution fitting applies signed conversion coefficients and exact "
          "absences",
          "[composition_fitting]") {
  DummyCompositionSolution solution({"MgFeSiO4", "MgFeSiO4"},
                                    {"[Mg][Fe]SiO4", "[Fe][Mg]SiO4"});
  const burnman::optim::VariableConversions conversions{
      {"twice_Mg", {{"Mg_A", 2.}}},
      {"Mg_equal", {{"twice_Mg", 0.5}, {"Mg_B", -1.}}}};
  const auto [fractions, covariance, residual] = fit_composition_to_solution(
      solution, {"Mg_A", "Mg_B", "Mg_equal"},
      (Vector(3) << 0.8, 0.2, 0.).finished(), std::nullopt, conversions);
  CHECK(fractions[0] == Approx(0.5));
  CHECK(fractions[1] == Approx(0.5));
  CHECK(covariance(0, 0) == Approx(1. / 6.));
  CHECK(residual == Approx(0.18));

  DummyCompositionSolution oxides({"MgO", "FeO"}, {"[Mg]O", "[Fe]O"});
  const auto repeated = fit_composition_to_solution(
      oxides, {"Mg", "Mg", "Fe"}, (Vector(3) << 1., 0., 1.).finished(),
      std::nullopt);
  CHECK(std::get<0>(repeated)[0] == Approx(0.).margin(1.e-14));
  CHECK(std::get<0>(repeated)[1] == Approx(1.));
}

TEST_CASE("Solution fitting permits signed endmembers and rejects undefined "
          "compositions",
          "[composition_fitting]") {
  DummyCompositionSolution solution({"Mg1/2Fe1/2O", "FeO"},
                                    {"[Mg1/2Fe1/2]O", "[Fe]O"});
  const auto fit = fit_composition_to_solution(
      solution, {"Mg", "Fe"}, (Vector(2) << 1., 0.).finished(), std::nullopt);
  const auto &fractions = std::get<0>(fit);
  CHECK(fractions[0] == Approx(2.));
  CHECK(fractions[1] == Approx(-1.));
  solution.set_composition(fractions.array());
  CHECK(solution.site_formula() == "[Mg1.00Fe0.00]O");
  CHECK_THROWS_AS(fit_composition_to_solution(solution, {"Mg", "Fe"},
                                              Vector::Zero(2), std::nullopt),
                  std::invalid_argument);
  CHECK_THROWS_AS(fit_composition_to_solution(solution, {"Mg", "unknown"},
                                              Vector::Ones(2), std::nullopt),
                  std::invalid_argument);
  CHECK_THROWS_AS(fit_composition_to_solution(solution, {"Mg"}, Vector::Ones(2),
                                              std::nullopt),
                  std::invalid_argument);
  CHECK_THROWS_AS(fit_composition_to_solution(solution, {"Mg", "Fe"},
                                              Vector::Ones(2), std::nullopt,
                                              {{"Mg", {{"Fe", 1.}}}}),
                  std::invalid_argument);
}

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
