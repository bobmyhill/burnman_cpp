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

#include "burnman/core/solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/thermobarometry.hpp"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace burnman;
using namespace burnman::thermobarometry;
using Catch::Approx;

TEST_CASE("Thermobarometry propagates dataset and P-T uncertainties",
          "[thermobarometry]") {
  Assemblage assemblage;
  assemblage.add_phases(
      {minerals::HP_2011_ds62::sill(), minerals::HP_2011_ds62::ky()});
  assemblage.set_fractions(Eigen::Array2d::Constant(0.5));
  assemblage.set_state(1.e8, 800.);
  const auto R = get_reaction_matrix(assemblage);
  REQUIRE(R.rows() == 1);
  CHECK((R * assemblage.get_stoichiometric_matrix()).norm() < 1.e-12);
  const auto data = minerals::HP_2011_ds62::cov();
  const auto sill = std::find(data.endmember_names.begin(),
                              data.endmember_names.end(), "sill") -
                    data.endmember_names.begin();
  const auto ky = std::find(data.endmember_names.begin(),
                            data.endmember_names.end(), "ky") -
                  data.endmember_names.begin();
  const double expected = data.covariance_matrix(sill, sill) +
                          data.covariance_matrix(ky, ky) -
                          2. * data.covariance_matrix(sill, ky);
  const auto C = assemblage_affinity_covariance_matrix(assemblage, R, data);
  CHECK(C(0, 0) == Approx(expected));
  const double affinity = (R * assemblage.get_partial_gibbs().matrix())[0];
  CHECK(assemblage_affinity_misfit(assemblage, R, data) ==
        Approx(affinity * affinity / C(0, 0)));
  assemblage.state_covariances = (Eigen::Vector2d(1.e16, 100.)).asDiagonal();
  const auto with_state =
      assemblage_affinity_covariance_matrix(assemblage, R, data, true);
  CHECK(with_state(0, 0) > C(0, 0));
  assemblage.state_priors = Eigen::Vector2d(2.e8, 810.);
  assemblage.state_inverse_covariances =
      Eigen::Vector2d(1.e-16, 0.01).asDiagonal();
  CHECK(assemblage_state_misfit(assemblage) == Approx(2.));
}

TEST_CASE("Thermobarometry estimates a polymorph transition at fixed pressure",
          "[thermobarometry]") {
  Assemblage assemblage;
  assemblage.add_phases(
      {minerals::HP_2011_ds62::sill(), minerals::HP_2011_ds62::ky()});
  assemblage.set_fractions(Eigen::Array2d::Constant(0.5));
  const auto result = estimate_conditions(
      assemblage, minerals::HP_2011_ds62::cov(), false,
      Eigen::Vector2d(5.e8, 500.), Eigen::Vector2d(1.e8, 1.e8));
  CHECK(result.success);
  CHECK(result.x[0] == 1.e8);
  CHECK(result.x[1] > 600.);
  CHECK(result.x[1] < 900.);
  CHECK(result.fun < 1.e-9);
  CHECK(result.n_reactions == 1);
  CHECK(result.n_params == 1);
  CHECK(result.degrees_of_freedom == 0);
  CHECK(std::isnan(result.fit));
  CHECK(result.xcov.rows() == 2);
  CHECK(result.dadx.cols() == 2);
  CHECK(assemblage.get_temperature() == result.x[1]);
}

TEST_CASE("Thermobarometry updates free compositions from a fixed baseline",
          "[thermobarometry]") {
  auto olivine = minerals::SLB_2024::olivine();
  olivine->set_composition((Eigen::Array2d() << 0.8, 0.2).finished());
  olivine->baseline_composition = olivine->get_molar_fractions();
  olivine->free_compositional_vectors =
      (Eigen::Matrix<double, 1, 2>() << -1., 1.).finished();
  Assemblage assemblage;
  assemblage.add_phases({olivine});
  assemblage.set_fractions(Eigen::ArrayXd::Ones(1));
  assemblage_set_state_from_params(
      assemblage, (Eigen::Vector3d() << 1.e9, 1000., 0.1).finished());
  CHECK(olivine->get_molar_fractions()[0] == Approx(0.7));
  assemblage_set_state_from_params(
      assemblage, (Eigen::Vector3d() << 1.e9, 1000., 0.2).finished());
  CHECK(olivine->get_molar_fractions()[0] == Approx(0.6));
  CHECK_THROWS_AS(assemblage_set_state_from_params(
                      assemblage, Eigen::Vector2d(1.e9, 1000.)),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      assemblage_set_state_from_params(
          assemblage, (Eigen::Vector3d() << 1.e9, 1000., 2.).finished()),
      std::invalid_argument);
  CHECK(olivine->get_molar_fractions()[0] == Approx(0.6));
}
