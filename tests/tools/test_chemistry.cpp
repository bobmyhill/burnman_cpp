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

#include "burnman/core/combined_mineral.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/chemistry.hpp"
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "burnman/tools/partitioning.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace burnman;

TEST_CASE("Hugoniot derivatives include phase amounts and solution composition",
          "[chemistry]") {
  const auto olivine = minerals::SLB_2011::mg_fe_olivine();
  olivine->set_composition(Eigen::Array2d(.7, .3));
  Assemblage a;
  a.add_phases({olivine, minerals::SLB_2011::periclase()});
  a.set_fractions(Eigen::Array2d(.6, .4));
  a.set_n_moles(2.);
  a.set_state(1.e5, 300.);
  const equilibration::HugoniotConstraint constraint(
      1.e5, a.get_molar_internal_energy() / a.get_molar_mass(),
      a.get_molar_volume() / a.get_molar_mass());
  a.set_state(10.e9, 1000.);
  const auto initial = equilibration::get_parameter_vector(a);
  const auto derivative = constraint.derivative(initial, a, initial.size());
  for (Eigen::Index j = 0; j < initial.size(); ++j) {
    const double step = j == 0 ? 1.e5 : (j == 1 ? .01 : 1.e-5);
    auto x = initial;
    x[j] += step;
    equilibration::set_composition_and_state_from_parameters(a, x);
    const double plus = constraint.evaluate(x, a);
    x[j] -= 2. * step;
    equilibration::set_composition_and_state_from_parameters(a, x);
    const double minus = constraint.evaluate(x, a);
    CHECK_THAT((plus - minus) / (2. * step),
               Catch::Matchers::WithinRel(derivative[j], 2.e-6));
  }
}

TEST_CASE("Reaction equilibria recover independent energy offsets",
          "[chemistry]") {
  const auto original = minerals::HP_2011_ds62::fo();
  const auto thermal = std::make_shared<Mineral>(make_combined_mineral(
      {*original}, Eigen::ArrayXd::Ones(1), {1000., 2., 0.}));
  const auto pressure = std::make_shared<Mineral>(make_combined_mineral(
      {*original}, Eigen::ArrayXd::Ones(1), {-2000., 0., 1.e-6}));
  const Eigen::Vector2d reaction(-1., 1.);
  CHECK_THAT(
      chemistry::equilibrium_temperature({original, thermal}, reaction, 2.e9),
      Catch::Matchers::WithinAbs(500., 1.e-5));
  CHECK_THAT(
      chemistry::equilibrium_pressure({original, pressure}, reaction, 500.),
      Catch::Matchers::WithinAbs(2.e9, 2.));
  const auto point = chemistry::invariant_point({original, thermal}, reaction,
                                                {original, pressure}, reaction);
  CHECK_THAT(point[0], Catch::Matchers::WithinAbs(2.e9, 2.));
  CHECK_THAT(point[1], Catch::Matchers::WithinAbs(500., 1.e-5));
  CHECK(original->get_pressure() == point[0]);
  CHECK(thermal->get_temperature() == point[1]);
}

TEST_CASE("Component chemical potentials conserve elemental composition",
          "[chemistry]") {
  const auto pv = minerals::SLB_2011::mg_perovskite();
  const auto per = minerals::SLB_2011::periclase();
  Assemblage assemblage;
  assemblage.add_phases({pv, per});
  assemblage.set_fractions(Eigen::Array2d(.5, .5));
  assemblage.set_state(25.e9, 2000.);
  const auto potentials = assemblage.chemical_potential(
      {{{"Si", 1.}, {"O", 2.}}, {{"Si", -1.}, {"Mg", 1.}, {"O", -1.}}});
  CHECK_THAT(potentials[0],
             Catch::Matchers::WithinAbs(
                 pv->get_molar_gibbs() - per->get_molar_gibbs(), 1.e-8));
  CHECK_THAT(potentials[1],
             Catch::Matchers::WithinAbs(
                 2. * per->get_molar_gibbs() - pv->get_molar_gibbs(), 1.e-8));
  CHECK_THROWS_AS(assemblage.chemical_potential({{{"Si", 1.}}}),
                  std::invalid_argument);
}

TEST_CASE("Reaction enumeration returns balanced primitive circuits",
          "[chemistry]") {
  Eigen::MatrixXd composition(4, 2);
  composition << 1., 0., 0., 1., 1., 1., 1., 1.;
  const auto reactions =
      chemistry::reactions_from_stoichiometric_matrix(composition);
  REQUIRE(reactions.rows() == 6);
  REQUIRE(reactions.cols() == 4);
  CHECK((reactions * composition).norm() < 1.e-12);
  for (Eigen::Index i = 0; i < reactions.rows(); ++i)
    CHECK((reactions.row(i) + reactions.row(reactions.rows() - i - 1)).norm() <
          1.e-12);
}

TEST_CASE("Hugoniot energy balance is independent of formula-unit scale",
          "[chemistry]") {
  const auto material = minerals::SLB_2011::periclase();
  auto reference =
      make_combined_mineral({*material}, Eigen::ArrayXd::Constant(1, 2.));
  const Eigen::Array3d pressures(1.e5, 10.e9, 30.e9);
  const auto ordinary = chemistry::hugoniot(*material, 1.e5, 300., pressures);
  const auto doubled =
      chemistry::hugoniot(*material, 1.e5, 300., pressures, &reference);
  for (Eigen::Index i = 0; i < pressures.size(); ++i) {
    CHECK_THAT(doubled.first[i],
               Catch::Matchers::WithinAbs(ordinary.first[i], 1.e-8));
    CHECK_THAT(doubled.second[i],
               Catch::Matchers::WithinRel(ordinary.second[i], 1.e-10));
  }
}

TEST_CASE("Iron partitioning respects bulk balance and unit exchange",
          "[partitioning]") {
  const types::FormulaMap bulk{{"Mg", .8}, {"Fe", .2}, {"Si", .6}};
  for (const double Kd : {.01, .5, 1., 2., 100.}) {
    const auto [fp, pv] =
        partitioning::calculate_nakajima_fp_pv_partition_coefficient(
            25.e9, 2000., bulk, Kd);
    CHECK_THAT(.4 * fp + .6 * pv, Catch::Matchers::WithinAbs(.2, 1.e-14));
    CHECK_THAT(pv * (1. - fp) / (fp * (1. - pv)),
               Catch::Matchers::WithinRel(Kd, 1.e-12));
  }
  const auto [fp, pv] =
      partitioning::calculate_nakajima_fp_pv_partition_coefficient(25.e9, 2000.,
                                                                   bulk, 1.);
  CHECK(fp == .2);
  CHECK(pv == .2);
}
