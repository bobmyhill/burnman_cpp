// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace burnman;
using Catch::Approx;

TEST_CASE("Mechanical mixing has no excesses or configurational entropy",
          "[core][solution_model][mechanical_solution]") {
  const Mineral mineral;
  const solution_models::MechanicalSolution model(
      {{mineral, ""}, {mineral, "[Mg]O"}, {mineral, "[Mg]2[Si]O4"}});
  CHECK(model.get_n_endmembers() == 3);
  CHECK(model.get_formulas() ==
        std::vector<std::string>{"", "[Mg]O", "[Mg]2[Si]O4"});
  CHECK(model.get_n_sites() == 0);
  CHECK(model.get_endmember_occupancies().rows() == 3);
  CHECK(model.get_endmember_occupancies().cols() == 0);
  for (const double T : {0., 300., 2000.}) {
    for (const Eigen::Array3d &x :
         {Eigen::Array3d(.2, .3, .5), Eigen::Array3d(1., 0., 0.),
          Eigen::Array3d(2., -.5, .5)}) {
      CHECK(model.compute_excess_gibbs_free_energy(1.e9, T, x) == 0.);
      CHECK(model.compute_excess_entropy(1.e9, T, x) == 0.);
      CHECK(model.compute_excess_volume(1.e9, T, x) == 0.);
      CHECK(model.compute_excess_enthalpy(1.e9, T, x) == 0.);
      CHECK(model.compute_excess_partial_gibbs_free_energies(1.e9, T, x)
                .isZero());
      CHECK(model.compute_excess_partial_entropies(1.e9, T, x).isZero());
      CHECK(model.compute_excess_partial_volumes(1.e9, T, x).isZero());
      CHECK(model.compute_activities(1.e9, T, x).isOnes());
      CHECK(model.compute_activity_coefficients(1.e9, T, x).isOnes());
      CHECK(model.compute_gibbs_hessian(1.e9, T, x).isZero());
      CHECK(model.compute_entropy_hessian(1.e9, T, x).isZero());
      CHECK(model.compute_volume_hessian(1.e9, T, x).isZero());
      CHECK(model.compute_Cp_excess(1.e9, T, x) == 0.);
      CHECK(model.compute_alphaV_excess(1.e9, T, x) == 0.);
      CHECK(model.compute_VoverKT_excess(1.e9, T, x) == 0.);
    }
  }
  CHECK_THROWS_AS(solution_models::MechanicalSolution({}),
                  std::invalid_argument);
}

TEST_CASE("Mechanical solutions preserve linear endmember properties",
          "[core][solution][mechanical_solution]") {
  auto a = minerals::SLB_2011::forsterite();
  auto b = minerals::SLB_2011::periclase();
  Solution solution;
  solution.set_solution_model(
      std::make_shared<solution_models::MechanicalSolution>(
          types::PairedEndmemberList{{*a, ""}, {*b, ""}}));
  solution.set_state(3.e9, 1200.);
  a->set_state(3.e9, 1200.);
  b->set_state(3.e9, 1200.);
  for (const Eigen::Array2d &x :
       {Eigen::Array2d(.4, .6), Eigen::Array2d(2., 1.),
        Eigen::Array2d(1.5, -.5), Eigen::Array2d(0., 1.)}) {
    solution.set_composition(x);
    const double volume =
        x[0] * a->get_molar_volume() + x[1] * b->get_molar_volume();
    CHECK(solution.get_molar_gibbs() ==
          Approx(x[0] * a->get_molar_gibbs() + x[1] * b->get_molar_gibbs()));
    CHECK(
        solution.get_molar_entropy() ==
        Approx(x[0] * a->get_molar_entropy() + x[1] * b->get_molar_entropy()));
    CHECK(solution.get_molar_volume() == Approx(volume));
    CHECK(solution.get_molar_heat_capacity_p() ==
          Approx(x[0] * a->get_molar_heat_capacity_p() +
                 x[1] * b->get_molar_heat_capacity_p()));
    CHECK(solution.get_thermal_expansivity() ==
          Approx((x[0] * a->get_molar_volume() * a->get_thermal_expansivity() +
                  x[1] * b->get_molar_volume() * b->get_thermal_expansivity()) /
                 volume));
    CHECK(solution.get_isothermal_bulk_modulus_reuss() ==
          Approx(volume / (x[0] * a->get_molar_volume() /
                               a->get_isothermal_bulk_modulus_reuss() +
                           x[1] * b->get_molar_volume() /
                               b->get_isothermal_bulk_modulus_reuss())));
    CHECK(solution.get_partial_gibbs().isApprox(
        Eigen::Array2d(a->get_molar_gibbs(), b->get_molar_gibbs())));
    CHECK(solution.get_gibbs_hessian().isZero());
  }
  const auto copy = std::dynamic_pointer_cast<Solution>(solution.copy());
  REQUIRE(copy);
  REQUIRE(std::dynamic_pointer_cast<const solution_models::MechanicalSolution>(
      copy->get_solution_model()));
  CHECK(copy->get_solution_model().get() !=
        solution.get_solution_model().get());
  copy->set_state(1.e9, 500.);
  CHECK(solution.get_solution_model()->endmembers[0].get_temperature() ==
        1200.);
}
