// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/relaxed_solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace burnman;
using Catch::Approx;

TEST_CASE(
    "Polynomial mineral interactions provide consistent thermal derivatives",
    "[core][polynomial_solution]") {
  auto a = minerals::SLB_2011::forsterite();
  auto b = minerals::SLB_2011::fayalite();
  auto model = std::make_shared<solution_models::PolynomialSolution>(
      types::PairedEndmemberList{{*a, "[Mg]2SiO4"}, {*b, "[Fe]2SiO4"}},
      solution_models::PolynomialSolution::Interactions{
          {12000., 3., 1.e-6, 0, 1, 1, 1}},
      std::vector<std::shared_ptr<Mineral>>{a, b},
      solution_models::PolynomialSolution::Interactions{{.2, -.2, 0, 1, 1, 1}});
  Solution s;
  s.set_solution_model(model);
  s.set_composition(Eigen::Array2d(.4, .6));
  s.set_state(10.e9, 1000.);
  const double v = s.get_molar_volume(),
               kt = s.get_isothermal_bulk_modulus_reuss(),
               alpha = s.get_thermal_expansivity(),
               cp = s.get_molar_heat_capacity_p();
  s.set_state(10.e9 - 1.e5, 1000.);
  const double vm = s.get_molar_volume();
  s.set_state(10.e9 + 1.e5, 1000.);
  CHECK(-(s.get_molar_volume() - vm) / 2.e5 / v ==
        Approx(1. / kt).epsilon(1.e-7));
  s.set_state(10.e9, 999.99);
  const double vt = s.get_molar_volume(), entropy = s.get_molar_entropy();
  s.set_state(10.e9, 1000.01);
  CHECK((s.get_molar_volume() - vt) / .02 / v == Approx(alpha).epsilon(1.e-7));
  CHECK(1000. * (s.get_molar_entropy() - entropy) / .02 ==
        Approx(cp).epsilon(1.e-7));
  const auto values =
      s.evaluate_with_volumes({"pressure", "V"}, Eigen::ArrayXd::Constant(1, v),
                              Eigen::ArrayXd::Constant(1, 1000.));
  CHECK(values(0, 0) == Approx(10.e9).epsilon(1.e-10));
  CHECK(values(1, 0) == Approx(v).epsilon(1.e-12));
  CHECK(s.get_temperature() == 1000.01);
  const auto copy =
      std::dynamic_pointer_cast<solution_models::PolynomialSolution>(
          model->clone());
  REQUIRE(copy);
  CHECK(copy->get_interaction_endmembers()[0].get() !=
        model->get_interaction_endmembers()[0].get());
  for (const Eigen::Array2d x :
       {Eigen::Array2d(1., 0.), Eigen::Array2d(0., 1.)}) {
    CHECK(model->compute_excess_partial_volumes(1.e9, 300., x).allFinite());
    CHECK(model->compute_volume_hessian(1.e9, 300., x).allFinite());
  }
}

TEST_CASE("Relaxed polynomial solutions leave unstable symmetric states",
          "[core][polynomial_solution]") {
  auto a = minerals::SLB_2011::forsterite();
  auto model = std::make_shared<solution_models::PolynomialSolution>(
      types::PairedEndmemberList{{*a, "[Mg]2SiO4"}, {*a, "[Mg]2SiO4"}},
      solution_models::PolynomialSolution::Interactions{
          {-4000., 4., 0., 0, 1, 1, 1}, {16000., 0., 0., 0, 2, 1, 2}});
  Solution base;
  base.set_solution_model(model);
  base.set_composition(Eigen::Array2d(.5, .5));
  RelaxedSolution s(base, Eigen::RowVector2d(1., -1.),
                    Eigen::RowVector2d(.5, .5));
  for (const double t : {1500., 500., 1500., 500.}) {
    s.set_state(1.e9, t);
    const auto x = s.get_molar_fractions();
    CHECK(std::abs(x[0] - x[1]) == Approx(t > 1000. ? 0. : .5).margin(2.e-6));
  }
}
