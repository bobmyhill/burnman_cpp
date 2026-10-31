// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/solution_model.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace burnman;
using namespace burnman::solution_models;
using Catch::Approx;

namespace {
types::PairedEndmemberList members() {
  const Mineral mineral;
  return {{mineral, "[Mg]3[Al]2Si3O12"},
          {mineral, "[Fe]3[Al]2Si3O12"},
          {mineral, "[Mg]3[Mg1/2Si1/2]2Si3O12"}};
}
} // namespace

TEST_CASE("Subregular binary coefficients and symmetric limit",
          "[core][solution_model][subregular_solution]") {
  auto binary_members = members();
  binary_members.pop_back();
  const SubregularSolution binary(binary_members, {{{10000., 20000.}}});
  for (double x : {0., .2, .7, 1.}) {
    const Eigen::Array2d p(x, 1. - x);
    CHECK(binary.compute_excess_gibbs_free_energy(0., 0., p) ==
          Approx(x * (1. - x) * ((1. - x) * 10000. + x * 20000.)));
  }
  CHECK(binary.compute_excess_partial_gibbs_free_energies(
            0., 0., Eigen::Array2d(1., 0.))[1] == 20000.);
  CHECK(binary.compute_excess_partial_gibbs_free_energies(
            0., 0., Eigen::Array2d(0., 1.))[0] == 10000.);

  const SubregularSolution subregular(
      members(), {{{10000., 10000.}, {5000., 5000.}}, {{-10000., -10000.}}},
      {{{1.e-6, 1.e-6}, {2.e-6, 2.e-6}}, {{-3.e-7, -3.e-7}}},
      {{{1., 1.}, {2., 2.}}, {{-.5, -.5}}});
  const SymmetricRegularSolution regular(
      members(), {{10000., 5000.}, {-10000.}}, {{1.e-6, 2.e-6}, {-3.e-7}},
      {{1., 2.}, {-.5}});
  const Eigen::Array3d p(.2, .3, .5);
  CHECK(subregular.compute_excess_partial_gibbs_free_energies(1.e9, 1200., p)
            .isApprox(regular.compute_excess_partial_gibbs_free_energies(
                1.e9, 1200., p)));
  CHECK(
      subregular.compute_excess_partial_entropies(1.e9, 1200., p)
          .isApprox(regular.compute_excess_partial_entropies(1.e9, 1200., p)));
  CHECK(subregular.compute_excess_partial_volumes(1.e9, 1200., p)
            .isApprox(regular.compute_excess_partial_volumes(1.e9, 1200., p)));
  CHECK(subregular.compute_gibbs_hessian(1.e9, 1200., p)
            .isApprox(regular.compute_gibbs_hessian(1.e9, 1200., p)));
  CHECK(subregular.compute_activities(1.e9, 1200., p)
            .isApprox(regular.compute_activities(1.e9, 1200., p)));
  CHECK_THROWS_AS(subregular.compute_activities(1.e9, 0., p),
                  std::domain_error);
}

TEST_CASE("Subregular partials and Hessians differentiate extensive potentials",
          "[core][solution_model][subregular_solution]") {
  const SubregularSolution model(
      members(), {{{10000., -10000.}, {5000., 3000.}}, {{-10000., 6000.}}},
      {{{1.e-6, -2.e-6}, {5.e-7, 1.e-6}}, {{-2.e-7, 7.e-7}}},
      {{{1., -2.}, {.3, 1.}}, {{-.2, .7}}}, {{0, 1, 2, 3000.}},
      {{0, 1, 2, 3.e-7}}, {{2, 0, 1, 1.2}});
  const Eigen::Array3d p(.2, .3, .5);
  const double P = 3.e9, T = 1200., step = 1.e-6;
  auto check_derivatives = [&](auto scalar, auto partial, auto hessian) {
    const Eigen::ArrayXd mu = (model.*partial)(P, T, p);
    const Eigen::MatrixXd h = (model.*hessian)(P, T, p);
    CHECK(h.isApprox(h.transpose()));
    CHECK((h * p.matrix()).norm() < 1.e-10 * h.norm());
    for (Eigen::Index i = 0; i < 3; ++i) {
      Eigen::ArrayXd plus = p, minus = p;
      plus[i] += step;
      minus[i] -= step;
      const double nplus = plus.sum(), nminus = minus.sum();
      plus /= nplus;
      minus /= nminus;
      CHECK((nplus * (model.*scalar)(P, T, plus) -
             nminus * (model.*scalar)(P, T, minus)) /
                (2. * step) ==
            Approx(mu[i]).epsilon(2.e-7));
      const Eigen::VectorXd derivative =
          ((model.*partial)(P, T, plus) - (model.*partial)(P, T, minus))
              .matrix() /
          (2. * step);
      CHECK(derivative.isApprox(h.col(i), 2.e-7));
    }
  };
  check_derivatives(&SolutionModel::compute_excess_gibbs_free_energy,
                    &SolutionModel::compute_excess_partial_gibbs_free_energies,
                    &SolutionModel::compute_gibbs_hessian);
  check_derivatives(&SolutionModel::compute_excess_entropy,
                    &SolutionModel::compute_excess_partial_entropies,
                    &SolutionModel::compute_entropy_hessian);
  check_derivatives(&SolutionModel::compute_excess_volume,
                    &SolutionModel::compute_excess_partial_volumes,
                    &SolutionModel::compute_volume_hessian);
  REQUIRE(std::dynamic_pointer_cast<SubregularSolution>(model.clone()));
  CHECK(model.compute_Cp_excess(P, T, p) == 0.);
  CHECK(model.compute_alphaV_excess(P, T, p) == 0.);
  CHECK(model.compute_VoverKT_excess(P, T, p) == 0.);
  CHECK_THROWS_AS(SubregularSolution(members(), {}), std::invalid_argument);
  CHECK_THROWS_AS(SubregularSolution(members(), {{{0., 0.}}, {{0., 0.}}}),
                  std::invalid_argument);
  CHECK_THROWS_AS(SubregularSolution(members(),
                                     {{{0., 0.}, {0., 0.}}, {{0., 0.}}}, {}, {},
                                     {{0., 1., 3., 1000.}}),
                  std::invalid_argument);
}
