// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/relaxed_anisotropic_solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include <Eigen/Eigenvalues>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace burnman;
using Catch::Approx;

namespace {
std::shared_ptr<AnisotropicSolution> unrelaxed() {
  std::vector<std::shared_ptr<Mineral>> members;
  for (int i = 0; i < 2; ++i) {
    auto scalar = minerals::SLB_2011::forsterite();
    *scalar->params.F_0 += 1500. * i;
    *scalar->params.V_0 *= 1. + .025 * i;
    utils::CellParameters cell;
    cell << 1. + .03 * i, 2., 1.4, 85., 80., 87.;
    cell.head<3>() *=
        std::cbrt(*scalar->params.V_0 /
                  utils::cell_parameters_to_vectors(cell).determinant());
    types::NDArray<double> c({6, 6, 2, 1});
    types::Matrix6d block = types::Matrix6d::Zero();
    block.topLeftCorner<3, 3>() << .44, -.12, -.1, -.12, .78, -.22, -.1, -.22,
        .66;
    block(3, 3) = 1.97;
    block(4, 4) = 1.61;
    block(5, 5) = 1.55;
    block(0, 4) = block(4, 0) = .02 + .01 * i;
    for (std::size_t j = 0; j < 6; ++j)
      for (std::size_t k = 0; k < 6; ++k)
        c({j, k, 1, 0}) =
            block(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(k));
    members.push_back(std::make_shared<AnisotropicMineral>(
        scalar, cell, c, utils::FrameConvention{0, 1, 2}));
  }
  auto model = std::make_shared<solution_models::SymmetricRegularSolution>(
      types::PairedEndmemberList{{*members[0], "[Mg]2SiO4"},
                                 {*members[1], "[Mg]2SiO4"}},
      std::vector<std::vector<double>>{{-100000.}});
  model->polymorphic_endmembers = members;
  return std::make_shared<AnisotropicSolution>(
      "structural solution", model,
      [](double, double, const Eigen::ArrayXd &) -> PsiExcessDerivatives {
        return {types::Matrix6d::Zero(), types::Matrix6d::Zero(),
                types::Matrix6d::Zero(),
                CompositionTensors(2, Eigen::Matrix3d::Zero())};
      });
}
Eigen::MatrixXd Q() { return Eigen::RowVector2d(1., -1.); }
Eigen::MatrixXd X()
{
  return Eigen::RowVector2d(.5, .5);
}
} // namespace

TEST_CASE("Relaxed anisotropic equilibrium and thermoelastic Hessian",
          "[core][relaxed_anisotropic_solution]") {
  auto source = unrelaxed();
  RelaxedAnisotropicSolution s(source, Q(), X());
  s.set_composition(Eigen::ArrayXd::Ones(1));
  s.set_state(20.e9, 1000.);
  const auto &scalar = *source->get_scalar_solution();
  const auto &endmembers = scalar.get_solution_model()->endmembers;
  const double x =
      .5 + (endmembers[1].get_molar_gibbs() - endmembers[0].get_molar_gibbs()) /
               200000.;
  CHECK(s.get_molar_fractions()[0] == Approx(x).margin(1.e-11));
  CHECK((Q() * s.get_partial_gibbs().matrix()).norm() < 1.e-6);
  CHECK(s.get_cell_vectors().determinant() ==
        Approx(s.get_molar_volume()).epsilon(1.e-12));
  CHECK(s.get_relaxed_hessian().isApprox(s.get_frozen_hessian() +
                                             s.get_coupling().transpose() *
                                                 s.get_dqdz_relaxed(),
                                         1.e-13));
  CHECK((s.get_isothermal_stiffness_tensor() *
         s.get_isothermal_compliance_tensor())
            .isApprox(types::Matrix6d::Identity(), 1.e-12));
  Eigen::SelfAdjointEigenSolver<types::Matrix6d> softening(
      source->get_isothermal_stiffness_tensor() -
      s.get_isothermal_stiffness_tensor());
  CHECK(softening.eigenvalues().minCoeff() > -1.e-3);
  const double v = s.get_molar_volume(),
               kt = s.get_isothermal_bulk_modulus_reuss(),
               cp = s.get_molar_heat_capacity_p(),
               alpha = s.get_thermal_expansivity();
  s.set_state(20.e9 - 1.e5, 1000.);
  const double vm = s.get_molar_volume();
  s.set_state(20.e9 + 1.e5, 1000.);
  CHECK(-(s.get_molar_volume() - vm) / 2.e5 / v ==
        Approx(1. / kt).epsilon(1.e-7));
  s.set_state(20.e9, 999.99);
  const double vt = s.get_molar_volume(), entropy = s.get_molar_entropy();
  s.set_state(20.e9, 1000.01);
  CHECK((s.get_molar_volume() - vt) / .02 / v == Approx(alpha).epsilon(1.e-7));
  CHECK(1000. * (s.get_molar_entropy() - entropy) / .02 ==
        Approx(cp).epsilon(1.e-7));
  s.set_state_with_volume(v, 1000.);
  CHECK(s.get_pressure() == Approx(20.e9).epsilon(1.e-10));
  CHECK((Q() * s.get_partial_gibbs().matrix()).norm() < 1.e-6);
}

TEST_CASE("Relaxed anisotropic initialization, frozen states and copies",
          "[core][relaxed_anisotropic_solution]") {
  auto source = unrelaxed();
  RelaxedAnisotropicSolution s(source, Q(), X());
  CHECK_THROWS_AS(s.set_state(1.e9, 300.), std::runtime_error);
  CHECK_THROWS_AS(s.set_composition(Eigen::Array2d(.8, .2)),
                  std::invalid_argument);
  s.set_composition(Eigen::ArrayXd::Ones(1), Eigen::VectorXd::Constant(1, .05),
                    false);
  s.set_state(10.e9, 800., false);
  const auto stiffness = s.get_isothermal_stiffness_tensor();
  auto copy = std::dynamic_pointer_cast<RelaxedAnisotropicSolution>(s.copy());
  REQUIRE(copy);
  CHECK(copy->get_unrelaxed().get() != source.get());
  copy->set_state(20.e9, 1000.);
  CHECK(s.get_molar_fractions().isApprox(Eigen::Array2d(.55, .45)));
  CHECK(s.get_unrelaxed()->get_pressure() == 10.e9);
  s.reset_cache();
  CHECK(s.get_isothermal_stiffness_tensor().isApprox(stiffness));
  RelaxedAnisotropicSolution initialized(source, Q(), X());
  CHECK((Q() * initialized.get_partial_gibbs().matrix()).norm() < 1.e-6);
  CHECK_THROWS_AS(s.set_state_with_volume(-1., 300.), std::invalid_argument);
  CHECK_THROWS_AS(RelaxedAnisotropicSolution(source, Q(), Q()),
                  std::invalid_argument);
  CHECK_THROWS_AS(RelaxedAnisotropicSolution(nullptr, Q(), X()),
                  std::invalid_argument);
}
