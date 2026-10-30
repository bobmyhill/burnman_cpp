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

#include "burnman/core/anisotropic_solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include <Eigen/LU>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace burnman;
using Catch::Approx;

namespace {
std::shared_ptr<AnisotropicMineral> endmember(bool iron) {
  auto mineral =
      iron ? minerals::SLB_2011::fayalite() : minerals::SLB_2011::forsterite();
  utils::CellParameters cell;
  cell << 1., 2., 1.4, 85., iron ? 82. : 80., 87.;
  cell.head<3>() *=
      std::cbrt(*mineral->params.V_0 /
                utils::cell_parameters_to_vectors(cell).determinant());
  types::NDArray<double> c({6, 6, 2, 1});
  types::Matrix6d block = types::Matrix6d::Zero();
  block.topLeftCorner<3, 3>() << .44, -.12, -.1, -.12, .78, -.22, -.1, -.22,
      .66;
  block(3, 3) = 1.97;
  block(4, 4) = 1.61;
  block(5, 5) = 1.55;
  block(0, 4) = block(4, 0) = iron ? .03 : .02;
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      c({i, j, 1, 0}) =
          block(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  return std::make_shared<AnisotropicMineral>(mineral, cell, c,
                                              utils::FrameConvention{0, 1, 2});
}

AnisotropicSolution solution(bool excess = false, bool composition = true) {
  auto a = endmember(false), b = endmember(true);
  auto model = std::make_shared<solution_models::SymmetricRegularSolution>(
      types::PairedEndmemberList{{*a, "[Mg]2SiO4"}, {*b, "[Fe]2SiO4"}},
      std::vector<std::vector<double>>{{10000.}},
      std::vector<std::vector<double>>{{1.e-6}});
  model->polymorphic_endmembers = {a, b};
  PsiExcessFunction fn =
      [excess](double lnV, double,
               const Eigen::ArrayXd &x) -> PsiExcessDerivatives {
    types::Matrix6d direction = types::Matrix6d::Zero();
    direction(0, 0) = 1.;
    direction(1, 1) = -1.;
    const double amplitude = excess ? .025 * (lnV - std::log(4.5e-5)) : 0.;
    CompositionTensors dp(2, Eigen::Matrix3d::Zero());
    dp[0].diagonal() << x[1] * amplitude, -x[1] * amplitude, 0.;
    dp[1].diagonal() << x[0] * amplitude, -x[0] * amplitude, 0.;
    return {x.prod() * amplitude * direction,
            (excess ? .025 : 0.) * x.prod() * direction,
            types::Matrix6d::Zero(), dp};
  };
  return AnisotropicSolution(
      "olivine", model, fn,
      composition ? std::optional<Eigen::ArrayXd>(Eigen::Array2d(.8, .2))
                  : std::nullopt);
}
} // namespace

TEST_CASE("Anisotropic solution scalar EOS and pure endmember limit",
          "[core][anisotropic_solution]") {
  auto s = solution();
  s.set_state(20.e9, 1000.);
  CHECK(s.get_molar_volume() ==
        Approx(s.get_scalar_solution()->get_molar_volume()));
  CHECK(s.get_cell_vectors().determinant() ==
        Approx(s.get_molar_volume()).epsilon(1.e-12));
  CHECK(s.get_isothermal_compliance_tensor().topLeftCorner<3, 3>().sum() ==
        Approx(s.get_isothermal_compressibility_reuss()));
  CHECK(s.get_thermal_expansivity_tensor().trace() ==
        Approx(s.get_thermal_expansivity()));
  CHECK(static_cast<Material *>(static_cast<Solution *>(&s)) ==
        static_cast<Material *>(static_cast<AnisotropicMineral *>(&s)));
  for (int i = 0; i < 2; ++i) {
    Eigen::Array2d x = Eigen::Array2d::Zero();
    x[i] = 1.;
    s.set_composition(x);
    const auto &m = s.get_endmembers()[static_cast<std::size_t>(i)];
    m->set_state(20.e9, 1000.);
    CHECK(s.get_cell_vectors().isApprox(m->get_cell_vectors(), 1.e-12));
    CHECK(s.get_isentropic_stiffness_tensor().isApprox(
        m->get_isentropic_stiffness_tensor(), 1.e-11));
  }
}

TEST_CASE("Anisotropic solution composition derivatives at fixed volume",
          "[core][anisotropic_solution]") {
  auto s = solution(true);
  s.set_state(20.e9, 1000.);
  const double volume = s.get_molar_volume(), step = 1.e-5;
  const Eigen::Array2d fractions(.8, .2);
  const Eigen::Matrix3d rotation = s.get_rotation_matrix(),
                        M = s.get_cell_vectors().transpose();
  const auto dn = s.get_dMdn_fixed_VT(), deps = s.get_depsdn_fixed_VT();
  for (const Eigen::Array2d &direction :
       {Eigen::Array2d(1., -1.), Eigen::Array2d(1., 0.)}) {
    Eigen::Matrix3d cells[2];
    for (int j = 0; j < 2; ++j) {
      const Eigen::Array2d amounts =
          fractions + (j == 0 ? step : -step) * direction;
      const double n = amounts.sum();
      s.set_composition(amounts / n);
      s.set_state_with_volume(volume / n, 1000.);
      cells[j] =
          rotation * s.get_unrotated_cell_vectors().transpose() * std::cbrt(n);
    }
    const Eigen::Matrix3d derivative = (cells[0] - cells[1]) / (2. * step);
    CHECK(derivative.isApprox(direction[0] * dn[0] + direction[1] * dn[1],
                              1.e-7));
    const Eigen::Matrix3d L = derivative * M.inverse();
    CHECK(
        (.5 * (L + L.transpose()))
            .isApprox(direction[0] * deps[0] + direction[1] * deps[1], 1.e-7));
  }
}

TEST_CASE("Anisotropic solution states, copies and validation",
          "[core][anisotropic_solution]") {
  auto s = solution(false, false);
  CHECK_THROWS_AS(s.set_state(1.e9, 300.), std::runtime_error);
  s.set_composition(Eigen::Array2d(.8, .2));
  s.set_state(20.e9, 1000.);
  const auto copy = std::dynamic_pointer_cast<AnisotropicSolution>(s.copy());
  REQUIRE(copy);
  CHECK(copy->get_endmembers()[0].get() != s.get_endmembers()[0].get());
  copy->set_composition(Eigen::Array2d(.3, .7));
  copy->set_state(1.e9, 500.);
  CHECK(s.get_pressure() == 20.e9);
  CHECK(s.get_molar_fractions().isApprox(Eigen::Array2d(.8, .2)));
  const auto stiffness = s.get_isentropic_stiffness_tensor();
  s.reset_cache();
  CHECK(s.get_isentropic_stiffness_tensor().isApprox(stiffness));
  CHECK_THROWS_AS(s.set_composition(Eigen::Array2d(.2, .2)),
                  std::invalid_argument);
  CHECK_THROWS_AS(s.set_composition(Eigen::Array2d(-.2, 1.2)),
                  std::invalid_argument);
  CHECK_THROWS_AS(s.set_state_with_volume(-1., 300.), std::invalid_argument);
}
