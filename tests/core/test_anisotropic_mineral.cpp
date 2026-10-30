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

#include "burnman/core/anisotropic_mineral.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/utils/exceptions.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace burnman;
using Catch::Approx;

namespace {
AnisotropicMineral isotropic_model() {
  const auto mineral = minerals::SLB_2011::periclase();
  const double a = std::cbrt(*mineral->params.V_0);
  const double s44 = *mineral->params.K_0 / *mineral->params.G_0;
  const double s11 = (1. + 3. * s44) / 9.;
  types::NDArray<double> coefficients({6, 6, 2, 1});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j)
      coefficients({i, j, 1, 0}) = i == j ? s11 : s11 - s44 / 2.;
    coefficients({i + 3, i + 3, 1, 0}) = s44;
  }
  return AnisotropicMineral(
      mineral, (utils::CellParameters() << a, a, a, 90., 90., 90.).finished(),
      coefficients);
}
} // namespace

TEST_CASE("Anisotropic mineral isotropic limit and shared material state",
          "[core][anisotropic_mineral]") {
  auto mineral = isotropic_model();
  CHECK_FALSE(mineral.has_state());
  CHECK_THROWS_AS(mineral.get_isothermal_stiffness_tensor(),
                  std::runtime_error);
  CHECK(static_cast<Material *>(static_cast<Mineral *>(&mineral)) ==
        static_cast<Material *>(static_cast<AnisotropicMaterial *>(&mineral)));
  for (double pressure : {1.e9, 20.e9}) {
    mineral.set_state(pressure, 1000.);
    const auto &reference = mineral.get_isotropic_mineral();
    CHECK(mineral.get_density() == Approx(reference->get_density()));
    CHECK(mineral.get_cell_vectors().determinant() ==
          Approx(mineral.get_molar_volume()));
    CHECK(mineral.get_thermal_expansivity_tensor().trace() ==
          Approx(mineral.get_thermal_expansivity()));
    CHECK(mineral.get_isothermal_compliance_tensor()
              .topLeftCorner<3, 3>()
              .sum() == Approx(mineral.get_isothermal_compressibility_reuss()));
    CHECK(mineral.get_isentropic_compliance_tensor()
              .topLeftCorner<3, 3>()
              .sum() == Approx(mineral.get_isentropic_compressibility_reuss()));
    CHECK(mineral.get_molar_isometric_heat_capacity() ==
          Approx(mineral.get_molar_heat_capacity_v()));
    CHECK(mineral.get_grueneisen_tensor().isApprox(
        reference->get_grueneisen_parameter() * Eigen::Matrix3d::Identity()));
    CHECK(mineral.get_isentropic_stiffness_tensor().isApprox(
        mineral.get_isentropic_compliance_tensor().inverse()));
    const AnisotropicMaterial &elastic = mineral;
    CHECK(elastic.get_isentropic_compliance_tensor().isApprox(
        mineral.get_isentropic_compliance_tensor()));
    const auto c = mineral.get_isothermal_stiffness_tensor();
    mineral.reset_cache();
    CHECK(mineral.get_isothermal_stiffness_tensor().isApprox(c));
  }
  CHECK_THROWS_AS(mineral.get_shear_modulus(), exceptions::NotImplementedError);
}

TEST_CASE("Anisotropic mineral derivative tensors and independent copies",
          "[core][anisotropic_mineral]") {
  auto mineral = isotropic_model();
  const double pressure = 2.e9, temperature = 1000., dp = 1.e4, dt = 0.01;
  mineral.set_state(pressure - dp, temperature);
  const auto fmP = mineral.get_deformation_gradient_tensor();
  mineral.set_state(pressure + dp, temperature);
  const auto fpP = mineral.get_deformation_gradient_tensor();
  mineral.set_state(pressure, temperature - dt);
  const auto fmT = mineral.get_deformation_gradient_tensor();
  mineral.set_state(pressure, temperature + dt);
  const auto fpT = mineral.get_deformation_gradient_tensor();
  mineral.set_state(pressure, temperature);
  const auto invF = mineral.get_deformation_gradient_tensor().inverse().eval();
  CHECK((-(fpP - fmP) / (2. * dp) * invF)
            .isApprox(mineral.get_isothermal_compressibility_tensor(), 1.e-7));
  CHECK(((fpT - fmT) / (2. * dt) * invF)
            .isApprox(mineral.get_thermal_expansivity_tensor(), 1.e-7));
  const auto copy =
      std::dynamic_pointer_cast<AnisotropicMineral>(mineral.copy());
  REQUIRE(copy);
  CHECK(copy->get_isotropic_mineral().get() !=
        mineral.get_isotropic_mineral().get());
  copy->set_state(10.e9, 500.);
  CHECK(mineral.get_pressure() == pressure);
  CHECK(mineral.get_isotropic_mineral()->get_pressure() == pressure);
  const auto volume = copy->get_molar_volume();
  mineral.set_state_with_volume(volume, 500.);
  CHECK(mineral.get_pressure() == Approx(10.e9));
  CHECK(mineral.get_cell_vectors().determinant() == Approx(volume));
}

TEST_CASE("Frechet derivative of the exponential for noncommuting matrices",
          "[core][anisotropic_mineral]") {
  const Eigen::Vector3d eigenvalues(-0.1, -0.2, 0.15);
  const Eigen::Matrix3d psi = eigenvalues.asDiagonal();
  Eigen::Matrix3d df;
  df << 0.2, 0.1, -0.04, 0.1, 0.3, 0.02, -0.04, 0.02, 0.5;
  const auto [F, dFdf, alpha, compliance] =
      deformation_gradient_alpha_and_compliance(2.e-5, 1.e-11, psi,
                                                types::Matrix6d::Identity(), df,
                                                Eigen::Matrix3d::Zero());
  const Eigen::Matrix3d expectedF =
      eigenvalues.array().exp().matrix().asDiagonal();
  CHECK(F.isApprox(expectedF, 1.e-14));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const double delta = eigenvalues[i] - eigenvalues[j];
      const double factor =
          i == j ? std::exp(eigenvalues[i])
                 : std::exp(eigenvalues[j]) * std::expm1(delta) / delta;
      CHECK(dFdf(i, j) == Approx(df(i, j) * factor).epsilon(1.e-13));
    }
  CHECK(alpha.allFinite());
  CHECK(compliance.allFinite());
  const Eigen::Matrix3d L = 1.e-11 * dFdf * F.inverse().transpose();
  const Eigen::Matrix3d expected_beta = 0.5 * (L + L.transpose());
  const auto full = utils::voigt_notation_to_compliance_tensor(compliance);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      double beta = 0.;
      for (std::size_t k = 0; k < 3; ++k)
        beta += full[i][j][k][k];
      CHECK(beta == Approx(expected_beta(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j)))
                        .epsilon(1.e-12));
    }
}

TEST_CASE("Unit cell conventions and coefficient validation",
          "[core][anisotropic_mineral]") {
  utils::CellParameters cell;
  cell << 1., 4.2, 2.2, 80., 85., 88.;
  utils::FrameConvention frame{0, 1, 2};
  do {
    const auto vectors = utils::cell_parameters_to_vectors(cell, frame);
    CHECK(utils::cell_vectors_to_parameters(vectors, frame).isApprox(cell));
  } while (std::next_permutation(frame.begin(), frame.end()));
  CHECK_THROWS_AS(utils::cell_parameters_to_vectors(cell, {0, 0, 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(AnisotropicMineral::check_standard_parameters(
                      types::NDArray<double>({6, 6, 1, 1})),
                  std::invalid_argument);
  auto mineral = isotropic_model();
  const auto reference = mineral.get_isotropic_mineral();
  const auto original_cell =
      utils::cell_vectors_to_parameters(mineral.get_cell_vectors_0());
  CHECK_THROWS_AS(AnisotropicMineral(
                      reference, original_cell,
                      [](double, double) -> PsiDerivatives {
                        return {types::Matrix6d::Identity(),
                                types::Matrix6d::Identity(),
                                types::Matrix6d::Zero()};
                      },
                      true),
                  std::invalid_argument);
}
