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

#include "burnman/core/anisotropy.hpp"
#include <Eigen/Geometry>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

using namespace burnman;
using Catch::Approx;

TEST_CASE("Isotropic elastic limits", "[core][anisotropy]") {
  const double lambda = 40.e9, mu = 24.e9, rho = 2735.;
  const IsotropicMaterial material(rho, {lambda, mu});
  const double bulk = lambda + 2. * mu / 3.;
  const double young = mu * (3. * lambda + 2. * mu) / (lambda + mu);
  const double poisson = lambda / (2. * (lambda + mu));
  CHECK(material.get_density() == rho);
  CHECK(material.get_isentropic_bulk_modulus_voigt() == Approx(bulk));
  CHECK(material.get_isentropic_bulk_modulus_reuss() == Approx(bulk));
  CHECK(material.get_isentropic_bulk_modulus_vrh() == Approx(bulk));
  CHECK(material.get_isentropic_shear_modulus_voigt() == Approx(mu));
  CHECK(material.get_isentropic_shear_modulus_reuss() == Approx(mu));
  CHECK(material.get_isentropic_shear_modulus_vrh() == Approx(mu));
  CHECK(std::abs(material.get_isentropic_universal_elastic_anisotropy()) <
        1.e-13);
  CHECK(material.get_isentropic_isotropic_poisson_ratio() == Approx(poisson));
  for (const Eigen::Vector3d &n :
       {Eigen::Vector3d(1., 0., 0.), Eigen::Vector3d(-1., 2., 3.)}) {
    const Eigen::Vector3d lateral = n.unitOrthogonal();
    CHECK(material.isentropic_linear_compressibility(n) ==
          Approx(1. / (3. * bulk)));
    CHECK(material.isentropic_youngs_modulus(n) == Approx(young));
    CHECK(material.isentropic_shear_modulus(n, lateral) == Approx(mu));
    CHECK(material.isentropic_poissons_ratio(n, lateral) == Approx(poisson));
    const auto [speeds, polarizations] = material.wave_velocities(n);
    CHECK(speeds[0] == Approx(std::sqrt((lambda + 2. * mu) / rho)));
    CHECK(speeds[1] == Approx(std::sqrt(mu / rho)));
    CHECK(speeds[2] == Approx(speeds[1]));
    CHECK(std::abs(polarizations.col(0).dot(n.normalized())) == Approx(1.));
  }
}

TEST_CASE("Anisotropic tensor inverses and acoustic eigenvectors",
          "[core][anisotropy]") {
  const OrthorhombicMaterial material(
      3300.,
      {320.e9, 68.e9, 72.e9, 201.e9, 77.e9, 233.e9, 64.e9, 78.e9, 80.e9});
  const auto c = material.get_full_isentropic_stiffness_tensor();
  const auto s = material.get_full_isentropic_compliance_tensor();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l) {
          double product = 0.;
          for (std::size_t m = 0; m < 3; ++m)
            for (std::size_t n = 0; n < 3; ++n)
              product += c[i][j][m][n] * s[m][n][k][l];
          const double identity =
              0.5 * ((i == k && j == l) + (i == l && j == k));
          CHECK(std::abs(product - identity) < 1.e-14);
        }
  const Eigen::Vector3d n(1., 2., -3.);
  const auto q = material.christoffel_tensor(n);
  const auto [speeds, polarizations] = material.wave_velocities(n);
  CHECK(speeds[0] >= speeds[1]);
  CHECK(speeds[1] >= speeds[2]);
  CHECK((polarizations.transpose() * polarizations)
            .isApprox(Eigen::Matrix3d::Identity()));
  CHECK((q * polarizations)
            .isApprox(polarizations *
                      (speeds.array().square() * material.get_density())
                          .matrix()
                          .asDiagonal()));
  const auto axis_q = material.christoffel_tensor(Eigen::Vector3d::UnitX());
  CHECK(axis_q.diagonal().isApprox(Eigen::Vector3d(320.e9, 80.e9, 78.e9)));
  CHECK(material.isentropic_shear_modulus(Eigen::Vector3d::UnitX(),
                                          Eigen::Vector3d::UnitY()) ==
        Approx(80.e9));
}

TEST_CASE("Nonhydrostatic acoustic correction and compliance cache reset",
          "[core][anisotropy]") {
  class StatefulMaterial : public AnisotropicMaterial {
  public:
    StatefulMaterial()
        : AnisotropicMaterial(3000., types::Matrix6d::Identity()) {}
    types::Matrix6d stiffness = types::Matrix6d::Identity();
    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
    const types::Matrix6d &get_isentropic_stiffness_tensor() const override {
      return stiffness;
    }
    std::optional<Eigen::Matrix3d> get_cauchy_stress() const override {
      return stress;
    }
  } material;
  CHECK(material.get_isentropic_bulk_modulus_reuss() == Approx(1. / 3.));
  material.stiffness *= 2.;
  material.set_state(1.e9, 300.);
  CHECK(material.get_isentropic_bulk_modulus_reuss() == Approx(2. / 3.));
  const auto baseline = material.christoffel_tensor(Eigen::Vector3d::UnitX());
  material.stress.diagonal() << 1., 3., 5.;
  Eigen::Matrix3d correction = Eigen::Matrix3d::Zero();
  correction.diagonal() << 0., -1., -2.;
  CHECK(material.christoffel_tensor(Eigen::Vector3d::UnitX())
            .isApprox(baseline + correction));
  material.stress = -10. * Eigen::Matrix3d::Identity();
  CHECK(material.christoffel_tensor(Eigen::Vector3d(5., 0., 0.))
            .isApprox(baseline));
}

TEST_CASE("Crystal symmetry constraints and copies", "[core][anisotropy]") {
  const TetragonalMaterial tetragonal(3000., {1., 2., 3., 4., 5., 6., 7.});
  const auto &t = tetragonal.get_isentropic_stiffness_tensor();
  CHECK(t(0, 5) == 4.);
  CHECK(t(1, 5) == -4.);
  const RhombohedralMaterial rhombohedral(3000.,
                                          {1., 2., 3., 4., 5., 6., 7., 8.});
  const auto &r = rhombohedral.get_isentropic_stiffness_tensor();
  CHECK(r(0, 3) == 4.);
  CHECK(r(1, 3) == -4.);
  CHECK(r(4, 5) == 4.);
  CHECK(r(0, 4) == 5.);
  CHECK(r(1, 4) == -5.);
  CHECK(r(3, 5) == -5.);
  auto copy = rhombohedral.copy();
  REQUIRE(dynamic_cast<RhombohedralMaterial *>(copy.get()));
  CHECK(copy->get_density() == 3000.);
  copy->set_name("copy");
  CHECK(copy->get_name() != rhombohedral.get_name());
}

TEST_CASE("Invalid elastic inputs", "[core][anisotropy]") {
  CHECK_THROWS_AS(IsotropicMaterial(0., {1., 1.}), std::invalid_argument);
  CHECK_THROWS_AS(CubicMaterial(3000., {1., 1.}), std::invalid_argument);
  CHECK_THROWS_AS(HexagonalMaterial(3000., {}), std::invalid_argument);
  CHECK_THROWS_AS(TetragonalMaterial(3000., {}), std::invalid_argument);
  CHECK_THROWS_AS(RhombohedralMaterial(3000., {}), std::invalid_argument);
  CHECK_THROWS_AS(OrthorhombicMaterial(3000., {}), std::invalid_argument);
  CHECK_THROWS_AS(MonoclinicMaterial(3000., {}), std::invalid_argument);
  CHECK_THROWS_AS(TriclinicMaterial(3000., {}), std::invalid_argument);
  types::Matrix6d c = types::Matrix6d::Identity();
  c(0, 1) = 1.;
  CHECK_THROWS_AS(AnisotropicMaterial(3000., c), std::invalid_argument);
  const AnisotropicMaterial singular(3000., types::Matrix6d::Zero());
  CHECK_THROWS_AS(singular.get_isentropic_compliance_tensor(),
                  std::runtime_error);
  const IsotropicMaterial material(3000., {1., 1.});
  CHECK_THROWS_AS(material.isentropic_shear_modulus(Eigen::Vector3d::UnitX(),
                                                    Eigen::Vector3d::UnitX()),
                  std::invalid_argument);
  CHECK_THROWS_AS(material.christoffel_tensor(Eigen::Vector3d(
                      std::numeric_limits<double>::quiet_NaN(), 0., 0.)),
                  std::invalid_argument);
}
