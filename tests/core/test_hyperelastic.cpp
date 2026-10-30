// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/core/hyperelastic.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/utils/exceptions.hpp"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>

using namespace burnman;
using Catch::Approx;

namespace {
HyperelasticMineral make_model() {
  auto scalar = minerals::SLB_2011::forsterite();
  Eigen::Vector3d lengths(4.7646, 10.2296, 5.9942);
  lengths *= std::cbrt(*scalar->params.V_0 / lengths.prod());
  utils::CellParameters cell;
  cell << lengths, 90., 90., 90.;
  types::Matrix6d compliance;
  compliance << .44, -.12, -.10, 0., 0., 0., -.12, .78, -.22, 0., 0., 0., -.10,
      -.22, .66, 0., 0., 0., 0., 0., 0., 1.97, 0., 0., 0., 0., 0., 0., 1.61, 0.,
      0., 0., 0., 0., 0., 1.55;
  types::NDArray<double> coefficients({6, 6, 2, 1});
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t j = 0; j < 6; ++j)
      coefficients({i, j, 1, 0}) = compliance(static_cast<Eigen::Index>(i),
                                              static_cast<Eigen::Index>(j));
  return HyperelasticMineral(
      std::make_shared<AnisotropicMineral>(scalar, cell, coefficients));
}
} // namespace

TEST_CASE("Hyperelastic hydrostatic limit and state validation",
          "[core][hyperelastic]") {
  auto m = make_model();
  CHECK_FALSE(m.has_state());
  CHECK_THROWS_AS(m.get_hencky_strain(), std::runtime_error);
  CHECK_THROWS_AS(m.set_state(1.e9, 1000.), exceptions::NotImplementedError);
  CHECK_THROWS_AS(m.set_state_with_volume(4.e-5, 1000.),
                  exceptions::NotImplementedError);
  const auto &hydro = m.get_hydrostatic();
  hydro->set_state(5.e9, 1000.);
  const Eigen::Matrix3d cell = hydro->get_cell_vectors();
  m.set_state_with_molar_cell_vectors(cell, 1000.);
  CHECK(m.get_hencky_strain().norm() < 1.e-14);
  CHECK(m.get_helmholtz_elastic() < 1.e-20);
  const Material &material = m;
  CHECK(material.get_pressure() == Approx(5.e9).margin(2.e4));
  CHECK(m.get_isothermal_stiffness_tensor().isApprox(
      hydro->get_isothermal_stiffness_tensor(), 2.e-6));
  CHECK(m.get_molar_isostress_heat_capacity() ==
        Approx(hydro->get_molar_heat_capacity_p()).epsilon(2.e-6));
  CHECK_THROWS_AS(m.set_output_frame("invalid"), std::invalid_argument);
  CHECK_THROWS_AS(m.set_state_with_molar_cell_vectors(-cell, 1000.),
                  std::invalid_argument);
  CHECK_THROWS_AS(m.set_state_with_molar_cell_vectors(cell, -1.),
                  std::invalid_argument);
  CHECK(m.get_current_cell_vectors().isApprox(cell));
}

TEST_CASE("Hyperelastic finite-strain energy, frames and independent copies",
          "[core][hyperelastic]") {
  auto m = make_model();
  m.get_hydrostatic()->set_state(5.e9, 1000.);
  Eigen::Matrix3d H;
  H << .01, .004, -.002, .004, -.006, .003, -.002, .003, -.004;
  const Eigen::Matrix3d cell =
      (H.exp() * m.get_hydrostatic()->get_unrotated_cell_vectors().transpose())
          .transpose();
  m.set_state_with_molar_cell_vectors(cell, 1000.);
  CHECK(m.get_hencky_strain().isApprox(H, 1.e-11));
  CHECK(m.get_molar_internal_energy() ==
        Approx(m.get_molar_helmholtz() + 1000. * m.get_molar_entropy()));
  const types::Matrix6d CT = m.get_isothermal_stiffness_tensor();
  CHECK((CT - CT.transpose()).norm() > 1.e7);
  CHECK((CT * m.get_isothermal_compliance_tensor())
            .isApprox(types::Matrix6d::Identity(), 1.e-12));
  const auto pi = utils::contract_stresses(m.get_thermal_stress_tensor());
  CHECK(m.get_isentropic_stiffness_tensor().isApprox(
      CT + m.get_molar_volume() * 1000. /
               m.get_molar_isometric_heat_capacity() * pi * pi.transpose(),
      1.e-12));

  const double energy = m.get_molar_helmholtz();
  const Eigen::Matrix3d stress = m.get_cauchy_stress_tensor();
  const Eigen::Matrix3d R =
      Eigen::AngleAxisd(.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d rotated_cell = cell * R.transpose();
  m.set_state_with_molar_cell_vectors(rotated_cell, 1000.);
  CHECK(m.get_molar_helmholtz() == Approx(energy));
  CHECK(
      m.get_cauchy_stress_tensor().isApprox(R * stress * R.transpose(), 1.e-7));
  m.set_output_frame(OutputFrame::crystal);
  CHECK(m.get_current_cell_vectors().isApprox(rotated_cell));
  const Eigen::Matrix3d Q = m.get_rotation_to_crystallographic_frame();
  CHECK(m.get_cauchy_stress_tensor().isApprox(
      Q * R * stress * R.transpose() * Q.transpose(), 1.e-7));
  const auto independent =
      std::dynamic_pointer_cast<HyperelasticMineral>(m.copy());
  REQUIRE(independent);
  CHECK(independent->get_hydrostatic() != m.get_hydrostatic());
  independent->set_state_with_molar_cell_vectors(cell, 900.);
  CHECK(m.get_temperature() == 1000.);
  CHECK(m.get_hydrostatic()->get_temperature() == 1000.);
  CHECK(m.get_current_cell_vectors().isApprox(rotated_cell));
}
