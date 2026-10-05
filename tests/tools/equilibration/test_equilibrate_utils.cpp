/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "equilibration_fixtures.hpp"
#include "tolerances.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;
using namespace burnman;

TEST_CASE_METHOD(PartitioningAssemblageFixture, "equilibrate_utils",
                 "[tools][equilibration][equilibrate_utils]") {
  assemblage.set_n_moles(10.0);
  auto x = equilibration::get_parameter_vector(assemblage);
  Eigen::VectorXd expected(6);
  expected << 1.0e9, 2000.0, 4.0, 0.2, 6.0, 0.8;
  REQUIRE(x.isApprox(expected, tol_rel));
  Eigen::Vector4d amounts(3.2, 0.8, 1.2, 4.8);
  CHECK(equilibration::get_endmember_amounts(assemblage)
            .isApprox(amounts, tol_rel));
  x << 2.0e9, 1500.0, 2.0, 0.4, 3.0, 0.7;
  equilibration::set_composition_and_state_from_parameters(assemblage, x);
  CHECK_THAT(assemblage.get_n_moles(), WithinAbs(5.0, tol_abs));
  CHECK(equilibration::get_parameter_vector(assemblage).isApprox(x, tol_rel));
  amounts << 1.2, 0.8, 0.9, 2.1;
  CHECK(equilibration::get_endmember_amounts(assemblage)
            .isApprox(amounts, tol_rel));
}

TEST_CASE_METHOD(PartitioningAssemblageFixture,
                 "equilibration parameters and composition bounds",
                 "[tools][equilibration][equilibrate_utils]") {
  std::vector<equilibration::FreeVectorMap> free{
      {{"Mg", -1.0}, {"Fe", 1.0}, {"O", 0.0}}};
  auto prm =
      equilibration::get_equilibration_parameters(assemblage, bulk, free);
  CHECK(prm.n_parameters == 7);
  CHECK(prm.parameter_names.size() == 7);
  CHECK(prm.phase_amount_indices(0) == 2);
  CHECK(prm.phase_amount_indices(1) == 4);
  CHECK(prm.bulk_composition_vector.isApprox(Eigen::Vector3d(0.5, 0.5, 1.0),
                                             tol_rel));
  CHECK(prm.reduced_composition_vector.isApprox(Eigen::Vector2d(0.5, 0.5),
                                                tol_rel));
  CHECK(prm.reduced_free_composition_vectors.row(0).isApprox(
      Eigen::RowVector2d(-1.0, 1.0), tol_rel));
  auto x = equilibration::get_parameter_vector(assemblage, 1);
  CHECK((prm.constraint_matrix * x + prm.constraint_vector).maxCoeff() <=
        tol_abs);
  for (Eigen::Index i = 0; i < 6; ++i) {
    Eigen::VectorXd invalid = x;
    invalid(i) = -0.1;
    CAPTURE(i);
    CHECK((prm.constraint_matrix * invalid + prm.constraint_vector).maxCoeff() >
          0.0);
  }
  x(3) = 1.1; // The dependent first endmember must remain nonnegative too.
  CHECK((prm.constraint_matrix * x + prm.constraint_vector).maxCoeff() > 0.0);
  bulk["O"] = 2.0;
  CHECK_THROWS(
      equilibration::get_equilibration_parameters(assemblage, bulk, {}));
}
