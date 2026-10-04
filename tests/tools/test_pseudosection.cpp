#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/tools/pseudosection.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using namespace burnman;
TEST_CASE("Native Gibbs LP dual amounts conserve components",
          "[pseudosection]") {
  Eigen::MatrixXd a(3, 2);
  a << 1, 0, 0, 1, 1, 1;
  Eigen::VectorXd g(3), b(2);
  g << 3, 5, 6;
  b << 2, 1;
  auto r = polytope::gibbs_linear_program(a, g, b);
  CHECK_THAT(r.gibbs, Catch::Matchers::WithinAbs(9., 1.e-12));
  CHECK((a.transpose() * r.amounts - b).norm() < 1.e-12);
  CHECK((g - a * r.chemical_potentials).minCoeff() > -1.e-12);
}

TEST_CASE("Native pseudosection resume retains calculation settings",
          "[pseudosection]") {
  namespace ps = pseudosections;
  namespace hp = minerals::HP_2011_ds62;
  types::FormulaMap bulk{{"Al", 2.}, {"Si", 1.}, {"O", 5.}};
  std::vector<std::shared_ptr<Material>> phases{hp::andalusite(), hp::ky(),
                                                hp::sill()};
  ps::Settings settings;
  settings.pressure_seeds = settings.temperature_seeds = 3;
  settings.max_phase_instances = 2;
  settings.exclude_invalid_eos = false;
  auto result =
      ps::pseudosection(bulk, phases, {1.e5, 1.e9}, {500., 1200.}, settings);
  REQUIRE(result.resolved);
  CHECK(result.settings.max_phase_instances == 2);
  auto resumed = ps::refine_pseudosection(bulk, phases, result);
  CHECK(resumed.resolved);
  CHECK(resumed.settings.max_phase_instances == 2);
  CHECK_FALSE(resumed.settings.exclude_invalid_eos);
  CHECK(resumed.boundaries.size() == 3);
  auto geometry = ps::field_polygons(resumed);
  REQUIRE(geometry.polygons.size() == 3);
  double area = 0.;
  for (auto &field : geometry.polygons) {
    CHECK(field.n_phases == 1);
    CHECK_FALSE(field.has_open_boundary);
    area += field.area;
  }
  CHECK_THAT(area, Catch::Matchers::WithinAbs(1., 1.e-12));
}

TEST_CASE("Invalid continuation phase amounts are recoverable",
          "[pseudosection]") {
  Assemblage assemblage;
  assemblage.add_phases({minerals::HP_2011_ds62::andalusite()});
  assemblage.set_fractions(Eigen::ArrayXd::Ones(1));
  assemblage.set_n_moles(1.);
  assemblage.set_state(1.e9, 800.);
  auto parameters = equilibration::get_parameter_vector(assemblage, 0);
  parameters[2] = -.1;
  CHECK_THROWS_AS(equilibration::set_composition_and_state_from_parameters(
                      assemblage, parameters),
                  std::invalid_argument);
  parameters[2] = 0.;
  CHECK_THROWS_AS(equilibration::set_composition_and_state_from_parameters(
                      assemblage, parameters),
                  std::invalid_argument);
  parameters[2] = 1.;
  CHECK_NOTHROW(equilibration::set_composition_and_state_from_parameters(
      assemblage, parameters));
  CHECK(assemblage.get_n_moles() == 1.);
}
