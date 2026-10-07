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

#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/tools/pseudosection.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using namespace burnman;

TEST_CASE("Entropy-volume sections solve physical P and T", "[pseudosection]") {
  namespace ps = pseudosections;
  auto mineral = minerals::SLB11::ca_perovskite();
  mineral->set_state(4.e9, 1000.);
  ps::CompositionSection section;
  section.type = ps::DiagramType::SV;
  section.entropy_range = {mineral->get_molar_entropy() - .05,
                           mineral->get_molar_entropy() + .05};
  section.volume_range = {mineral->get_molar_volume() - 1.e-9,
                          mineral->get_molar_volume() + 1.e-9};
  ps::Settings settings;
  settings.entropy_seeds = settings.volume_seeds = 2;
  auto result =
      ps::pseudosection({{"Ca", 1.}, {"Si", 1.}, {"O", 3.}}, {mineral},
                        {0., 8.e9}, {500., 1500.}, settings, section);
  REQUIRE(result.resolved);
  CHECK(result.coordinate_ranges()[0] == section.entropy_range);
  CHECK(result.coordinate_ranges()[1] == section.volume_range);
  for (const auto &sample : result.samples) {
    REQUIRE(sample.success);
    mineral->set_state(sample.pressure, sample.temperature);
    CHECK_THAT(sample.entropy,
               Catch::Matchers::WithinRel(mineral->get_molar_entropy(), 1.e-9));
    CHECK_THAT(sample.volume,
               Catch::Matchers::WithinRel(mineral->get_molar_volume(), 1.e-9));
    CHECK(sample.mass_balance_error < 1.e-8);
  }
  auto geometry = ps::field_polygons(result);
  REQUIRE(geometry.polygons.size() == 1);
  CHECK(geometry.polygons[0].n_phases == 1);
  CHECK_THAT(geometry.polygons[0].area, Catch::Matchers::WithinAbs(1., 1.e-9));
}

TEST_CASE("Fe-O TX closes fields with unequal composition endpoints",
          "[pseudosection]") {
  namespace ps = pseudosections;
  namespace hp = minerals::HGP18;
  ps::CompositionSection path;
  path.type = ps::DiagramType::TX;
  path.composition_end = {{"Fe", 2.}, {"O", 3.}};
  ps::Settings settings;
  settings.temperature_seeds = settings.composition_seeds = 5;
  settings.max_phase_instances = 1;
  auto result = ps::pseudosection({{"Fe", 1.}},
                                  {hp::iron(), hp::wu(), hp::mt(), hp::hem()},
                                  {1.e5, 1.e5}, {300., 1500.}, settings, path);
  REQUIRE(result.resolved);
  auto geometry = ps::field_polygons(result);
  REQUIRE(geometry.polygons.size() == 4);
  double area = 0.;
  for (const auto &field : geometry.polygons) {
    CHECK(field.n_phases == 2);
    CHECK_FALSE(field.has_open_boundary);
    area += field.area;
  }
  CHECK_THAT(area, Catch::Matchers::WithinAbs(1., 1.e-8));
  int invariants = 0;
  for (const auto &node : result.nodes)
    if (node.kind == "junction") {
      CHECK_THAT(node.temperature,
                 Catch::Matchers::WithinAbs(829.2029337, .01));
      ++invariants;
    }
  CHECK(invariants > 0);
}

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
  namespace hp = minerals::HP11;
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
  auto refined = ps::refine_pseudosection(bulk, phases, result,
                                          ps::LineResolution{{21, 17}});
  REQUIRE(refined.resolved);
  REQUIRE(refined.boundaries.size() == 3);
  for (const auto &line : refined.boundaries)
    for (std::size_t i = 1; i < line.points.size(); ++i) {
      CHECK(std::abs(line.points[i].pressure - line.points[i - 1].pressure) <=
            (result.pressure_range[1] - result.pressure_range[0]) / 20.);
      CHECK(std::abs(line.points[i].temperature -
                     line.points[i - 1].temperature) <=
            (result.temperature_range[1] - result.temperature_range[0]) / 16.);
      CHECK(line.points[i].mass_balance_error < 1.e-8);
      CHECK(line.points[i].minimum_affinity >= -settings.affinity_tolerance);
    }
}

TEST_CASE("Invalid continuation phase amounts are recoverable",
          "[pseudosection]") {
  Assemblage assemblage;
  assemblage.add_phases({minerals::HP11::andalusite()});
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
