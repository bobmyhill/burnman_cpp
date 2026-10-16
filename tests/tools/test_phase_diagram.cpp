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

#include "burnman/core/combined_mineral.hpp"
#include "burnman/core/solution.hpp"
#include "burnman/core/solution_models/symmetric_regular_solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/equilibration/equilibrate_utils.hpp"
#include "burnman/tools/phase_diagram.hpp"
#include "burnman/tools/phase_diagram/boundary_repair.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/refinement.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/constants.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
using namespace burnman;

TEST_CASE("Midpoint equilibrium tries both bracket ends before phase discovery",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  const auto type = GENERATE(ps::DiagramType::PT, ps::DiagramType::PS,
                             ps::DiagramType::TV, ps::DiagramType::SV);
  auto first = minerals::HGP18::per();
  auto hot = std::make_shared<Mineral>(make_combined_mineral(
      {*first}, Eigen::ArrayXd::Ones(1), {22500., 20., -.5e-6}, "hot"));
  auto state_at = [](Material &phase, double pressure, double temperature) {
    phase.set_state(pressure, temperature);
    ps::State state;
    state.pressure = pressure;
    state.temperature = temperature;
    state.entropy = phase.get_molar_entropy();
    state.volume = phase.get_molar_volume();
    return state;
  };
  const auto left_state = state_at(*first, 4.e9, 900.);
  const auto right_state = state_at(*hot, 8.e9, 1200.);
  const auto target = state_at(*hot, 7.e9, 1150.);
  ps::Settings settings;
  settings.max_phase_instances = 1;
  ps::CompositionSection section;
  section.type = type;
  ps::detail::Engine engine({{"Mg", 1.}, {"O", 1.}}, {first, hot}, settings,
                            section);
  ps::PhaseState initial;
  initial.amount = 1.;
  initial.composition = Eigen::VectorXd::Ones(1);
  initial.id = initial.candidate_index = 0;
  auto left = engine.fixed_at(
      {0}, {initial}, ps::diagram_coordinates(left_state, type),
      Eigen::Vector2d(left_state.pressure, left_state.temperature));
  initial.id = initial.candidate_index = 1;
  auto right = engine.fixed_at(
      {1}, {initial}, ps::diagram_coordinates(right_state, type),
      Eigen::Vector2d(right_state.pressure, right_state.temperature));
  REQUIRE(left.state.success);
  REQUIRE(right.state.success);
  const auto solves = engine.equilibrium_solves;
  // The first endpoint is unstable here. The other endpoint already contains
  // the stable phase, so no potential optimisation or cold phase search is
  // needed to establish the midpoint equilibrium.
  auto middle =
      engine.stable_at(ps::diagram_coordinates(target, type), &left, &right);
  REQUIRE(middle.state.success);
  CHECK(middle.ids == std::vector<int>{1});
  CHECK(engine.equilibrium_solves - solves == 2);
  CHECK(middle.state.mass_balance_error <= settings.mass_balance_tolerance);
  CHECK(middle.state.minimum_affinity >= -settings.affinity_tolerance);
  CHECK_THAT(middle.state.pressure,
             Catch::Matchers::WithinAbs(target.pressure, 1.));
  CHECK_THAT(middle.state.temperature,
             Catch::Matchers::WithinAbs(target.temperature, 1.e-6));
  CHECK_THAT(middle.state.entropy,
             Catch::Matchers::WithinAbs(target.entropy, 1.e-7));
  CHECK(left.ids == std::vector<int>{0});
  CHECK(right.ids == std::vector<int>{1});
  CHECK_THAT(left.state.entropy,
             Catch::Matchers::WithinAbs(left_state.entropy, 1.e-7));
  CHECK_THAT(right.state.entropy,
             Catch::Matchers::WithinAbs(right_state.entropy, 1.e-7));
}

TEST_CASE("P-T warm seeds reject a weak solution saddle before phase discovery",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  const double rt = constants::physics::gas_constant * 1000.;
  auto phase = std::make_shared<Solution>();
  phase->set_solution_model(
      std::make_shared<solution_models::SymmetricRegularSolution>(
          types::PairedEndmemberList{{*minerals::HGP18::per(), "[Mg]O"},
                                     {*minerals::HGP18::wu(), "[Fe]O"}},
          std::vector<std::vector<double>>{{2. * rt - 1000.}},
          std::vector<std::vector<double>>{{1.e-6}}));
  phase->set_composition(Eigen::Array2d(.5, .5));
  ps::Settings settings;
  ps::detail::Engine engine({{"Mg", .5}, {"Fe", .5}, {"O", 1.}}, {phase},
                            settings);
  auto warm = engine.stable_at({.9e9, 1000.});
  REQUIRE(warm.state.success);
  REQUIRE(warm.ids.size() == 1);
  auto saddle = engine.fixed_at(warm.ids, warm.state.phases, {1.08e9, 1000.});
  INFO(saddle.state.message);
  REQUIRE(saddle.state.success);
  CHECK(saddle.state.minimum_affinity >= -settings.affinity_tolerance);
  CHECK_FALSE(engine.locally_stable(*saddle.assemblage));
  auto result = engine.stable_at({1.08e9, 1000.}, &warm);
  REQUIRE(result.state.success);
  REQUIRE(result.ids.size() == 2);
  CHECK(engine.locally_stable(*result.assemblage));
  CHECK(result.state.mass_balance_error <= settings.mass_balance_tolerance);
  for (const auto &state : result.state.phases) {
    const double x = state.composition[0];
    CHECK_THAT(rt * std::log(x / (1. - x)) + (2. * rt + 80.) * (1. - 2. * x),
               Catch::Matchers::WithinAbs(0., 1.e-5));
  }
  CHECK((warm.state.phases[0].composition - Eigen::Vector2d(.5, .5)).norm() <
        1.e-12);
}

TEST_CASE("Inverse assemblage reconstruction retains positive trace sites",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  auto phase = std::make_shared<Solution>();
  phase->set_solution_model(std::make_shared<solution_models::IdealSolution>(
      types::PairedEndmemberList{{*minerals::HGP18::per(), "[Mg]O"},
                                 {*minerals::HGP18::wu(), "[Fe]O"}}));
  const double trace = 5.e-8;
  ps::CompositionSection section;
  section.type = ps::DiagramType::PS;
  ps::detail::Engine engine({{"Mg", 1. - trace}, {"Fe", trace}, {"O", 1.}},
                            {phase}, {}, section);
  ps::PhaseState state;
  state.id = state.candidate_index = 0;
  state.amount = 1.;
  state.composition = Eigen::Vector2d(1. - trace, trace);
  auto assembly =
      engine.make_at({0}, {state}, {1.e9, 100.}, Eigen::Vector2d(1.e9, 500.));
  REQUIRE(assembly->get_phase<Solution>(0));
  CHECK((engine.snapshot(*assembly, {0})[0].composition - state.composition)
            .norm() < 1.e-14);
  CHECK(engine.mass_error(*assembly) < 1.e-14);
  // A known constrained face can itself contain positive trace proportions.
  // Its allowed composition directions cannot be recovered from x alone.
  state.composition_basis = state.composition.transpose();
  assembly =
      engine.make_at({0}, {state}, {1.e9, 100.}, Eigen::Vector2d(1.e9, 500.));
  CHECK_FALSE(assembly->get_phase<Solution>(0));
  const auto saved = engine.snapshot(*assembly, {0})[0];
  REQUIRE(saved.composition_basis);
  CHECK((*saved.composition_basis - *state.composition_basis).norm() < 1.e-14);
  CHECK((saved.composition - state.composition).norm() < 1.e-14);
  CHECK(engine.mass_error(*assembly) < 1.e-14);
  // Exactly inactive sites can still use a reduced face.
  state.composition_basis.reset();
  state.composition = Eigen::Vector2d(1., 0.);
  assembly = engine.make_assemblage({0}, {state}, 1.e9, 500.);
  CHECK_FALSE(assembly->get_phase<Solution>(0));
  CHECK(engine.composition_basis(*assembly->get_phase(0), 0).rows() == 1);
}

TEST_CASE("Drawn interior boundaries are candidates without inventing nodes",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::Result result;
  result.pressure_range = result.temperature_range = {0., 1.};
  ps::FieldPolygon polygon;
  polygon.vertices.resize(5, 2);
  polygon.vertices << .2, .2, .8, .2, .8, .8, .2, .8, .2, .2;
  ps::Boundary boundary;
  boundary.id = boundary.zero_phase = 0;
  ps::BoundaryPoint a, b;
  a.pressure = 0.;
  b.pressure = 1.;
  a.temperature = b.temperature = .5;
  boundary.points = {a, b};
  result.boundaries = {boundary};
  CHECK(ps::detail::interior_boundaries(result, polygon) ==
        std::vector<int>{0});
  CHECK(result.nodes.empty());
  CHECK(result.boundaries[0].points.size() == 2);
  polygon.boundary_ids = {0};
  CHECK(ps::detail::interior_boundaries(result, polygon).empty());
  polygon.boundary_ids.clear();
  result.boundaries[0].points[0].temperature = .2;
  result.boundaries[0].points[1].temperature = .2;
  CHECK(ps::detail::interior_boundaries(result, polygon).empty());
  polygon.holes.push_back(polygon.vertices);
  result.boundaries[0] = boundary;
  CHECK(ps::detail::interior_boundaries(result, polygon).empty());
}

TEST_CASE("Intensive branch matching preserves distinct coexisting copies",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::PhaseState rich, poor, extra;
  rich.id = 3;
  poor.id = 4;
  extra.id = 5;
  rich.candidate_index = poor.candidate_index = extra.candidate_index = 1;
  rich.amount = poor.amount = 1.;
  rich.composition = Eigen::Vector2d(.9, .1);
  poor.composition = Eigen::Vector2d(.1, .9);
  extra.composition = Eigen::Vector2d(.5, .5);
  auto exchanged = std::vector<ps::PhaseState>{poor, rich, extra};
  exchanged[0].id = 3;
  exchanged[1].id = 4;
  CHECK(ps::detail::same_phase_compositions({rich, poor}, exchanged, 1.e-5,
                                            1.e-7));
  CHECK(ps::detail::same_phase_compositions(exchanged, {rich, poor}, 1.e-5,
                                            1.e-7));
  exchanged[2].amount = .01;
  CHECK_FALSE(ps::detail::same_phase_compositions({rich, poor}, exchanged,
                                                  1.e-5, 1.e-7));
  CHECK_FALSE(ps::detail::same_phase_compositions({rich, poor}, {rich, rich},
                                                  1.e-5, 1.e-7));
  exchanged[2].amount = 0.;
  CHECK(ps::detail::same_boundary_compositions({rich, poor}, 3, exchanged, 4,
                                               1.e-5, 1.e-7));
  CHECK_FALSE(ps::detail::same_boundary_compositions({rich, poor}, 3, exchanged,
                                                     3, 1.e-5, 1.e-7));
  CHECK_FALSE(ps::detail::same_boundary_compositions({rich, poor}, 5, exchanged,
                                                     4, 1.e-5, 1.e-7));
  CHECK_FALSE(ps::detail::same_boundary_compositions(
      {rich, extra}, 3, exchanged, 4, 1.e-5, 1.e-7));
  ps::Node node;
  node.zero_phases = {3, 4};
  exchanged[1].amount = 0.;
  CHECK(ps::detail::zero_phases_at(node, exchanged, 3) ==
        std::vector<int>{4, 5});
}

TEST_CASE("Boundary variance is independent of local phase-instance numbers",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::Result result;
  result.settings.max_phase_instances = 3;
  ps::Node node;
  node.id = 0;
  node.kind = "junction";
  node.gibbs_variance = 0;
  node.assemblage = {2, 5};
  result.nodes = {node};
  ps::Boundary line;
  line.start_node = 0;
  line.assemblage = {0};
  CHECK(ps::detail::boundary_variance(line, result) == 1);
  line.assemblage = {0, 3};
  CHECK(ps::detail::boundary_variance(line, result) == 0);
  for (auto unrelated : {std::vector<int>{0, 1}, std::vector<int>{6}}) {
    line.assemblage = unrelated;
    CHECK(ps::detail::boundary_variance(line, result) == -1);
  }
}

TEST_CASE("Reaction-extent seeds preserve present phase compositions",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::PhaseState phase, anchor;
  phase.id = phase.candidate_index = 1;
  anchor.id = anchor.candidate_index = 2;
  phase.amount = anchor.amount = 1.;
  phase.composition = Eigen::Vector2d(.8, .2);
  anchor.composition = Eigen::VectorXd::Ones(1);
  std::vector<ps::PhaseState> warm{phase, anchor}, seed = warm;
  seed[0].composition = Eigen::Vector2d(.2, .8);
  CHECK_FALSE(ps::detail::seed_phase_compositions(warm, seed, 1.e-5, 1.e-7));
  CHECK(warm[0].composition == phase.composition);
  warm[0].amount = 0.;
  CHECK(ps::detail::seed_phase_compositions(warm, seed, 1.e-5, 1.e-7));
  CHECK(warm[0].composition == seed[0].composition);
  CHECK(warm[0].amount == 0.);
}

TEST_CASE("Reaction entropy and volume orient singular P-T boundaries",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::Settings settings;
  settings.max_phase_instances = 1;
  ps::detail::Engine engine(
      {{"Al", 2.}, {"Si", 1.}, {"O", 5.}},
      {minerals::HP11::andalusite(), minerals::HP11::ky()}, settings);
  std::vector<ps::PhaseState> states(2);
  for (int k = 0; k < 2; ++k) {
    states[static_cast<std::size_t>(k)].id = k;
    states[static_cast<std::size_t>(k)].candidate_index = k;
    states[static_cast<std::size_t>(k)].composition = Eigen::VectorXd::Ones(1);
    states[static_cast<std::size_t>(k)].amount = k;
  }
  auto a = engine.make_at({0, 1}, states, {5.e8, 800.});
  auto prm = engine.parameters(*a);
  auto constraints = ps::detail::constraints(
      std::make_unique<equilibration::PhaseFractionConstraint>(0, 0., prm),
      std::make_unique<equilibration::TemperatureConstraint>(800.));
  auto solved = engine.solve(*a, constraints);
  REQUIRE(solved.success);
  const Eigen::Vector2d range(1.e9, 500.);
  std::array<ps::State, 2> limits;
  auto pressure = ps::detail::boundary_sides(engine, *a, {0, 1}, 0, solved,
                                             range, {1., 0.}, &limits);
  REQUIRE(pressure[0] == std::vector<int>{0});
  REQUIRE(pressure[1] == std::vector<int>{1}); // Denser kyanite at higher P.
  for (int side : {0, 1}) {
    const auto &state = limits[static_cast<std::size_t>(side)];
    REQUIRE(state.success);
    REQUIRE(state.phases.size() == 1);
    CHECK(state.phases[0].id == pressure[static_cast<std::size_t>(side)][0]);
    CHECK(state.phases[0].amount > 0.);
    CHECK(state.mass_balance_error < 1.e-12);
    CHECK(state.temperature == a->get_temperature());
  }
  auto temperature = ps::detail::boundary_sides(engine, *a, {0, 1}, 0, solved,
                                                range, {0., 1.});
  CHECK(temperature[0] == std::vector<int>{1});
  CHECK(temperature[1] == std::vector<int>{0}); // Higher-entropy andalusite.
  CHECK(engine.equilibrium_solves ==
        1); // Side identification performs no solves.
}

TEST_CASE("Entropy-volume sections solve physical P and T", "[phase_diagram]") {
  namespace ps = phase_diagrams;
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
  const double pressure_limit = GENERATE(8.e9, 8.2e9);
  auto result =
      ps::phase_diagram({{"Ca", 1.}, {"Si", 1.}, {"O", 3.}}, {mineral},
                        {0., pressure_limit}, {500., 1500.}, settings, section);
  CAPTURE(pressure_limit, result.diagnostics);
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
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  namespace hp = minerals::HGP18;
  ps::CompositionSection path;
  path.type = ps::DiagramType::TX;
  path.composition_end = {{"Fe", 2.}, {"O", 3.}};
  ps::Settings settings;
  settings.temperature_seeds = settings.composition_seeds = 5;
  settings.max_phase_instances = 1;
  auto result = ps::phase_diagram({{"Fe", 1.}},
                                  {hp::iron(), hp::wu(), hp::mt(), hp::hem()},
                                  {1.e5, 1.e5}, {300., 1500.}, settings, path);
  auto geometry = ps::field_polygons(result);
  CHECK(result.resolved == geometry.diagnostics.empty());
  for (const auto &diagnostic : geometry.diagnostics)
    CHECK(diagnostic.find("label is provisional") != std::string::npos);
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
          "[phase_diagram]") {
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

TEST_CASE("Native phase diagram resume retains calculation settings",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  namespace hp = minerals::HP11;
  types::FormulaMap bulk{{"Al", 2.}, {"Si", 1.}, {"O", 5.}};
  std::vector<std::shared_ptr<Material>> phases{hp::andalusite(), hp::ky(),
                                                hp::sill()};
  ps::Settings settings;
  settings.pressure_seeds = settings.temperature_seeds = 3;
  settings.max_phase_instances = 2;
  settings.exclude_invalid_eos = false;
  auto result =
      ps::phase_diagram(bulk, phases, {1.e5, 1.e9}, {500., 1200.}, settings);
  REQUIRE(result.resolved);
  CHECK(result.settings.max_phase_instances == 2);
  auto resumed = ps::refine_phase_diagram(bulk, phases, result);
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
  auto refined = ps::refine_phase_diagram(bulk, phases, result,
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
          "[phase_diagram]") {
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
