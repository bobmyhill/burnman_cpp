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
#include "burnman/tools/phase_diagram/domain.hpp"
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
  auto saddle_constraints = engine.state_constraints({1.08e9, 1000.});
  CHECK_FALSE(ps::detail::corrected_point(
      engine, *saddle.assemblage, saddle.ids, saddle_constraints,
      {1.08e9, 1000.}, ps::detail::equilibrium_pt_tolerance(), 1., true));
  auto result = engine.stable_at({1.08e9, 1000.}, &warm);
  REQUIRE(result.state.success);
  REQUIRE(result.ids.size() == 2);
  CHECK(engine.locally_stable(*result.assemblage));
  // At the same bulk composition, P and T, the common-tangent state must
  // lower G even when the energy gain is below the discovery tolerance.
  CHECK(result.assemblage->get_molar_gibbs() <
        saddle.assemblage->get_molar_gibbs() - 1.e-5);
  auto stable_constraints = engine.state_constraints({1.08e9, 1000.});
  REQUIRE(ps::detail::corrected_point(
      engine, *result.assemblage, result.ids, stable_constraints,
      {1.08e9, 1000.}, ps::detail::equilibrium_pt_tolerance(), 1., true));
  CHECK(result.state.mass_balance_error <= settings.mass_balance_tolerance);
  for (const auto &state : result.state.phases) {
    const double x = state.composition[0];
    CHECK_THAT(rt * std::log(x / (1. - x)) + (2. * rt + 80.) * (1. - 2. * x),
               Catch::Matchers::WithinAbs(0., 1.e-5));
  }
  CHECK((warm.state.phases[0].composition - Eigen::Vector2d(.5, .5)).norm() <
        1.e-12);
}

TEST_CASE("P-T correction rejects the saved pyrolite garnet saddle",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  // Positive-abundance garnet at the reported boundary 595 endpoint. Its
  // stationary composition passes the candidate-affinity tolerance but has a
  // negative composition-Hessian eigenvalue of approximately -7 J/mol.
  const double pressure = 19782184730.53887, temperature = 666.9954233210317;
  Eigen::VectorXd composition(7);
  composition << .2582836683484322, .21761423464074586, .4816869685608627,
      1.868074133110774e-7, .013004023288018411, 7.210725843903899e-5,
      .029338811096088473;
  auto garnet = minerals::SLB24::gt();
  garnet->set_composition(composition.array());
  ps::detail::Engine engine(garnet->get_formula(), {garnet}, {});
  ps::PhaseState phase;
  phase.id = phase.candidate_index = 0;
  phase.amount = 1.;
  phase.composition = composition;
  auto a =
      engine.make_assemblage({0}, {phase}, pressure, temperature, 1.e-7, true);
  auto c = engine.state_constraints({pressure, temperature});
  const auto solved = engine.solve(*a, c);
  INFO(solved.message);
  REQUIRE(solved.success);
  REQUIRE(engine.mass_error(*a) <= engine.settings.mass_balance_tolerance);
  REQUIRE(engine.stability(*a) >= -engine.settings.affinity_tolerance);
  REQUIRE_FALSE(engine.locally_stable(*a));
  CHECK_FALSE(ps::detail::corrected_point(
      engine, *a, {0}, c, {pressure, temperature},
      ps::detail::equilibrium_pt_tolerance(), 1., true));
}

TEST_CASE("Face reconstruction separates numerical zeros from positive traces",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  // Akimotoite where the saved S-P branch corrector became ill-conditioned.
  // The same composition reconstruction is needed in every section.
  ps::CompositionSection section;
  section.type = GENERATE(ps::DiagramType::PT, ps::DiagramType::PS,
                          ps::DiagramType::TV, ps::DiagramType::SV);
  auto phase = minerals::SLB24::il();
  Eigen::VectorXd x(5);
  x << .00013623712887933515, 1.3739024951054258e-13, .9992750956610574,
      3.775353408283415e-9, .0005886634345724743;
  phase->set_composition(x.array());
  const double pressure = 32326494944.69746, temperature = 487.564983404134;
  phase->set_state(pressure, temperature);
  ps::detail::Engine engine(phase->get_formula(), {phase}, {}, section);
  ps::PhaseState state;
  state.id = state.candidate_index = 0;
  state.amount = 1.;
  state.composition = x;
  auto a =
      engine.make_assemblage({0}, {state}, pressure, temperature, 1.e-7, true);
  REQUIRE(a->get_phase<Solution>(0));
  REQUIRE(a->get_phase<Solution>(0)->get_n_endmembers() == 4);
  const auto reconstructed = engine.snapshot(*a, {0})[0].composition;
  CHECK(reconstructed[1] == 0.);
  CHECK_THAT(reconstructed[3], Catch::Matchers::WithinAbs(x[3], 1.e-12));
  CHECK((reconstructed - x).norm() < 1.e-10);
  CHECK(engine.mass_error(*a) < 1.e-10);
  CHECK_THAT(a->get_molar_gibbs(),
             Catch::Matchers::WithinAbs(phase->get_molar_gibbs(), 1.e-5));
  CHECK(engine.locally_stable(*a));
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

TEST_CASE("Reaction continuation preserves zero-amount solution branches",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  auto phase = std::make_shared<Solution>();
  phase->set_solution_model(
      std::make_shared<solution_models::SymmetricRegularSolution>(
          types::PairedEndmemberList{{*minerals::HGP18::per(), "[Mg]O"},
                                     {*minerals::HGP18::wu(), "[Fe]O"}},
          std::vector<std::vector<double>>{{20000.}}));
  ps::detail::Engine engine({{"Mg", .5}, {"Fe", .5}, {"O", 1.}}, {phase}, {});
  ps::PhaseState reference;
  reference.id = reference.candidate_index = 0;
  reference.amount = 0.; // Its composition still identifies a reaction branch.
  reference.composition = Eigen::Vector2d(.83, .17);
  auto nearby = reference, switched = reference;
  nearby.composition = Eigen::Vector2d(.82, .18);
  switched.composition = Eigen::Vector2d(.17, .83);
  CHECK(ps::detail::same_solution_branches(engine, {reference}, {nearby}, 1.e9,
                                           1000.));
  CHECK_FALSE(ps::detail::same_solution_branches(engine, {reference},
                                                 {switched}, 1.e9, 1000.));
  // Above the solvus this large composition change is in one convex basin.
  CHECK(ps::detail::same_solution_branches(engine, {reference}, {switched},
                                           1.e9, 4000.));
  auto copy = switched;
  copy.id = 1;
  auto exchanged = reference;
  exchanged.id = 1;
  CHECK_FALSE(ps::detail::same_solution_branches(
      engine, {reference, copy}, {switched, exchanged}, 1.e9, 1000.));
  CHECK_FALSE(ps::detail::same_solution_branches(
      engine, {reference, copy}, {reference, exchanged}, 1.e9, 1000.));
  CHECK(ps::detail::same_solution_branches(engine, {reference}, {nearby, copy},
                                           1.e9, 1000.));
  auto ideal = std::make_shared<Solution>();
  ideal->set_solution_model(std::make_shared<solution_models::IdealSolution>(
      types::PairedEndmemberList{{*minerals::HGP18::per(), "[Mg]O"},
                                 {*minerals::HGP18::wu(), "[Fe]O"}}));
  ps::detail::Engine flat({{"Mg", .5}, {"Fe", .5}, {"O", 1.}}, {ideal}, {});
  CHECK(ps::detail::same_solution_branches(flat, {reference}, {switched}, 1.e9,
                                           0.));
}

TEST_CASE("Saved pyrolite reaction changes akimotoite composition branch",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::detail::Engine engine({{"Mg", 1.}, {"Si", 1.}, {"O", 3.}},
                            {minerals::SLB24::il()}, {});
  ps::PhaseState a;
  a.id = a.candidate_index = 0;
  a.amount = 0.;
  a.composition.resize(5);
  a.composition << .00112315767, .00000952227, .05655020975, .0003743600,
      .9419427503;
  a.composition /= a.composition.sum();
  auto b = a;
  b.composition << .0091880713, .0000742248, .9507114101, .0000532276,
      .0399730662;
  b.composition /= b.composition.sum();
  const bool ordinary_step = GENERATE(false, true);
  if (ordinary_step) {
    a.composition << .0070242274607, .00004670788049, .9430409836713,
        .00004117196224, .0498469090253;
    b.composition << .00075572977525, .00000509130025, .05351790308947,
        .00030253184767, .94541874398735;
  }
  CHECK_FALSE(ps::detail::same_solution_branches(
      engine, {a}, {b},
      ordinary_step ? 21.528567230528004e9 : 22.2191710868274e9,
      ordinary_step ? 757.2081344508965 : 794.5311732431134));
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

TEST_CASE("Shared composition faces retain independent limiting populations",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::PhaseState a, b;
  a.id = b.id = 0;
  a.candidate_index = b.candidate_index = 0;
  a.amount = .3;
  b.amount = .7;
  a.composition = Eigen::Vector3d(.6, .4, 0.);
  b.composition = Eigen::Vector3d(.6, 0., .4);
  Eigen::MatrixXd left(2, 3), right(2, 3);
  left << 1., 0., 0., 0., 1., 0.;
  right << 1., 0., 0., 0., 0., 1.;
  a.composition_basis = left;
  b.composition_basis = right;
  ps::detail::PhaseStates first{a}, second{b};
  REQUIRE(ps::detail::share_composition_faces(first, second));
  CHECK(first[0].composition == a.composition);
  CHECK(second[0].composition == b.composition);
  CHECK(first[0].amount == a.amount);
  CHECK(second[0].amount == b.amount);
  REQUIRE(first[0].composition_basis);
  REQUIRE(second[0].composition_basis);
  CHECK(first[0].composition_basis->rows() == 3);
  CHECK(*first[0].composition_basis == *second[0].composition_basis);
}

TEST_CASE("Compound reaction limiting states preserve positive trace sites",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  const double epsilon = 2.e-8;
  auto per = minerals::HGP18::per();
  auto first = std::make_shared<Mineral>(make_combined_mineral(
      {*per}, Eigen::ArrayXd::Ones(1),
      {constants::physics::gas_constant * 1000. * std::log(1. - epsilon), 0.,
       0.},
      "first"));
  auto second = std::make_shared<Mineral>(make_combined_mineral(
      {*first}, Eigen::ArrayXd::Ones(1), {10000., 10., 0.}, "second"));
  auto solution = std::make_shared<Solution>();
  solution->set_solution_model(
      std::make_shared<solution_models::SymmetricRegularSolution>(
          types::PairedEndmemberList{{*per, "[Mg]O"},
                                     {*minerals::HGP18::wu(), "[Fe]O"}},
          std::vector<std::vector<double>>{{0.}}));
  solution->set_composition(Eigen::Array2d(1. - epsilon, epsilon));
  ps::Settings settings;
  settings.max_phase_instances = 1;
  settings.active_solution_faces = true;
  ps::detail::Engine engine(
      {{"Mg", 1. - .5 * epsilon}, {"Fe", .5 * epsilon}, {"O", 1.}},
      {first, second, solution}, settings);
  ps::detail::PhaseStates states;
  for (int id = 0; id < 3; ++id) {
    ps::PhaseState p;
    p.id = p.candidate_index = id;
    p.amount = id ? .5 : 0.;
    p.composition =
        id == 2 ? Eigen::VectorXd(Eigen::Vector2d(1. - epsilon, epsilon))
                : Eigen::VectorXd::Ones(1);
    states.push_back(p);
  }
  auto a = engine.make_at({0, 1, 2}, states, {1.e9, 1000.});
  const auto prm = engine.parameters(*a);
  auto c = ps::detail::constraints(
      std::make_unique<equilibration::PhaseFractionConstraint>(0, 0., prm),
      std::make_unique<equilibration::PressureConstraint>(1.e9));
  auto solved = engine.solve(*a, c);
  REQUIRE(solved.success);
  std::array<ps::State, 2> limiting;
  auto sides = ps::detail::boundary_sides(engine, *a, {0, 1, 2}, 0, solved,
                                          {1.e9, 1000.}, {0., 1.}, &limiting);
  REQUIRE(!sides[0].empty());
  REQUIRE(!sides[1].empty());
  for (const auto &state : limiting) {
    REQUIRE(state.success);
    const auto trace = std::find_if(state.phases.begin(), state.phases.end(),
                                    [](const auto &p) { return p.id == 2; });
    REQUIRE(trace != state.phases.end());
    CHECK_THAT(trace->composition[1],
               Catch::Matchers::WithinAbs(epsilon, 1.e-12));
  }
}

TEST_CASE("Invariant reaction amounts reject an infeasible outgoing mixture",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  auto per = minerals::HGP18::per();
  auto polymorph = std::make_shared<Mineral>(make_combined_mineral(
      {*per}, Eigen::ArrayXd::Ones(1), {10000., 10., 0.}, "polymorph"));
  ps::Settings settings;
  settings.max_phase_instances = 1;
  ps::detail::Engine engine({{"Mg", .5}, {"Fe", .5}, {"O", 1.}},
                            {per, polymorph, minerals::HGP18::wu()}, settings);
  ps::detail::PhaseStates states;
  for (int id = 0; id < 3; ++id) {
    ps::PhaseState p;
    p.id = p.candidate_index = id;
    p.amount = id == 2 ? .5 : .25;
    p.composition = Eigen::VectorXd::Ones(1);
    states.push_back(p);
  }
  CHECK_FALSE(ps::detail::reaction_seed_amounts(engine, {0, 1, 2}, states, 2));
  auto feasible =
      ps::detail::reaction_seed_amounts(engine, {0, 1, 2}, states, 0);
  REQUIRE(feasible);
  CHECK_THAT((*feasible)[0].amount, Catch::Matchers::WithinAbs(0., 1.e-12));
  CHECK_THAT((*feasible)[1].amount, Catch::Matchers::WithinAbs(.5, 1.e-12));
  CHECK_THAT((*feasible)[2].amount, Catch::Matchers::WithinAbs(.5, 1.e-12));
  CHECK(engine.equilibrium_solves == 0);
  CHECK(engine.minimization_calls == 0);
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
  path.fixed_coordinate = ps::Coordinate::P;
  path.fixed_value = 1.e5;
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

TEST_CASE("Absent-field affinity orients an ill-conditioned pyrolite reaction",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  // Saved cold cpx/plagioclase coexistence. The full-field response loses an
  // amount direction near 1e-15; the absent branch still fixes the affinity
  // gradient, including its sign, without finite-offset discovery.
  ps::Settings settings;
  settings.active_solution_faces = true;
  settings.max_phase_instances = 3;
  types::FormulaMap bulk{
      {"Cr", 0.007500518941903751}, {"Si", 0.7489522445881484},
      {"O", 2.7663350706839376},    {"Mg", 0.9646647623082004},
      {"Ca", 0.0567076785900418},   {"Al", 0.07846169170147545},
      {"Na", 0.004194994559401659}, {"Fe", 0.11430278558091557}};
  ps::detail::Engine engine(bulk,
                            {minerals::SLB24::olivine(),
                             minerals::SLB24::clinopyroxene(),
                             minerals::SLB24::mg_fe_aluminous_spinel(),
                             minerals::SLB24::plagioclase(),
                             minerals::SLB24::ky(), minerals::SLB24::qtz()},
                            settings);
  ps::detail::PhaseStates states;

  {
    ps::PhaseState p;
    p.id = 0;
    p.candidate_index = 0;
    p.amount = 0.5066820363102009;
    p.composition = Eigen::VectorXd(2);
    p.composition << 0.8959615424397204, 0.10403845756027966;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 3;
    p.candidate_index = 1;
    p.amount = 1.7430302287154792e-23;
    p.composition = Eigen::VectorXd(6);
    p.composition << 0.0006333915320763106, 0.0, 0.0, 0.0, 0.9993310506808353,
        3.555778708842587e-05;
    Eigen::MatrixXd basis(3, 6);
    basis << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0;
    p.composition_basis = basis;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 4;
    p.candidate_index = 1;
    p.amount = 0.056743613248853374;
    p.composition = Eigen::VectorXd(6);
    p.composition << 0.9992329680184284, 0.00013360906129156191, 0.0, 0.0,
        0.0006334229202800191, 0.0;
    Eigen::MatrixXd basis(3, 6);
    basis << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0;
    p.composition_basis = basis;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 6;
    p.candidate_index = 2;
    p.amount = 0.003752276451856276;
    p.composition = Eigen::VectorXd(4);
    p.composition << -0.9916010338784302, 0.992112697399459,
        4.8340162984117374e-05, 0.9994399963159871;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 7;
    p.candidate_index = 2;
    p.amount = 0.0017144447057897603;
    p.composition = Eigen::VectorXd(4);
    p.composition << -3.651001787918773e-05, 3.651001787918773e-05,
        0.9999508250503569, 4.9174949643071564e-05;
    Eigen::MatrixXd basis(3, 4);
    basis << 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    p.composition_basis = basis;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 9;
    p.candidate_index = 3;
    p.amount = 0.004159059900590091;
    p.composition = Eigen::VectorXd(2);
    p.composition << 1.93466791820196e-06, 0.9999980653320818;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 12;
    p.candidate_index = 4;
    p.amount = 0.0371314206216559;
    p.composition = Eigen::VectorXd(1);
    p.composition << 1.0;
    states.push_back(p);
  }
  {
    ps::PhaseState p;
    p.id = 15;
    p.candidate_index = 5;
    p.amount = 0.07917438950321423;
    p.composition = Eigen::VectorXd(1);
    p.composition << 1.0;
    states.push_back(p);
  }

  const std::vector<int> ids{0, 3, 4, 6, 7, 9, 12, 15};
  auto a =
      engine.make_assemblage(ids, states, 88033576.68982078, 198.1957069055568);
  const auto prm = engine.parameters(*a);
  auto c = ps::detail::constraints(
      std::make_unique<equilibration::PhaseFractionConstraint>(1, 0., prm),
      std::make_unique<equilibration::TemperatureConstraint>(
          198.1957069055568));
  auto solved = engine.solve(*a, c);
  REQUIRE(solved.success);
  const int solves = engine.equilibrium_solves,
            minimizations = engine.minimization_calls;
  const std::vector<int> absent{0, 3, 6, 7, 9, 12, 15};
  auto sides = ps::detail::boundary_sides(engine, *a, ids, 3, solved,
                                          {25.e9, 2500.}, {1., 0.});
  CHECK(sides[0] == absent);
  CHECK(sides[1] == ids);
  auto reverse = ps::detail::boundary_sides(engine, *a, ids, 3, solved,
                                            {25.e9, 2500.}, {-1., 0.});
  CHECK(reverse[0] == ids);
  CHECK(reverse[1] == absent);
  CHECK(engine.equilibrium_solves == solves);
  CHECK(engine.minimization_calls == minimizations);
}

TEST_CASE("Calculation-frame reactions do not include nearby interior curves",
          "[phase_diagram]") {
  namespace ps = phase_diagrams;
  ps::Result result;
  result.pressure_range = {0., 25.e9};
  result.temperature_range = {0., 2500.};
  ps::Boundary line;
  ps::BoundaryPoint first, last;
  first.pressure = 8.6e9;
  last.pressure = 8.9e9;
  first.temperature = last.temperature = 2.5e-12;
  line.points = {first, last};
  CHECK(ps::detail::lies_on_calculation_frame(line, result));
  line.points[0].temperature = 1.e-7;
  CHECK_FALSE(ps::detail::lies_on_calculation_frame(line, result));
  line.points[1].temperature = 1.e-7;
  CHECK_FALSE(ps::detail::lies_on_calculation_frame(line, result));
}

TEST_CASE(
    "A homogeneous mixture excludes spurious solution copies in every section",
    "[phase_diagram]") {
  namespace ps = phase_diagrams;
  const auto type = GENERATE(ps::DiagramType::PT, ps::DiagramType::PS,
                             ps::DiagramType::TV, ps::DiagramType::SV);
  const double interaction = GENERATE(0., 40000.);
  auto phase = std::make_shared<Solution>();
  phase->set_solution_model(
      std::make_shared<solution_models::SymmetricRegularSolution>(
          types::PairedEndmemberList{{*minerals::HGP18::per(), "[Mg]O"},
                                     {*minerals::HGP18::wu(), "[Fe]O"}},
          std::vector<std::vector<double>>{{interaction}}));
  phase->set_composition(Eigen::Array2d(.5, .5));
  ps::Settings settings;
  settings.max_phase_instances = 2;
  ps::CompositionSection section;
  section.type = type;
  ps::detail::Engine engine({{"Mg", .5}, {"Fe", .5}, {"O", 1.}}, {phase},
                            settings, section);
  const Eigen::Vector2d first(.25, .75), second(.75, .25);
  CHECK(engine.distinct_compositions(0, first, second, 1.e9, 1000.) ==
        (interaction > 0.));
  ps::detail::PhaseStates states(2);
  for (int i = 0; i < 2; ++i) {
    states[static_cast<std::size_t>(i)].id = i;
    states[static_cast<std::size_t>(i)].candidate_index = 0;
    states[static_cast<std::size_t>(i)].amount = .5;
    states[static_cast<std::size_t>(i)].composition = i ? second : first;
  }
  auto a = engine.make_assemblage({0, 1}, states, 1.e9, 1000.);
  CHECK(engine.distinct(*a, {0, 1}) == (interaction > 0.));
  CHECK(engine.minimization_calls == 0);
  CHECK(engine.equilibrium_solves == 0);
}

TEST_CASE(
    "A univariant reaction splits at an invariant with other zero amounts",
    "[phase_diagram]") {
  namespace ps = phase_diagrams;
  auto first = minerals::HGP18::per();
  auto second = std::make_shared<Mineral>(make_combined_mineral(
      {*first}, Eigen::ArrayXd::Ones(1), {10000., 10., 0.}, "second"));
  auto third = std::make_shared<Mineral>(make_combined_mineral(
      {*first}, Eigen::ArrayXd::Ones(1), {11000., 10., -1.e-6}, "third"));
  ps::Settings settings;
  settings.max_phase_instances = 1;
  const types::FormulaMap bulk{{"Mg", 1.}, {"O", 1.}};
  ps::detail::Engine engine(bulk, {first, second, third}, settings);
  ps::Result result;
  result.composition_start = bulk;
  result.phase_names = {"first", "second", "third"};
  result.settings = settings;
  result.pressure_range = {.5e9, 1.5e9};
  result.temperature_range = {980., 1100.};
  auto point = [](double pressure, double temperature, bool junction) {
    ps::BoundaryPoint p;
    p.pressure = pressure;
    p.temperature = temperature;
    for (int id = 0; id < (junction ? 3 : 2); ++id) {
      ps::PhaseState phase;
      phase.id = phase.candidate_index = id;
      phase.composition = Eigen::VectorXd::Ones(1);
      phase.amount = (junction ? id == 0 : id == 1) ? 1. : 0.;
      p.phases.push_back(phase);
    }
    return p;
  };
  for (int id = 0; id < 4; ++id) {
    ps::Node node;
    node.id = id;
    node.kind = id == 2 ? "junction" : "domain_edge";
    node.gibbs_variance = id == 2 ? 0 : 1;
    node.pt_nullity = 0;
    node.assemblage = id == 2   ? std::vector<int>{0, 1, 2}
                      : id == 3 ? std::vector<int>{0, 2}
                                : std::vector<int>{0, 1};
    node.zero_phases =
        id == 2 ? std::vector<int>{1, 2} : std::vector<int>{id == 3 ? 2 : 0};
    node.pressure = id == 0 ? .5e9 : id == 1 ? 1.5e9 : id == 2 ? 1.e9 : 1.2e9;
    node.temperature = id == 3 ? 980. : 1000.;
    result.nodes.push_back(node);
  }
  ps::Boundary long_arc;
  long_arc.id = 0;
  long_arc.assemblage = {0, 1};
  long_arc.zero_phase = 0;
  long_arc.start_node = 0;
  long_arc.end_node = 1;
  long_arc.termination = "domain edge; domain edge";
  long_arc.points = {point(.5e9, 1000., false), point(1.5e9, 1000., false)};
  ps::Boundary arm;
  arm.id = 1;
  arm.assemblage = {0, 2};
  arm.zero_phase = 2;
  arm.start_node = 2;
  arm.end_node = 3;
  arm.termination = "junction; domain edge";
  arm.points = {point(1.e9, 1000., true), point(1.2e9, 980., true)};
  result.boundaries = {long_arc, arm};
  ps::detail::repair_field_boundaries(
      result, engine, [](ps::Boundary &) {},
      [](ps::Boundary &, std::size_t) { return false; });
  int split = 0;
  for (const auto &line : result.boundaries)
    if (line.assemblage == long_arc.assemblage) {
      CHECK((line.start_node == 2 || line.end_node == 2));
      ++split;
    }
  CHECK(split == 2);
}
