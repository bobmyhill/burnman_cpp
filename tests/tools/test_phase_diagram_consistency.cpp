// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#include "burnman/core/combined_mineral.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/tools/phase_diagram/consistency.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

using namespace burnman::phase_diagrams;
using namespace burnman::phase_diagrams::detail;

TEST_CASE("Natural potentials obey tangent bounds across phase changes",
          "[phase_diagram][consistency]") {
  CompositionSection section;
  section.type = GENERATE(DiagramType::PT, DiagramType::PS, DiagramType::TV,
                          DiagramType::SV);
  const std::size_t axis = GENERATE(std::size_t{0}, std::size_t{1});
  const double scale = GENERATE(1.e-3, 1., 1000.);
  State a;
  a.success = true;
  a.pressure = 1.e9;
  a.temperature = 1000.;
  a.entropy = 100. * scale;
  a.volume = 1.e-5 * scale;
  a.phases.resize(1);
  a.phases[0].amount = scale;
  a.gibbs = -1.e6 * scale;
  auto b = a;
  double dx = 0., derivative = 0., change = 0.;
  // Analytic stable quadratic along each natural coordinate. Its endpoint
  // derivatives can also bracket a latent mixture with a derivative jump.
  switch (diagram_axes(section.type)[axis]) {
  case Coordinate::P:
    dx = 1.e8;
    derivative = a.volume;
    change = -1.e-6 * scale;
    b.pressure += dx;
    b.volume += change;
    break;
  case Coordinate::T:
    dx = 100.;
    derivative = -a.entropy;
    change = -10. * scale;
    b.temperature += dx;
    b.entropy -= change;
    break;
  case Coordinate::S:
    dx = 10. * scale;
    derivative = a.temperature;
    change = 100.;
    b.entropy += dx;
    b.temperature += change;
    break;
  case Coordinate::V:
    dx = 1.e-6 * scale;
    derivative = -a.pressure;
    change = 1.e8;
    b.volume += dx;
    b.pressure -= change;
    break;
  case Coordinate::X:
    FAIL("Unexpected composition coordinate");
  }
  const double initial = thermodynamic_potential(a, section);
  const double offset = thermodynamic_potential(b, section) - b.gibbs;
  auto set_energy = [&](double energy) { b.gibbs = energy - offset; };
  set_energy(initial + (derivative + .5 * change) * dx);
  CHECK(thermodynamic_violation(a, b, section, axis) == 0.);
  CHECK(thermodynamic_violation(b, a, section, axis) == 0.);
  // A real derivative discontinuity is permitted; do not demand equal
  // endpoint derivatives or a finite positive curvature.
  set_energy(initial + derivative * dx);
  CHECK(thermodynamic_violation(a, b, section, axis) == 0.);
  set_energy(initial + (derivative + change) * dx);
  CHECK(thermodynamic_violation(a, b, section, axis) == 0.);
  set_energy(initial + (derivative + .5 * change) * dx +
             2. * std::abs(change * dx));
  CHECK(thermodynamic_violation(a, b, section, axis) > 0.);
  CHECK(thermodynamic_violation(b, a, section, axis) > 0.);
  b.composition_coordinate = 1.;
  CHECK(thermodynamic_violation(a, b, section, axis) == 0.);
}

TEST_CASE("Measured solve uncertainty permits unresolved energy differences",
          "[phase_diagram][consistency]") {
  CompositionSection section;
  section.type = DiagramType::PS;
  State a;
  a.success = true;
  a.temperature = 1000.;
  a.entropy = 100.;
  a.gibbs = -1.e6;
  a.phases.resize(1);
  a.phases[0].amount = 1.;
  auto b = a;
  b.entropy += 1.;
  b.gibbs += .01; // A small energy inconsistency on an otherwise flat H(S).
  CHECK(thermodynamic_violation(a, b, section, 1) > 0.);
  b.equilibrium_error = .02;
  CHECK(thermodynamic_violation(a, b, section, 1) == 0.);
  b.equilibrium_error = 0.;
  b.outside_model_domain = true;
  b.success = false;
  CHECK(thermodynamic_violation(a, b, section, 1) == 0.);
}

TEST_CASE("Affinity uncertainty respects changes in reaction mole amounts",
          "[phase_diagram][consistency]") {
  CompositionSection section;
  section.type = GENERATE(DiagramType::PT, DiagramType::PS, DiagramType::TV,
                          DiagramType::SV);
  const double scale = GENERATE(1.e-3, 1., 1000.);
  State reactant;
  reactant.success = true;
  reactant.pressure = 1.e9;
  reactant.temperature = 1000.;
  reactant.entropy = 100. * scale;
  reactant.volume = 1.e-5 * scale;
  reactant.gibbs = -1.e6 * scale;
  reactant.phases.resize(1);
  reactant.phases[0].amount = scale;
  auto products = reactant;
  products.phases.push_back(products.phases.front());
  // AB -> A + B makes two moles of products per mole of reactant. Each
  // candidate affinity can be -0.2 J/mol within the accepted solve tolerance.
  reactant.minimum_affinity = -.2;
  reactant.gibbs += .3 * scale;
  CHECK(thermodynamic_violation(reactant, products, section, 0) == 0.);
  CHECK(thermodynamic_violation(products, reactant, section, 0) == 0.);
  reactant.gibbs += .2 * scale; // Exceeds the combined affinity uncertainty.
  CHECK(thermodynamic_violation(reactant, products, section, 0) > 0.);
  CHECK(thermodynamic_violation(products, reactant, section, 0) > 0.);
  reactant.gibbs = products.gibbs;
  reactant.minimum_affinity = std::numeric_limits<double>::quiet_NaN();
  CHECK(thermodynamic_violation(reactant, products, section, 0) > 0.);
}

TEST_CASE("Seed consistency recovery replaces a higher energy phase before "
          "bracketing",
          "[phase_diagram][consistency]") {
  auto stable = burnman::minerals::HGP_2018_ds633::per();
  auto metastable =
      std::make_shared<burnman::Mineral>(burnman::make_combined_mineral(
          {*stable}, Eigen::ArrayXd::Ones(1), {10000., 0., 0.}, "metastable"));
  Settings settings;
  settings.max_phase_instances = 1;
  Engine engine({{"Mg", 1.}, {"O", 1.}}, {stable, metastable}, settings);
  std::vector<WorkState> grid{engine.stable_at({3.e9, 1000.})};
  REQUIRE(grid[0].state.success);
  PhaseState phase;
  phase.id = phase.candidate_index = 1;
  phase.amount = 1.;
  phase.composition = Eigen::VectorXd::Ones(1);
  auto a = engine.make_at({1}, {phase}, {3.e9, 1001.});
  // This root has exact mass balance and reaction equilibrium. As in an
  // assigned-field grid solve, those local conditions alone accept it.
  grid.push_back({equilibrium_state(engine, *a, {1}), a, {1}});
  REQUIRE(thermodynamic_violation(grid[0].state, grid[1].state, engine.section,
                                  1) > 0.);
  std::vector<std::size_t> updated;
  const auto diagnostics = recover_thermodynamic_grid(
      engine, grid, 2, [&](std::size_t k) { updated.push_back(k); });
  CHECK(diagnostics.empty());
  CHECK(!updated.empty());
  CHECK(grid[1].state.success);
  CHECK(grid[1].ids == std::vector<int>{0});
  CHECK(thermodynamic_violation(grid[0].state, grid[1].state, engine.section,
                                1) == 0.);
  CHECK(grid[1].state.mass_balance_error <= settings.mass_balance_tolerance);
}

TEST_CASE("Final diagram audit revisits pairs after late changes",
          "[phase_diagram][consistency]") {
  Result result;
  result.section.type = DiagramType::PS;
  result.resolved = true;
  for (double entropy : {100., 101., 102.}) {
    State state;
    state.success = true;
    state.pressure = 1.e9;
    state.temperature = 1000.;
    state.entropy = entropy;
    state.gibbs = -1.e6;
    result.samples.push_back(state);
  }
  audit_thermodynamic_samples(result);
  CHECK(result.resolved);
  CHECK(result.thermodynamic_validation.checked_pairs == 2);
  CHECK(result.thermodynamic_validation.violations == 0);
  // An earlier comparison passed, but a later repair changed one endpoint.
  result.samples.front().gibbs += 10.;
  result.diagnostics.push_back("Keep this unrelated boundary diagnostic.");
  audit_thermodynamic_samples(result);
  CHECK_FALSE(result.resolved);
  CHECK(result.thermodynamic_validation.violations == 1);
  CHECK(result.diagnostics.size() == 2);
  CHECK(result.diagnostics.front() ==
        "Keep this unrelated boundary diagnostic.");
  audit_thermodynamic_samples(result);
  CHECK(result.diagnostics.size() == 2); // No duplicate audit messages.
  result.samples.front().gibbs -= 10.;
  audit_thermodynamic_samples(result);
  CHECK(result.thermodynamic_validation.violations == 0);
  CHECK(result.diagnostics.size() == 1);
  CHECK(result.equilibrium_solves == 0);
  CHECK(result.samples.front().gibbs == -1.e6); // Read-only evidence.
}

TEST_CASE("Final audit distinguishes unchecked and inapplicable pairs",
          "[phase_diagram][consistency]") {
  Result result;
  CHECK(result.thermodynamic_validation.scope == "not_checked");
  result.section.type = DiagramType::PT;
  result.samples.resize(2);
  auto &a = result.samples[0];
  a.success = true;
  a.temperature = 1000.;
  a.gibbs = -1.e6;
  result.samples[1] = a;
  result.samples[1].gibbs += 1.;
  audit_thermodynamic_samples(result);
  CHECK(result.thermodynamic_validation.violations >
        0); // Same location, two roots.
  result.samples[1].outside_model_domain = true;
  audit_thermodynamic_samples(result);
  CHECK(result.thermodynamic_validation.checked_pairs == 0);
  CHECK(result.thermodynamic_validation.skipped_pairs == 2);
  CHECK(result.thermodynamic_validation.violations == 0);
  result.section.type = DiagramType::PV;
  audit_thermodynamic_samples(result);
  CHECK(result.thermodynamic_validation.scope == "not_applicable");
}
