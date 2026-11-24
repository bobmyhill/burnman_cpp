// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/minerals/model_sets.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace burnman;

TEST_CASE("A third garnet basin is tested beyond the initial start budget",
          "[phase_diagram]") {
  // Tangent plane at a failed basalt reaction near 7 GPa. Ranking the initial
  // energies of the 15 garnet vertices omitted the most stable relaxed basin.
  const types::FormulaMap bulk{
      {"Al", .32953711804098035},   {"Ca", .22361949733760836},
      {"Na", .07195985737104503},   {"Fe", .11222579765361212},
      {"Cr", .0009211108069983367}, {"Mg", .24240529569972505},
      {"O", 2.7949270633834957},    {"Si", .8391543215116095}};
  const types::FormulaMap potentials{
      {"O", -143774.778434},  {"Fe", -56732.5863853}, {"Al", -497022.270071},
      {"Mg", -386067.733121}, {"Cr", -302679.981204}, {"Ca", -473581.166324},
      {"Na", -223756.248899}, {"Si", -434623.296005}};
  phase_diagrams::Settings settings;
  settings.minimization_starts = 10;
  auto models = minerals::model_sets::slb2024();
  phase_diagrams::detail::Engine engine(bulk, models.phases, settings);
  engine.set_pt(7019186018.196769, 385.41364900928954);
  Eigen::VectorXd mu(static_cast<Eigen::Index>(engine.components.size()));
  for (std::size_t i = 0; i < engine.components.size(); ++i)
    mu[static_cast<Eigen::Index>(i)] =
        potentials.at(engine.elements[engine.components[i]]);
  const auto ordinary = engine.minima(6, mu);
  REQUIRE(ordinary.size() >= 3);
  CHECK(ordinary.front().affinity < -9.);
  engine.settings.minimization_starts = 100;
  const auto exhaustive = engine.minima(6, mu);
  CHECK_THAT(ordinary.front().affinity,
             Catch::Matchers::WithinAbs(exhaustive.front().affinity, 1.e-3));
}

TEST_CASE("A junction correction cannot skip a solution coalescence",
          "[phase_diagram]") {
  // Both roots pass the old zero-copy basin check at the target P,T, but
  // the persistent garnet copies exchange their order across a critical state.
  auto models = minerals::model_sets::slb2024();
  phase_diagrams::Settings settings;
  phase_diagrams::detail::Engine engine({{"Mg", 1.}, {"O", 1.}}, models.phases,
                                        settings);
  phase_diagrams::detail::PhaseStates before, after;
  {
    phase_diagrams::PhaseState phase;
    phase.id = 18;
    phase.candidate_index = 6;
    phase.amount = 0.12995206540804377;
    phase.composition.resize(7);
    phase.composition << 0.31177874517042714, 0.2513630803631991,
        0.42933298916842189, 7.2371182040934261e-06, 0.00028455899818621865,
        0.0036905592145717405, 0.003542829966989841;
    before.push_back(phase);
  }
  {
    phase_diagrams::PhaseState phase;
    phase.id = 19;
    phase.candidate_index = 6;
    phase.amount = 0;
    phase.composition.resize(7);
    phase.composition << 0.39991445822562444, 0.25382979999944988,
        0.34100675380274892, 1.0014941900856369e-05, 0.00034079267383324568,
        0.0029038543369565937, 0.0019943260194861157;
    before.push_back(phase);
  }
  {
    phase_diagrams::PhaseState phase;
    phase.id = 18;
    phase.candidate_index = 6;
    phase.amount = 0.12993970368423235;
    phase.composition.resize(7);
    phase.composition << 0.38572829382079932, 0.25347970966450883,
        0.35357550770771079, 1.1312357352063924e-05, 0.00037150412848198637,
        0.0032913822467663259, 0.0035422900743806406;
    after.push_back(phase);
  }
  {
    phase_diagrams::PhaseState phase;
    phase.id = 19;
    phase.candidate_index = 6;
    phase.amount = 0;
    phase.composition.resize(7);
    phase.composition << 0.30491649743484683, 0.24997583415385038,
        0.43474778168309608, 8.4226264246934961e-06, 0.00031450380349963797,
        0.0040632062282964622, 0.0059737540699858873;
    after.push_back(phase);
  }
  const double pressure = 11649049839.652113, temperature = 476.38733662394253;
  REQUIRE(engine.same_composition_basin(
      6, before[1].composition, after[1].composition, pressure, temperature));
  CHECK(phase_diagrams::detail::same_solution_branches(engine, before, before,
                                                       pressure, temperature));
  CHECK_FALSE(phase_diagrams::detail::same_solution_branches(
      engine, before, after, pressure, temperature));
}

TEST_CASE("Positive garnet copies cannot jump between mixing basins",
          "[phase_diagram]") {
  // A V-T pyrolite Fe-out trace had stable endpoints but skipped a third
  // garnet field. Its positive first copy changed basin without reversing
  // the pair's composition order; checking only zero-amount copies missed it.
  auto models = minerals::model_sets::slb2024();
  phase_diagrams::detail::Engine engine({{"Mg", 1.}, {"O", 1.}}, models.phases,
                                        phase_diagrams::Settings{});
  auto state = [](int id, std::initializer_list<double> composition) {
    phase_diagrams::PhaseState phase;
    phase.id = id;
    phase.candidate_index = 6;
    phase.amount = .02;
    phase.composition = Eigen::Map<const Eigen::VectorXd>(
        composition.begin(), static_cast<Eigen::Index>(composition.size()));
    return phase;
  };
  const phase_diagrams::detail::PhaseStates before{
      state(18,
            {-.21711276276447508, .09483631598950071, .8787993918819401, 0.,
             .0004728457788676194, .000004585340224072532, .24299962377394246}),
      state(19, {.4649358718494968, .3065990353984142, .2225340518997586, 0.,
                 .0028295339666611708, .00000830167868226539,
                 .0030932052069868876})};
  const phase_diagrams::detail::PhaseStates after{
      state(18,
            {.07257697369029838, .24030799344719847, .637273682610792, 0.,
             .0014212340728593333, .000014042295295557672, .04840607388355622}),
      state(19, {.4712941681627739, .29761783409956766, .22477919585949888, 0.,
                 .003203439064527085, .00000884120524491383,
                 .0030965216083875535})};
  const double pressure = 17844152422.41068, temperature = 522.6257538736935;
  REQUIRE((before[0].composition - before[1].composition)
              .dot(after[0].composition - after[1].composition) > 0.);
  REQUIRE_FALSE(engine.same_composition_basin(
      6, before[0].composition, after[0].composition, pressure, temperature));
  CHECK(phase_diagrams::detail::same_solution_branches(engine, before, before,
                                                       pressure, temperature));
  CHECK_FALSE(phase_diagrams::detail::same_solution_branches(
      engine, before, after, pressure, temperature));
}
