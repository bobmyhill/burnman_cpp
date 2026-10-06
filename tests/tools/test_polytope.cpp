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
#include "burnman/tools/polytope.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace burnman;
using namespace burnman::polytope;
using namespace Catch::Matchers;

TEST_CASE("cddlib native rational polytope enumeration", "[polytope]") {
  Eigen::MatrixXd eq(1, 3), ineq(2, 3);
  eq << -1. / 3., 1., 1.;
  ineq << 0., 1., 0., 0., 0., 1.;
  MaterialPolytope p(eq, ineq);
  CHECK_FALSE(p.is_empty());
  CHECK(p.is_bounded());
  REQUIRE(p.get_vertices().rows() == 2);
  for (Eigen::Index i = 0; i < 2; ++i)
    CHECK_THAT(p.get_vertices().row(i).sum(), WithinAbs(1. / 3., 1.e-14));
}

TEST_CASE("General native solution simplification", "[polytope]") {
  auto opx = minerals::JH15::orthopyroxene();
  Assemblage a;
  a.add_phases({opx});
  a.set_fractions({1.});
  auto reduced = simplify_composite_with_composition(
      a, {{"Mg", 1.}, {"Fe", 1.}, {"Si", 2.}, {"O", 6.}});
  REQUIRE(reduced->get_n_phases() == 1);
  auto child = reduced->get_phase<Solution>(0);
  REQUIRE(child);
  REQUIRE(child->get_n_endmembers() == 3);
  Eigen::ArrayXd fractions(3);
  fractions << .2, .3, .5;
  child->set_composition(fractions);
  opx->set_composition(
      (child->get_basis().transpose() * fractions.matrix()).array());
  child->set_state(1.e9, 1000.);
  opx->set_state(1.e9, 1000.);
  CHECK_THAT(child->get_molar_gibbs(),
             WithinRel(opx->get_molar_gibbs(), 1.e-12));
  CHECK_THAT(child->get_molar_entropy(),
             WithinRel(opx->get_molar_entropy(), 1.e-12));
}
