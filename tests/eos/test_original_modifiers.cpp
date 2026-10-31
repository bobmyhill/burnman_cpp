// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
// Ports of tests/test_modifiers.py; reference: tests/reference/burnman.json.

#include "burnman/eos/components/property_modifiers.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace burnman::eos::excesses;

TEST_CASE("Original modifier reference excesses", "[original][modifiers]") {
  // Modifiers.test_excess_functions: original parameter sets and rounded G.
  const ExcessParamVector params{
      LinearParams{1.e-7, 5., 1200.},
      LandauParams{800., 1.e-7, 5.},
      LandauParams{1200., 1.e-7, 5.},
      LandauHPParams{298.15, 1.e5, 800., 1.e-7, 5.},
      LandauHPParams{298.15, 1.e5, 1200., 1.e-7, 5.},
      BraggWilliamsParams{1, .8, 1000., 1.e-7, 1000., 1.e-7},
      MagneticChsParams{.4, 800., 1.e-8, 2.2, 1.e-10},
      MagneticChsParams{.4, 1200., 1.e-8, 2.2, 1.e-10},
      DebyeParams{1., 1200.},
      DebyeDeltaParams{1., 1200.},
      EinsteinParams{1., 1200.},
      EinsteinDeltaParams{1., 1200.}};
  const std::array<double, 12> expected{
      6200.,       1137.8563,   1019.3716, -2534.185, -1696.3556, -551.3463,
      -14069.1506, -21582.5564, -565.315,  -620.7977, -358.3824,  -517.2153};
  for (std::size_t i = 0; i < params.size(); ++i) {
    CAPTURE(i);
    const double G = std::visit(
        [](auto p) { return compute_excesses(1.e11, 1000., p).G; }, params[i]);
    const double rounded = std::round(G * 1.e4) / 1.e4;
    CHECK(std::abs(rounded - expected[i]) <=
          std::max(1.e-16,
                   std::max(std::abs(rounded), std::abs(expected[i])) * 1.e-5));
  }
}

TEST_CASE("Original SLB 2022 modifier derivatives", "[original][modifiers]") {
  // Modifiers.test_landau_slb_2022_derivatives, including the Q=2 limit.
  for (double Tc_0 : {800., 1200.}) {
    const LandauSLB2022Params params{Tc_0, 1.e-7, 5.};
    const double Tc = Tc_0 + 1.e-7 * 2.e9 / 5.;
    const std::array<std::array<double, 2>, 5> states{{{0., 300.},
                                                       {2.e9, 300.},
                                                       {2.e9, Tc - 10.},
                                                       {2.e9, Tc + 100.},
                                                       {1.e12, 300.}}};
    for (const auto &[P, T] : states) {
      CAPTURE(Tc_0, P, T);
      const double dP = 1.e5, dT = .01;
      const auto xs = compute_excesses(P, T, params);
      const auto pp = compute_excesses(P + dP, T, params);
      const auto pm = compute_excesses(P - dP, T, params);
      const auto tp = compute_excesses(P, T + dT, params);
      const auto tm = compute_excesses(P, T - dT, params);
      const std::array<std::array<double, 2>, 6> derivatives{
          {{xs.dGdP, (pp.G - pm.G) / (2. * dP)},
           {xs.dGdT, (tp.G - tm.G) / (2. * dT)},
           {xs.d2GdP2, (pp.dGdP - pm.dGdP) / (2. * dP)},
           {xs.d2GdT2, (tp.dGdT - tm.dGdT) / (2. * dT)},
           {xs.d2GdPdT, (pp.dGdT - pm.dGdT) / (2. * dP)},
           {xs.d2GdPdT, (tp.dGdP - tm.dGdP) / (2. * dT)}}};
      for (const auto &[analytic, numerical] : derivatives)
        CHECK(std::abs(analytic - numerical) <=
              1.e-26 + 1.e-6 * std::abs(numerical));
    }
  }
}
