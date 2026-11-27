// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/tools/averaging/averaging_schemes.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

using namespace Catch::Matchers;
using namespace burnman::averaging;

namespace {
std::pair<double, double> moduli(std::initializer_list<double> volumes,
                                 std::initializer_list<double> bulk,
                                 std::initializer_list<double> shear) {
  return utils::self_consistent_fn(
      Eigen::Map<const Eigen::ArrayXd>(
          volumes.begin(), static_cast<Eigen::Index>(volumes.size())),
      Eigen::Map<const Eigen::ArrayXd>(bulk.begin(),
                                       static_cast<Eigen::Index>(bulk.size())),
      Eigen::Map<const Eigen::ArrayXd>(
          shear.begin(), static_cast<Eigen::Index>(shear.size())));
}
} // namespace

TEST_CASE("Self-consistent pure phases and absent phases",
          "[tools][averaging_schemes][self_consistent]") {
  CHECK(moduli({2.0}, {160.0}, {100.0}) == std::make_pair(160.0, 100.0));
  CHECK(moduli({2.0}, {10.0}, {0.0}) == std::make_pair(10.0, 0.0));
  CHECK(moduli({2.0}, {0.0}, {0.0}) == std::make_pair(0.0, 0.0));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(moduli({0.0, 2.0, 0.0}, {nan, 160.0, -1.0}, {nan, 100.0, -1.0}) ==
        std::make_pair(160.0, 100.0));
  const auto identical =
      moduli({1.0, 2.0, 3.0}, {160.0, 160.0, 160.0}, {100.0, 100.0, 100.0});
  CHECK_THAT(identical.first, WithinRel(160.0, 1.e-12));
  CHECK_THAT(identical.second, WithinRel(100.0, 1.e-12));
  CHECK(moduli({1.0, 2.0}, {0.0, 0.0}, {0.0, 0.0}) == std::make_pair(0.0, 0.0));
}

TEST_CASE("Self-consistent analytic two-phase solution",
          "[tools][averaging_schemes][self_consistent]") {
  // With K_i=4G_i/3 and equal volumes, G=sqrt(G_1 G_2), K=4G/3.
  const auto result = moduli({1.0, 1.0}, {4.0 / 3.0, 16.0 / 3.0}, {1.0, 4.0});
  CHECK_THAT(result.first, WithinRel(8.0 / 3.0, 1.e-12));
  CHECK_THAT(result.second, WithinRel(2.0, 1.e-12));
}

TEST_CASE("Self-consistent published Echoes reference values",
          "[tools][averaging_schemes][self_consistent]") {
  // Independent SC outputs in the spherical pore examples:
  // https://jfbarthelemy.github.io/echoes/4_schemes/schemes.html
  const auto first = moduli({0.8, 0.2}, {72.0, 1.e-6}, {32.0, 1.e-6});
  CHECK_THAT(first.first, WithinAbs(37.156, 5.1e-4));
  CHECK_THAT(first.second, WithinAbs(19.629, 5.1e-4));
  const auto second = moduli({0.7, 0.3}, {72.0, 1.e-6}, {32.0, 1.e-6});
  CHECK_THAT(second.first, WithinAbs(22.689245, 2.e-5));
  CHECK_THAT(second.second, WithinAbs(13.264364, 2.e-5));
  // Python BurnMan's more precise scalar result for the same inputs.
  CHECK_THAT(first.first, WithinRel(37.15644096821864, 1.e-11));
  CHECK_THAT(first.second, WithinRel(19.629142176318542, 1.e-11));
}

TEST_CASE("Self-consistent multiphase equations, bounds and scaling",
          "[tools][averaging_schemes][self_consistent]") {
  const Eigen::ArrayXd volumes =
      (Eigen::ArrayXd(3) << 2.0, 3.0, 5.0).finished();
  const Eigen::ArrayXd bulk =
      (Eigen::ArrayXd(3) << 160.0, 80.0, 240.0).finished();
  const Eigen::ArrayXd shear =
      (Eigen::ArrayXd(3) << 100.0, 30.0, 60.0).finished();
  const auto result = utils::self_consistent_fn(volumes, bulk, shear);
  const double k = result.first, g = result.second;
  const Eigen::ArrayXd fractions = volumes / volumes.sum();
  const double zeta = g * (9.0 * k + 8.0 * g) / (6.0 * (k + 2.0 * g));
  CHECK_THAT((fractions * (bulk - k) / (bulk + 4.0 * g / 3.0)).sum(),
             WithinAbs(0.0, 1.e-12));
  CHECK_THAT((fractions * (shear - g) / (shear + zeta)).sum(),
             WithinAbs(0.0, 1.e-12));
  HashinShtrikmanLower lower;
  HashinShtrikmanUpper upper;
  CHECK(k >= lower.average_bulk_moduli(volumes, bulk, shear));
  CHECK(k <= upper.average_bulk_moduli(volumes, bulk, shear));
  CHECK(g >= lower.average_shear_moduli(volumes, bulk, shear));
  CHECK(g <= upper.average_shear_moduli(volumes, bulk, shear));
  SelfConsistent scheme;
  CHECK(scheme.average_bulk_moduli(volumes, bulk, shear) == k);
  CHECK(scheme.average_shear_moduli(volumes, bulk, shear) == g);
  for (double amount : {1.e-100, 1.e100}) {
    for (double unit : {1.e-9, 1.e9}) {
      const auto scaled = utils::self_consistent_fn(volumes * amount,
                                                    bulk * unit, shear * unit);
      CHECK_THAT(scaled.first / unit, WithinRel(k, 1.e-11));
      CHECK_THAT(scaled.second / unit, WithinRel(g, 1.e-11));
    }
  }
  const auto reversed = utils::self_consistent_fn(
      volumes.reverse(), bulk.reverse(), shear.reverse());
  CHECK_THAT(reversed.first, WithinRel(k, 1.e-11));
  CHECK_THAT(reversed.second, WithinRel(g, 1.e-11));
}

TEST_CASE("Self-consistent fluid and void rigidity limits",
          "[tools][averaging_schemes][self_consistent]") {
  const auto fluids = moduli({1.0, 3.0}, {10.0, 20.0}, {0.0, 0.0});
  CHECK_THAT(fluids.first, WithinRel(16.0, 1.e-12));
  CHECK(fluids.second == 0.0);
  for (double porosity : {0.49, 0.4999, 0.5, 0.7, 1.0}) {
    const auto result =
        moduli({1.0 - porosity, porosity}, {72.0, 0.0}, {32.0, 0.0});
    if (porosity < 0.5) {
      CHECK(result.first > 0.0);
      CHECK(result.second > 0.0);
    } else {
      CHECK(result.first == 0.0);
      CHECK(result.second == 0.0);
    }
  }
  for (double fraction : {0.59, 0.6, 0.8, 1.0}) {
    const auto result =
        moduli({1.0 - fraction, fraction}, {72.0, 2.0}, {32.0, 0.0});
    if (fraction < 0.6) {
      CHECK(result.second > 0.0);
    } else {
      CHECK(result.second == 0.0);
      CHECK_THAT(
          result.first,
          WithinRel(1.0 / ((1.0 - fraction) / 72.0 + fraction / 2.0), 1.e-12));
    }
  }
  const auto soft_pores = moduli({0.4, 0.6}, {72.0, 1.e-6}, {32.0, 1.e-6});
  CHECK(soft_pores.first > 1.e-6);
  CHECK(soft_pores.second > 1.e-6);
}

TEST_CASE("Self-consistent input validation",
          "[tools][averaging_schemes][self_consistent]") {
  CHECK_THROWS_AS(moduli({}, {}, {}), std::invalid_argument);
  CHECK_THROWS_AS(moduli({1.0}, {1.0, 2.0}, {1.0}), std::invalid_argument);
  CHECK_THROWS_AS(moduli({0.0, 0.0}, {1.0, 2.0}, {1.0, 2.0}),
                  std::invalid_argument);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double invalid : {-1.0, nan, inf}) {
    CHECK_THROWS_AS(moduli({invalid}, {1.0}, {1.0}), std::invalid_argument);
    CHECK_THROWS_AS(moduli({1.0}, {invalid}, {1.0}), std::invalid_argument);
    CHECK_THROWS_AS(moduli({1.0}, {1.0}, {invalid}), std::invalid_argument);
  }
}
