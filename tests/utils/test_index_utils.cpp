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

#include "burnman/utils/index_utils.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>

TEST_CASE(
    "Native integer index conversions preserve values and reject overflow",
    "[utils][indices]") {
  using burnman::utils::checked_int;
  constexpr auto low = std::numeric_limits<int>::min();
  constexpr auto high = std::numeric_limits<int>::max();
  CHECK(checked_int(0) == 0);
  CHECK(checked_int(-1) == -1);
  CHECK(checked_int(static_cast<std::int64_t>(low)) == low);
  CHECK(checked_int(static_cast<std::uint64_t>(high)) == high);
  CHECK_THROWS_AS(checked_int(static_cast<std::int64_t>(low) - 1),
                  std::overflow_error);
  CHECK_THROWS_AS(checked_int(static_cast<std::int64_t>(high) + 1),
                  std::overflow_error);
  CHECK_THROWS_AS(checked_int(std::numeric_limits<std::uint64_t>::max()),
                  std::overflow_error);
}
