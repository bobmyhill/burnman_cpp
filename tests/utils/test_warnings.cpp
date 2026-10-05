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

#include "burnman/utils/warnings.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <sstream>
#include <string>

using namespace burnman;

TEST_CASE("Validate warnings", "[utils][warnings]") {
  // Temporarily allow warnings
  utils::suppress_warnings = false;
  // Redirect cerr to grab output
  std::streambuf *orig_cerr = std::cerr.rdbuf();
  std::ostringstream oss;
  std::cerr.rdbuf(oss.rdbuf());
  // Generate a warning
  utils::warn("This is a test warning!");
  REQUIRE(oss.str() == "Warning: This is a test warning!\n");
  // Restore cerr buffer
  std::cerr.rdbuf(orig_cerr);
  // Block warnings again
  utils::suppress_warnings = true;
}
