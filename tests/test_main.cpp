/*
 * Copyright (c) 2025 Benedict Heinen
 *
 * This file is part of burnman_cpp and is licensed under the
 * GNU General Public License v3.0 or later. See the LICENSE file
 * or <https://www.gnu.org/licenses/> for details.
 *
 * burnman_cpp is based on BurnMan: <https://geodynamics.github.io/burnman/>
 */
#define CATCH_CONFIG_MAIN
#include "burnman/utils/warnings.hpp"
#include <catch2/catch_all.hpp>

namespace {
// Helper struct to suppress warnings across tests
struct SuppressAllWarnings {
  SuppressAllWarnings() { burnman::utils::suppress_warnings = true; }
};
SuppressAllWarnings suppress_warnings;
} // namespace
