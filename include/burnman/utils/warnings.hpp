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

#ifndef BURNMAN_UTILS_WARNINGS_HPP_INCLUDED
#define BURNMAN_UTILS_WARNINGS_HPP_INCLUDED

#include <iostream>
#include <string>

namespace burnman {
namespace utils {
/**
 * @brief Flag to suppress warnings across burnman.
 */
inline bool suppress_warnings = false;

/**
 * @brief Prints a warning message.
 *
 * @param message The warning message to print.
 */
inline void warn(const std::string &message) {
  // Simple implementation: print to std::cerr
  // Could switch to a logging framework if needed
  if (!suppress_warnings) {
    std::cerr << "Warning: " << message << std::endl;
  }
}
} // namespace utils
} // namespace burnman

#endif // BURNMAN_UTILS_WARNINGS_HPP_INCLUDED
