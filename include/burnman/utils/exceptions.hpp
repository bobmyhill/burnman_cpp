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

#ifndef BURNMAN_UTILS_EXCEPTIONS_HPP_INCLUDED
#define BURNMAN_UTILS_EXCEPTIONS_HPP_INCLUDED

#include <stdexcept>
#include <string>

namespace burnman {
/**
 * @namespace burnman::exceptions
 * @brief Custom exceptions
 */
namespace exceptions {
/**
 * @class NotImplementedError
 * @brief Custom exception derived from std::logic_error
 * Usage: throw NotImplementedError(class_name, func_name)
 *        class_name from e.g. typeid(*this).name();
 *        func_name from e.g. __func__ identifier
 */
class NotImplementedError : public std::logic_error {
public:
  explicit NotImplementedError(
      const std::string &class_name, const std::string &func_name,
      const std::string &message = "Function not implemented!")
      : std::logic_error("[" + class_name + "::" + func_name + "] " + message) {
  }
};
} // namespace exceptions
} // namespace burnman

#endif // BURNMAN_UTILS_EXCEPTIONS_HPP_INCLUDED