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

#include "burnman/utils/exceptions.hpp"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

using namespace burnman;

TEST_CASE("NotImplementedError message", "[utils][exceptions]") {
  const std::string class_name = "AClass";
  const std::string func_name = "some_function";
  const std::string message = "Hello!";
  try {
    throw exceptions::NotImplementedError(class_name, func_name, message);
  } catch (const exceptions::NotImplementedError &e) {
    REQUIRE(std::string(e.what()) == "[AClass::some_function] Hello!");
  } catch (...) {
    FAIL("NotImplementedError not thrown?");
  }
}

TEST_CASE("NotImplementedError default message", "[utils][exceptions]") {
  try {
    throw exceptions::NotImplementedError("Class", "func");
  } catch (const std::logic_error &e) {
    REQUIRE(std::string(e.what()) == "[Class::func] Function not implemented!");
  }
}

TEST_CASE("NotImplementedError derived from std::logic_error",
          "[utils][exceptions]") {
  exceptions::NotImplementedError err("Class", "func", "msg");
  std::logic_error *base_ptr = &err;
  REQUIRE(base_ptr != nullptr);
}
