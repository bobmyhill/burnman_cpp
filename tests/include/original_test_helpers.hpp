// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
#pragma once

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <memory>

namespace original {
// BurnManTest.assertFloatEqual's original relative/absolute rule.
inline void check_close(double actual, double expected, double rtol = 1.e-5,
                        double atol = 1.e-16) {
  CHECK(std::abs(actual - expected) <=
        std::max(atol, rtol * std::max(std::abs(actual), std::abs(expected))));
}

// SciPy quad's adaptive Gauss-Kronrod integration and default tolerances.
template <typename Function>
double quad(Function function, double a, double b, double atol = 1.49e-8) {
  if (a == b)
    return 0.;
  std::unique_ptr<gsl_integration_workspace,
                  decltype(&gsl_integration_workspace_free)>
      workspace(gsl_integration_workspace_alloc(1000),
                &gsl_integration_workspace_free);
  gsl_function integrand;
  integrand.function = [](double x, void *data) {
    return (*static_cast<Function *>(data))(x);
  };
  integrand.params = &function;
  double result, error;
  const int status =
      gsl_integration_qags(&integrand, std::min(a, b), std::max(a, b), atol,
                           1.49e-8, 1000, workspace.get(), &result, &error);
  REQUIRE(status == 0);
  return a < b ? result : -result;
}
} // namespace original
