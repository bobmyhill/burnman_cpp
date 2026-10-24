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

#include "burnman/optim/linear_fitting.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_linear_fitting(py::module_ &m) {
  auto optimize = m.def_submodule("optimize");
  auto linear = optimize.def_submodule("linear_fitting");
  linear.def(
      "weighted_constrained_least_squares",
      &optim::weighted_constrained_least_squares, py::arg("A"), py::arg("b"),
      py::arg("Cov_b") = py::none(),
      py::arg("equality_constraints") = py::none(),
      py::arg("inequality_constraints") = py::none(),
      py::arg("allow_rank_deficient") = false,
      R"doc(Fit parameters by weighted least squares with linear constraints.

Minimise (A*x-b).T @ pinv(Cov_b) @ (A*x-b), subject to C*x=c and D*x<=d.
Cov_b defaults to the identity and must be symmetric and positive semidefinite.
Covariance eigenvalues at or below 1e-15 times the largest eigenvalue have zero
weight. Supply constraints as (C,c) and
(D,d), each containing a matrix and a vector. Empty constraints are allowed.

The weighted design must have full column rank unless allow_rank_deficient is
True. This check also applies when constraints determine a unique fit.

Return (popt, pcov, residual): the fitted parameters, the covariance
pinv(A.T @ pinv(Cov_b) @ A), and the weighted sum of squared residuals.
The covariance is not conditioned on the constraints or scaled by the residual.
Rank-deficient fits may have multiple parameter vectors with the same minimum.
)doc");
}
} // namespace burnman::python
