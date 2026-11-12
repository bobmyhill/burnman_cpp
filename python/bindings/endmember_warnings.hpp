// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once

#include "bindings.hpp"
#include "burnman/core/solution.hpp"

namespace burnman::python {
inline void warn_if_outside_endmember_constraints(const Solution &solution) {
  const auto &constraints = solution.endmember_constraints;
  if (!constraints || !constraints->rows() ||
      solution.get_molar_fractions().size() != constraints->cols())
    return;
  const Eigen::VectorXd values =
      solution.evaluate_endmember_constraints(solution.get_molar_fractions());
  for (Eigen::Index i = 0; i < constraints->rows(); ++i) {
    if (values[i] < -1.e-10 * constraints->row(i).norm()) {
      const std::string message =
          "The composition of " + solution.get_name() +
          " violates its endmember_constraints; thermodynamic properties "
          "are still evaluated.";
      py::module_::import("warnings").attr("warn")(message);
      return;
    }
  }
}
} // namespace burnman::python
