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

#pragma once

#include "burnman/core/material.hpp"
#include "burnman/eos/components/excess_params.hpp"
#include <cmath>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace burnman::python {
void bind_params(py::module_ &m);
void bind_materials(py::module_ &m);
void bind_composition(py::module_ &m);
void bind_solutions(py::module_ &m);
void bind_equilibration(py::module_ &m);
void bind_combined(py::module_ &m);
void bind_minerals(py::module_ &m);
void bind_polytope(py::module_ &m);
void bind_pseudosection(py::module_ &m);
eos::excesses::ExcessParamVector parse_modifiers(const py::iterable &modifiers);

inline void check_state(const Material &material) {
  if (!material.has_state()) {
    throw std::runtime_error("Call set_state(pressure, temperature) before "
                             "querying state properties.");
  }
}

inline void check_pt(double pressure, double temperature) {
  if (!std::isfinite(pressure) || !std::isfinite(temperature) ||
      temperature < 0.0) {
    throw std::invalid_argument("Pressure must be finite and temperature must "
                                "be finite and nonnegative.");
  }
}

inline void check_fractions(const Eigen::ArrayXd &fractions, Eigen::Index size,
                            bool unit_sum = true) {
  if (fractions.size() != size || !fractions.isFinite().all() ||
      (fractions < 0.0).any() || fractions.sum() <= 0.0) {
    throw std::invalid_argument(
        "Fractions must match the number of components, be finite and "
        "nonnegative, and have a positive sum.");
  }
  if (unit_sum && std::abs(fractions.sum() - 1.0) > 1.0e-12) {
    throw std::invalid_argument("Molar fractions must sum to one.");
  }
}

inline void check_site_fractions(const Eigen::ArrayXd &fractions,
                                 const Eigen::ArrayXXd &occupancies) {
  if (fractions.size() != occupancies.rows() || !fractions.isFinite().all() ||
      std::abs(fractions.sum() - 1.0) > 1.0e-12) {
    throw std::invalid_argument("Solution fractions must be finite, match the "
                                "endmembers, and sum to one.");
  }
  // Transformed ordering models can have signed endmember coordinates.
  // Physical bounds apply to site occupancies, as in the native solver.
  if ((occupancies.matrix().transpose() * fractions.matrix()).minCoeff() <
      -1.0e-10) {
    throw std::invalid_argument(
        "Solution fractions must give nonnegative site occupancies.");
  }
}

// Return copies: cached Eigen arrays must not become dangling or mutable NumPy
// views.
template <typename Class, typename Getter, typename Binding>
void bind_property(Binding &cls, const char *name, const char *getter_name,
                   Getter getter, bool state_required = false) {
  auto call = [getter, state_required](const Class &self) {
    if (state_required)
      check_state(self);
    return (self.*getter)();
  };
  cls.def_property_readonly(name, call, py::return_value_policy::copy);
  cls.def(getter_name, call, py::return_value_policy::copy);
}
} // namespace burnman::python
