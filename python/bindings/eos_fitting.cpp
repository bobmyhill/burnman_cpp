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

#include "burnman/optim/eos_fitting.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_eos_fitting(py::module_ &m) {
  auto eos_fitting =
      m.attr("optimize").cast<py::module_>().def_submodule("eos_fitting");
  eos_fitting.def(
      "default_mle_tolerances", &optim::default_mle_tolerances,
      py::arg("material"), py::arg("flags"),
      R"doc(Return one default MLE tolerance per property flag, in input order.

Sets and leaves material at 1e5 Pa and 300 K, including for empty flags.
H, energy, molar_internal_energy, helmholtz, molar_helmholtz, gibbs,
molar_gibbs, enthalpy and molar_enthalpy use 1 J/mol. P and pressure use
100 Pa. All other flags use 1e-5 times the material property at that state.
Raises ValueError if any resulting tolerance is not strictly positive.
)doc");
}
} // namespace burnman::python
