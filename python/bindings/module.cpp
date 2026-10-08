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

#include "bindings.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/exceptions.hpp"

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native BurnMan C++ bindings. All quantities use SI units.";
  m.attr("__version__") = BURNMAN_VERSION;
  auto constants = m.def_submodule(
      "constants", "Physical constants and the native logarithm tolerance.");
  constants.attr("gas_constant") = burnman::constants::physics::gas_constant;
  constants.attr("Avogadro") = burnman::constants::physics::avogadro;
  constants.attr("Boltzmann") = burnman::constants::physics::boltzmann;
  constants.attr("G") = burnman::constants::physics::gravitation;
  constants.attr("Dirac") = burnman::constants::physics::dirac;
  constants.attr("invcm") = burnman::constants::physics::invcm;
  constants.attr("logish_eps") = burnman::constants::precision::logish_eps;
  py::register_exception<burnman::exceptions::NotImplementedError>(
      m, "NotImplementedError", PyExc_NotImplementedError);
  burnman::python::bind_params(m);
  burnman::python::bind_materials(m);
  burnman::python::bind_composition(m);
  burnman::python::bind_combined(m);
  burnman::python::bind_solutions(m);
  burnman::python::bind_minerals(m);
  burnman::python::bind_model_sets(m);
  burnman::python::bind_equilibration(m);
  burnman::python::bind_polytope(m);
  burnman::python::bind_pseudosection(m);
}
