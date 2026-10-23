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

#include "bindings.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
inline Eigen::ArrayXd radial_array(const py::object &value) {
  return value.is_none() ? Eigen::ArrayXd{} : value.cast<Eigen::ArrayXd>();
}
inline py::object radial_values(const Eigen::MatrixXd &values) {
  if (values.rows() == 1)
    return py::cast(Eigen::ArrayXd(values.row(0).transpose()));
  return py::cast(values);
}
template <class Radial, class Binding>
void bind_radial_properties(Binding &binding) {
  const std::vector<std::string> names = {"pressure",
                                          "pressures",
                                          "temperature",
                                          "temperatures",
                                          "radii",
                                          "gravity",
                                          "bullen",
                                          "brunt_vasala",
                                          "molar_internal_energy",
                                          "molar_gibbs",
                                          "molar_helmholtz",
                                          "molar_mass",
                                          "molar_volume",
                                          "density",
                                          "molar_entropy",
                                          "molar_enthalpy",
                                          "isothermal_bulk_modulus_reuss",
                                          "isentropic_bulk_modulus_reuss",
                                          "isothermal_compressibility_reuss",
                                          "isentropic_compressibility_reuss",
                                          "shear_modulus",
                                          "p_wave_velocity",
                                          "bulk_sound_velocity",
                                          "shear_wave_velocity",
                                          "grueneisen_parameter",
                                          "thermal_expansivity",
                                          "molar_heat_capacity_v",
                                          "molar_heat_capacity_p",
                                          "isentropic_thermal_gradient",
                                          "P",
                                          "T",
                                          "energy",
                                          "gibbs",
                                          "helmholtz",
                                          "V",
                                          "rho",
                                          "S",
                                          "H",
                                          "K_T",
                                          "K_S",
                                          "beta_T",
                                          "beta_S",
                                          "G",
                                          "v_p",
                                          "v_phi",
                                          "v_s",
                                          "gr",
                                          "alpha",
                                          "C_v",
                                          "C_p"};
  for (const auto &name : names)
    binding.def_property_readonly(name.c_str(), [name](const Radial &self) {
      return self.get_property(name);
    });
}
} // namespace burnman::python
