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

#include "burnman/core/anisotropic_mineral.hpp"
#include "bindings.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
types::NDArray<double> coefficients(const Array &values) {
  std::vector<std::size_t> shape;
  for (py::ssize_t i = 0; i < values.ndim(); ++i)
    shape.push_back(static_cast<std::size_t>(values.shape(i)));
  types::NDArray<double> result(shape);
  std::copy(values.data(), values.data() + values.size(),
            result.data().begin());
  return result;
}
py::array_t<double> full_tensor(const types::ElasticTensor &tensor) {
  py::array_t<double> result({3, 3, 3, 3});
  auto out = result.mutable_unchecked<4>();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        for (std::size_t l = 0; l < 3; ++l)
          out(i, j, k, l) = tensor[i][j][k][l];
  return result;
}
} // namespace

void bind_anisotropic_mineral(py::module_ &m) {
  auto mineral = py::class_<AnisotropicMineral, Mineral, AnisotropicMaterial,
                            py::smart_holder>(m, "AnisotropicMineral");
  mineral
      .def(py::init([](std::shared_ptr<Mineral> reference,
                       const utils::CellParameters &cell, const Array &values,
                       std::optional<utils::FrameConvention> frame) {
             return std::make_shared<AnisotropicMineral>(
                 reference, cell, coefficients(values), frame);
           }),
           py::arg("isotropic_mineral"), py::arg("cell_parameters"),
           py::arg("anisotropic_parameters"),
           py::arg("frame_convention") = py::none())
      .def(py::init<std::shared_ptr<Mineral>, const utils::CellParameters &,
                    PsiFunction, bool, std::optional<utils::FrameConvention>>(),
           py::arg("isotropic_mineral"), py::arg("cell_parameters"),
           py::arg("psi_function"), py::arg("orthotropic"),
           py::arg("frame_convention") = py::none());
#define PROPERTY(name)                                                         \
  bind_property<AnisotropicMineral>(mineral, #name, "get_" #name,              \
                                    &AnisotropicMineral::get_##name, true)
  PROPERTY(density);
  PROPERTY(isentropic_bulk_modulus_reuss);
  PROPERTY(deformation_gradient_tensor);
  PROPERTY(unrotated_cell_vectors);
  PROPERTY(rotation_matrix);
  PROPERTY(cell_vectors);
  PROPERTY(cell_parameters);
  PROPERTY(isothermal_compliance_tensor);
  PROPERTY(isothermal_stiffness_tensor);
  PROPERTY(isentropic_compliance_tensor);
  PROPERTY(isentropic_stiffness_tensor);
  PROPERTY(thermal_expansivity_tensor);
  PROPERTY(isothermal_compressibility_tensor);
  PROPERTY(isentropic_compressibility_tensor);
  PROPERTY(grueneisen_tensor);
  PROPERTY(thermal_stress_tensor);
  PROPERTY(molar_isometric_heat_capacity);
  PROPERTY(isothermal_bulk_modulus_voigt);
  PROPERTY(isothermal_compressibility_voigt);
  PROPERTY(isentropic_compressibility_voigt);
#undef PROPERTY
  mineral
      .def_property_readonly("orthotropic",
                             &AnisotropicMineral::get_orthotropic)
      .def_property_readonly("isotropic_mineral",
                             &AnisotropicMineral::get_isotropic_mineral)
      .def_property_readonly("frame_convention",
                             &AnisotropicMineral::get_frame_convention)
      .def_property_readonly("cell_vectors_0",
                             &AnisotropicMineral::get_cell_vectors_0,
                             py::return_value_policy::copy)
      .def_property_readonly("Pth", &AnisotropicMineral::get_thermal_pressure)
      .def_property_readonly("dPthdf", &AnisotropicMineral::get_dPthdf)
      .def_property_readonly(
          "_unrotated_isothermal_stiffness_tensor",
          &AnisotropicMineral::get_unrotated_isothermal_stiffness_tensor,
          py::return_value_policy::copy)
      .def("evaluate_psi", &AnisotropicMineral::evaluate_psi, py::arg("f"),
           py::arg("Pth"));
  auto bind_full = [&](const char *name, auto getter) {
    auto call = [getter](const AnisotropicMineral &self) {
      check_state(self);
      return full_tensor((self.*getter)());
    };
    mineral.def_property_readonly(name, call)
        .def((std::string("get_") + name).c_str(), call);
  };
  bind_full("full_isothermal_compliance_tensor",
            &AnisotropicMineral::get_full_isothermal_compliance_tensor);
  bind_full("full_isothermal_stiffness_tensor",
            &AnisotropicMineral::get_full_isothermal_stiffness_tensor);
  bind_full("full_isentropic_compliance_tensor",
            &AnisotropicMineral::get_full_isentropic_compliance_tensor);
  bind_full("full_isentropic_stiffness_tensor",
            &AnisotropicMineral::get_full_isentropic_stiffness_tensor);
  m.def("_standard_anisotropic_psi",
        [](double f, double Pth, const Array &values) {
          return AnisotropicMineral::standard_psi_function(
              f, Pth, coefficients(values));
        });
  m.def("_check_anisotropic_parameters", [](const Array &values) {
    return AnisotropicMineral::check_standard_parameters(coefficients(values));
  });
  m.def("convert_f_Pth_to_f_T_derivatives", &convert_f_Pth_to_f_T_derivatives,
        py::arg("dPsidf_Pth_Voigt"), py::arg("dPsidPth_f_Voigt"),
        py::arg("aK_T"), py::arg("dPthdf_T"));
  m.def("deformation_gradient_alpha_and_compliance",
        &deformation_gradient_alpha_and_compliance, py::arg("alpha_V"),
        py::arg("beta_TR"), py::arg("PsiI"), py::arg("dPsidf_T_Voigt"),
        py::arg("dPsiIdf_T"), py::arg("dPsiIdT_f"));
}
} // namespace burnman::python
