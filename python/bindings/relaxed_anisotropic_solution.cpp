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

#include "burnman/core/relaxed_anisotropic_solution.hpp"
#include "bindings.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
void bind_relaxed_anisotropic_solution(py::module_ &m) {
  auto solution = py::class_<RelaxedAnisotropicSolution, AnisotropicSolution,
                             py::smart_holder>(m, "RelaxedAnisotropicSolution");
  solution
      .def(py::init<std::shared_ptr<AnisotropicSolution>,
                    const Eigen::MatrixXd &, const Eigen::MatrixXd &>(),
           py::arg("anisotropic_solution"), py::arg("relaxation_vectors"),
           py::arg("unrelaxed_vectors"), py::keep_alive<1, 2>())
      .def(py::init([](const RelaxedAnisotropicSolution &other) {
        return std::dynamic_pointer_cast<RelaxedAnisotropicSolution>(
            other.copy());
      }))
      .def(
          "set_composition",
          [](RelaxedAnisotropicSolution &self, const Eigen::ArrayXd &fractions,
             const std::optional<Eigen::VectorXd> &q,
             bool relaxed) { self.set_composition(fractions, q, relaxed); },
          py::arg("molar_fractions"), py::arg("q_initial") = py::none(),
          py::arg("relaxed") = true)
      .def(
          "set_state",
          [](RelaxedAnisotropicSolution &self, double p, double t,
             bool relaxed) {
            check_pt(p, t);
            self.set_state(p, t, relaxed);
          },
          py::arg("pressure"), py::arg("temperature"),
          py::arg("relaxed") = true)
      .def("set_state_with_volume",
           &RelaxedAnisotropicSolution::set_state_with_volume,
           py::arg("volume"), py::arg("temperature"),
           py::arg("pressure_guesses") = Eigen::Vector2d(0., 1.e10))
      .def("_relax_at_PTX", &RelaxedAnisotropicSolution::relax_at_PTX)
      .def_property_readonly("unrelaxed",
                             &RelaxedAnisotropicSolution::get_unrelaxed)
      .def_property_readonly("n_relaxation_vectors",
                             [](const RelaxedAnisotropicSolution &self) {
                               return self.get_dndq().cols();
                             })
      .def_property_readonly("n_unrelaxed_vectors",
                             [](const RelaxedAnisotropicSolution &self) {
                               return self.get_dndx().cols();
                             })
      .def_property("q_initial", &RelaxedAnisotropicSolution::get_q_initial,
                    &RelaxedAnisotropicSolution::set_q_initial,
                    py::return_value_policy::copy);
#define PROPERTY(name)                                                         \
  bind_property<RelaxedAnisotropicSolution>(                                   \
      solution, #name, "get_" #name, &RelaxedAnisotropicSolution::get_##name)
  PROPERTY(dndq);
  PROPERTY(dndx);
  PROPERTY(unrelaxed_vectors);
  PROPERTY(dqdz_relaxed);
#undef PROPERTY
  auto diagnostic = [&](const char *name, auto getter) {
    solution.def_property_readonly(
        name,
        [getter](const RelaxedAnisotropicSolution &self) {
          return (self.*getter)();
        },
        py::return_value_policy::copy);
  };
  diagnostic("_depsdT_fixed_volume",
             &RelaxedAnisotropicSolution::get_depsdT_fixed_volume);
  diagnostic("_dSdq", &RelaxedAnisotropicSolution::get_dSdq);
  diagnostic("_dPdq", &RelaxedAnisotropicSolution::get_dPdq);
  diagnostic("_d2Fdqdq_fixed_VT",
             &RelaxedAnisotropicSolution::get_structural_hessian_fixed_VT);
  diagnostic("_d2Fdqdq_fixed_epsT",
             &RelaxedAnisotropicSolution::get_structural_hessian_fixed_epsT);
  diagnostic("_d2Fdqdq_fixed_epsT_pinv",
             &RelaxedAnisotropicSolution::get_susceptibility);
  diagnostic("_d2Fdqdz", &RelaxedAnisotropicSolution::get_coupling);
  diagnostic("_d2Fdzdz_Q", &RelaxedAnisotropicSolution::get_frozen_hessian);
  diagnostic("_d2Fdzdz", &RelaxedAnisotropicSolution::get_relaxed_hessian);
  solution.def_property_readonly(
      "_depsdq_fixed_VT", [](const RelaxedAnisotropicSolution &self) {
        const auto &tensors = self.get_depsdq_fixed_VT();
        py::array_t<double> result({py::ssize_t(3), py::ssize_t(3),
                                    static_cast<py::ssize_t>(tensors.size())});
        auto out = result.mutable_unchecked<3>();
        for (std::size_t q = 0; q < tensors.size(); ++q)
          for (Eigen::Index i = 0; i < 3; ++i)
            for (Eigen::Index j = 0; j < 3; ++j)
              out(i, j, q) = tensors[q](i, j);
        return result;
      });
  solution.def_property_readonly("_dbarepsdeps",
                                 [](const RelaxedAnisotropicSolution &self) {
                                   const auto tensor = self.get_dbarepsdeps();
                                   py::array_t<double> result({3, 3, 3, 3});
                                   auto out = result.mutable_unchecked<4>();
                                   for (std::size_t i = 0; i < 3; ++i)
                                     for (std::size_t j = 0; j < 3; ++j)
                                       for (std::size_t k = 0; k < 3; ++k)
                                         for (std::size_t l = 0; l < 3; ++l)
                                           out(i, j, k, l) = tensor[i][j][k][l];
                                   return result;
                                 });
}
} // namespace burnman::python
