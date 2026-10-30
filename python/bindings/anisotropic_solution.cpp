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

#include "burnman/core/anisotropic_solution.hpp"
#include "bindings.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
py::array_t<double> pack(const CompositionTensors &tensors) {
  py::array_t<double> result({py::ssize_t(3), py::ssize_t(3),
                              static_cast<py::ssize_t>(tensors.size())});
  auto out = result.mutable_unchecked<3>();
  for (std::size_t k = 0; k < tensors.size(); ++k)
    for (Eigen::Index i = 0; i < 3; ++i)
      for (Eigen::Index j = 0; j < 3; ++j)
        out(i, j, k) = tensors[k](i, j);
  return result;
}
} // namespace

void bind_anisotropic_solution(py::module_ &m) {
  auto solution = py::class_<AnisotropicSolution, Solution, AnisotropicMineral,
                             py::smart_holder>(m, "AnisotropicSolution");
  solution.def(
      py::init([](const std::string &name,
                  const std::shared_ptr<solution_models::SolutionModel> &model,
                  py::function function,
                  const std::optional<Eigen::ArrayXd> &fractions) {
        PsiExcessFunction excess =
            [function = std::move(function)](
                double lnV, double Pth,
                const Eigen::ArrayXd &p) -> PsiExcessDerivatives {
          py::gil_scoped_acquire gil;
          const auto output = function(lnV, Pth, p).cast<py::tuple>();
          if (output.size() != 4)
            throw std::invalid_argument(
                "Psi excess must return three 6x6 matrices and a "
                "(3,3,n_endmembers) array.");
          const auto gradient =
              output[3].cast<py::array_t<double, py::array::forcecast>>();
          if (gradient.ndim() != 3 || gradient.shape(0) != 3 ||
              gradient.shape(1) != 3 || gradient.shape(2) != p.size())
            throw std::invalid_argument("Psi composition derivatives require "
                                        "shape (3,3,n_endmembers).");
          const auto values = gradient.unchecked<3>();
          CompositionTensors derivatives(static_cast<std::size_t>(p.size()));
          for (Eigen::Index k = 0; k < p.size(); ++k)
            for (Eigen::Index i = 0; i < 3; ++i)
              for (Eigen::Index j = 0; j < 3; ++j)
                derivatives[static_cast<std::size_t>(k)](i, j) =
                    values(i, j, k);
          return {output[0].cast<types::Matrix6d>(),
                  output[1].cast<types::Matrix6d>(),
                  output[2].cast<types::Matrix6d>(), std::move(derivatives)};
        };
        return std::make_shared<AnisotropicSolution>(
            name, model, std::move(excess), fractions);
      }),
      py::arg("name"), py::arg("solution_model"),
      py::arg("psi_excess_function"), py::arg("molar_fractions") = py::none());
  solution
      .def("set_state_with_volume", &AnisotropicSolution::set_state_with_volume,
           py::arg("volume"), py::arg("temperature"),
           py::arg("pressure_guesses") = Eigen::Vector2d(0., 1.e10))
      .def("compute_base_properties",
           &AnisotropicSolution::compute_base_properties)
      .def_readwrite("T_0", &AnisotropicSolution::T_0)
      .def_property_readonly("_scalar_solution",
                             &AnisotropicSolution::get_scalar_solution)
      .def_property_readonly("_f",
                             [](const AnisotropicSolution &self) {
                               check_state(self);
                               return std::log(self.get_molar_volume());
                             })
      .def_property_readonly(
          "ones",
          [](const AnisotropicSolution &self) {
            return Eigen::ArrayXd::Ones(self.get_n_endmembers()).eval();
          })
      .def_property_readonly("eye",
                             [](const AnisotropicSolution &self) {
                               return Eigen::MatrixXd::Identity(
                                          self.get_n_endmembers(),
                                          self.get_n_endmembers())
                                   .eval();
                             })
      .def_property_readonly("endmembers", [](const AnisotropicSolution &self) {
        py::list result;
        const auto &endmembers = self.get_endmembers();
        const auto &formulas = self.get_solution_model()->get_formulas();
        for (std::size_t i = 0; i < endmembers.size(); ++i)
          result.append(py::make_tuple(endmembers[i], formulas[i]));
        return result;
      });
  auto bind_tensor = [&](const char *name, auto getter) {
    solution.def_property_readonly(name,
                                   [getter](const AnisotropicSolution &self) {
                                     return pack((self.*getter)());
                                   });
  };
  bind_tensor("_logm_M0_mbr", &AnisotropicSolution::get_logm_M0_mbr);
  bind_tensor("_PsiI_mbr", &AnisotropicSolution::get_PsiI_mbr);
  bind_tensor("_dPsiIdp_xs", &AnisotropicSolution::get_dPsiIdp_excess);
  bind_tensor("_dM0dp_fixed_VT", &AnisotropicSolution::get_dM0dp_fixed_VT);
  bind_tensor("_unrotated_dFdp_fixed_VT",
              &AnisotropicSolution::get_unrotated_dFdp_fixed_VT);
  bind_tensor("_unrotated_dMdn_fixed_VT",
              &AnisotropicSolution::get_unrotated_dMdn_fixed_VT);
  bind_tensor("_dMdn_fixed_VT", &AnisotropicSolution::get_dMdn_fixed_VT);
  bind_tensor("depsdn_fixed_VT", &AnisotropicSolution::get_depsdn_fixed_VT);
}
} // namespace burnman::python
