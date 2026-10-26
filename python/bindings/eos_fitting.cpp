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
#include "burnman/utils/exceptions.hpp"

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
  using Model = optim::MineralFit;
  py::class_<Model, optim::NonLinearModel>(
      eos_fitting, "MineralFit",
      R"doc(A native P-T-property fitting model that updates the supplied mineral.

Parameters are flattened in fit_params order. Scalar parameters, Cp and numeric
coefficient arrays are supported, with the Python parameter aliases n, Debye_0,
Kprime_prime_0 and the AA/Brosh aliases. Bounds apply to the flattened vector.
Default differences are params*1e-5 + 1e-10, including their signs.
Array attributes return independent snapshots; assign a new array to change one.
)doc")
      .def(py::init<std::shared_ptr<Mineral>, const Eigen::MatrixXd &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<std::string> &,
                    const std::vector<std::string> &, const Eigen::VectorXd &,
                    const std::optional<Eigen::VectorXd> &,
                    const std::optional<Eigen::MatrixXd> &>(),
           py::arg("mineral"), py::arg("data"), py::arg("data_covariances"),
           py::arg("flags"), py::arg("fit_params"), py::arg("mle_tolerances"),
           py::arg("delta_params") = py::none(), py::arg("bounds") = py::none())
      .def_readonly("m", &Model::m)
      .def_property_readonly("fit_params",
                             [](const Model &self) { return self.fit_params; })
      .def_property_readonly("bounds",
                             [](const Model &self) { return self.bounds; })
      .def_property(
          "flags",
          [](const Model &self) {
            std::vector<std::string> flags;
            for (const auto &flag : self.flags)
              flags.push_back(std::any_cast<std::string>(flag));
            return flags;
          },
          [](Model &self, const std::vector<std::string> &flags) {
            self.flags.clear();
            for (const auto &flag : flags)
              self.flags.emplace_back(flag);
          })
      .def(
          "function",
          [](Model &self, const Eigen::VectorXd &x, const std::string &flag) {
            return self.function(x, flag);
          },
          py::arg("x"), py::arg("flag") = "V")
      .def(
          "normal",
          [](Model &self, const Eigen::VectorXd &x, const std::string &flag) {
            return self.normal(x, flag);
          },
          py::arg("x"), py::arg("flag") = "V");
  eos_fitting.def(
      "fit_PTV_data",
      [](std::shared_ptr<Mineral> mineral,
         const std::vector<std::string> &fit_params,
         const Eigen::MatrixXd &data,
         const std::vector<Eigen::MatrixXd> &data_covariances,
         const std::optional<Eigen::VectorXd> &delta_params,
         const std::optional<Eigen::MatrixXd> &bounds, double param_tolerance,
         int max_lm_iterations,
         const std::optional<Eigen::VectorXd> &param_priors,
         const std::optional<Eigen::MatrixXd> &param_prior_inv_cov_matrix,
         bool first_run_differential_evolution, const py::object &,
         bool verbose) {
        if (first_run_differential_evolution)
          throw exceptions::NotImplementedError(
              "eos_fitting", "fit_PTV_data",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        return optim::fit_PTV_data(std::move(mineral), fit_params, data,
                                   data_covariances, delta_params, bounds,
                                   options);
      },
      py::arg("mineral"), py::arg("fit_params"), py::arg("data"),
      py::arg("data_covariances") = std::vector<Eigen::MatrixXd>{},
      py::arg("delta_params") = py::none(), py::arg("bounds") = py::none(),
      py::arg("param_tolerance") = 1.e-5, py::arg("max_lm_iterations") = 100,
      py::arg("param_priors") = py::none(),
      py::arg("param_prior_inv_cov_matrix") = py::none(),
      py::arg("first_run_differential_evolution") = false,
      py::arg("differential_evolution_seed") = py::none(),
      py::arg("verbose") = true,
      R"doc(Fit mineral parameters to rows of pressure, temperature and molar volume.

Units are Pa, K and m3/mol. Supply one 3x3 observation covariance per row;
omitting covariances gives unit pressure variance with exact T and V. Default
MLE tolerances are evaluated at 1e5 Pa and 300 K. The supplied mineral is updated
in place and the returned MineralFit contains the parameters and statistics.
Bounds and Gaussian priors follow nonlinear_least_squares_fit(). Verbose output
reports LM iterations; automatic outlier diagnostics are not yet implemented.
)doc");
}
} // namespace burnman::python
