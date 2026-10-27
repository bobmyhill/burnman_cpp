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
namespace {
std::vector<optim::SolutionFitParameter>
solution_fit_parameters(const py::sequence &params) {
  std::vector<optim::SolutionFitParameter> result;
  for (const auto &item : params) {
    const auto p = py::cast<py::sequence>(item);
    if (py::len(p) != 2 && py::len(p) != 3)
      throw py::value_error("param length must be two or three");
    optim::SolutionFitParameter parameter{py::cast<std::string>(p[0]),
                                          py::cast<Eigen::Index>(p[1])};
    if (py::len(p) == 3)
      parameter.partner = py::cast<Eigen::Index>(p[2]);
    result.push_back(std::move(parameter));
  }
  return result;
}
} // namespace

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
      "fit_PTp_data",
      [](std::shared_ptr<Mineral> mineral,
         const std::vector<std::string> &fit_params,
         const optim::PropertyFlags &flags, const Eigen::MatrixXd &data,
         const std::vector<Eigen::MatrixXd> &data_covariances,
         const Eigen::VectorXd &mle_tolerances, double param_tolerance,
         const std::optional<Eigen::VectorXd> &delta_params,
         const std::optional<Eigen::MatrixXd> &bounds, int max_lm_iterations,
         const std::optional<Eigen::VectorXd> &param_priors,
         const std::optional<Eigen::MatrixXd> &param_prior_inv_cov_matrix,
         bool first_run_differential_evolution, const py::object &,
         bool verbose) {
        if (first_run_differential_evolution)
          throw exceptions::NotImplementedError(
              "eos_fitting", "fit_PTp_data",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        return optim::fit_PTp_data(std::move(mineral), fit_params, flags, data,
                                   data_covariances, mle_tolerances,
                                   delta_params, bounds, options);
      },
      py::arg("mineral"), py::arg("fit_params"), py::arg("flags"),
      py::arg("data"),
      py::arg("data_covariances") = std::vector<Eigen::MatrixXd>{},
      py::arg("mle_tolerances") = Eigen::VectorXd{},
      py::arg("param_tolerance") = 1.e-5, py::arg("delta_params") = py::none(),
      py::arg("bounds") = py::none(), py::arg("max_lm_iterations") = 100,
      py::arg("param_priors") = py::none(),
      py::arg("param_prior_inv_cov_matrix") = py::none(),
      py::arg("first_run_differential_evolution") = false,
      py::arg("differential_evolution_seed") = py::none(),
      py::arg("verbose") = true,
      R"doc(Fit mineral parameters to rows of pressure, temperature and a scalar property.

flags is a property name shared by all rows, or one name per row. A dataset
may combine properties, for example V, H, S, gibbs and alpha. Coordinates use
Pa, K and the property's SI units. Supply one 3x3 covariance per row; omission
gives unit pressure variance with exact temperature and property values.

Empty mle_tolerances select the property defaults at 1e5 Pa and 300 K; explicit
tolerances skip this state change. Updates the supplied mineral and returns a
MineralFit with parameters and fit statistics. Bounds and Gaussian priors follow
nonlinear_least_squares_fit(). Verbose output reports LM iterations and Python's
extreme-residual diagnostic when observation covariances are supplied.
Differential evolution is not yet implemented.
)doc");
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
reports LM iterations and Python's extreme-residual diagnostic when observation
covariances are supplied.
)doc");
  py::class_<optim::MineralFitV, Model>(
      eos_fitting, "MineralFitV",
      "Volume-coordinate fitting model; parameter handling follows MineralFit.")
      .def(py::init<std::shared_ptr<Mineral>, const Eigen::MatrixXd &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<std::string> &,
                    const std::vector<std::string> &, const Eigen::VectorXd &,
                    const std::optional<Eigen::VectorXd> &,
                    const std::optional<Eigen::MatrixXd> &>(),
           py::arg("mineral"), py::arg("data"), py::arg("data_covariances"),
           py::arg("flags"), py::arg("fit_params"), py::arg("mle_tolerances"),
           py::arg("delta_params") = py::none(), py::arg("bounds") = py::none())
      .def(
          "function",
          [](optim::MineralFitV &self, const Eigen::VectorXd &x,
             const std::string &flag) { return self.function(x, flag); },
          py::arg("x"), py::arg("flag") = "P")
      .def(
          "normal",
          [](optim::MineralFitV &self, const Eigen::VectorXd &x,
             const std::string &flag) { return self.normal(x, flag); },
          py::arg("x"), py::arg("flag") = "P");
  eos_fitting.def(
      "fit_VTp_data",
      [](std::shared_ptr<Mineral> mineral,
         const std::vector<std::string> &fit_params,
         const optim::PropertyFlags &flags, const Eigen::MatrixXd &data,
         const std::vector<Eigen::MatrixXd> &data_covariances,
         const Eigen::VectorXd &mle_tolerances, double param_tolerance,
         const std::optional<Eigen::VectorXd> &delta_params,
         const std::optional<Eigen::MatrixXd> &bounds, int max_lm_iterations,
         const std::optional<Eigen::VectorXd> &param_priors,
         const std::optional<Eigen::MatrixXd> &param_prior_inv_cov_matrix,
         bool first_run_differential_evolution, const py::object &,
         bool verbose) {
        if (first_run_differential_evolution)
          throw exceptions::NotImplementedError(
              "eos_fitting", "fit_VTp_data",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        return optim::fit_VTp_data(std::move(mineral), fit_params, flags, data,
                                   data_covariances, mle_tolerances,
                                   delta_params, bounds, options);
      },
      py::arg("mineral"), py::arg("fit_params"), py::arg("flags"),
      py::arg("data"),
      py::arg("data_covariances") = std::vector<Eigen::MatrixXd>{},
      py::arg("mle_tolerances") = Eigen::VectorXd{},
      py::arg("param_tolerance") = 1.e-5, py::arg("delta_params") = py::none(),
      py::arg("bounds") = py::none(), py::arg("max_lm_iterations") = 100,
      py::arg("param_priors") = py::none(),
      py::arg("param_prior_inv_cov_matrix") = py::none(),
      py::arg("first_run_differential_evolution") = false,
      py::arg("differential_evolution_seed") = py::none(),
      py::arg("verbose") = true,
      R"doc(Fit mineral parameters to rows of molar volume, temperature and a scalar property.

flags is a property name shared by all rows, or one name per row. A dataset
may combine properties, for example P, S, helmholtz and energy. Coordinates use
m3/mol, K and the property's SI units. Supply one 3x3 covariance per row; omission
gives unit property variance with exact volume and temperature.

Empty mle_tolerances select the property defaults at 1e5 Pa and 300 K; explicit
tolerances skip this state change. Updates the supplied mineral and returns a
MineralFitV with parameters and fit statistics. Bounds and Gaussian priors follow
nonlinear_least_squares_fit(). Verbose output reports LM iterations and Python's
extreme-residual diagnostic when observation covariances are supplied.
Differential evolution is not yet implemented.
)doc");
  eos_fitting.def(
      "fit_VTP_data",
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
              "eos_fitting", "fit_VTP_data",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        return optim::fit_VTP_data(std::move(mineral), fit_params, data,
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
      R"doc(Fit mineral parameters to rows of molar volume, temperature and pressure.

Units are m3/mol, K and Pa. Supply one 3x3 observation covariance per row;
omitting covariances gives unit pressure variance with exact V and T. Default
MLE tolerances are evaluated at 1e5 Pa and 300 K. The supplied mineral is updated
in place and the returned MineralFitV contains the parameters and statistics.
Bounds and Gaussian priors follow nonlinear_least_squares_fit(). Verbose output
reports LM iterations and Python's extreme-residual diagnostic when observation
covariances are supplied.
)doc");
  using SolutionModel = optim::SolutionFit;
  py::class_<SolutionModel, optim::NonLinearModel>(
      eos_fitting, "SolutionFit",
      R"doc(Fit endmember and binary interaction parameters to X-P-T-property data.

fit_params contains [name, i] for an endmember scalar or coefficient array,
or [E/S/V, i, j] for a binary interaction. Indices start at zero. Parameters
are flattened in list order, and bounds apply to the flattened vector.
Binary values use Python's W_ij*alpha_i*alpha_j/2 convention. The supplied
solution is shared and updated in place. Normal vectors follow the pin,
including zero composition components and coordinate-2 numerical differences.
)doc")
      .def(py::init([](std::shared_ptr<Solution> solution,
                       const Eigen::MatrixXd &data,
                       const std::vector<Eigen::MatrixXd> &covariances,
                       const std::vector<std::string> &flags,
                       const py::sequence &fit_params,
                       const Eigen::VectorXd &mle_tolerances,
                       const std::optional<Eigen::VectorXd> &delta_params,
                       const std::optional<Eigen::MatrixXd> &bounds) {
             return SolutionModel(std::move(solution), data, covariances, flags,
                                  solution_fit_parameters(fit_params),
                                  mle_tolerances, delta_params, bounds);
           }),
           py::arg("solution"), py::arg("data"), py::arg("data_covariances"),
           py::arg("flags"), py::arg("fit_params"), py::arg("mle_tolerances"),
           py::arg("delta_params") = py::none(), py::arg("bounds") = py::none())
      .def_readonly("m", &SolutionModel::m)
      .def_property_readonly("fit_params",
                             [](const SolutionModel &self) {
                               py::list result;
                               for (const auto &p : self.fit_params) {
                                 py::list parameter;
                                 parameter.append(p.name);
                                 parameter.append(p.endmember);
                                 if (p.partner)
                                   parameter.append(*p.partner);
                                 result.append(parameter);
                               }
                               return result;
                             })
      .def_readonly("fit_params_strings", &SolutionModel::fit_params_strings)
      .def_property_readonly(
          "bounds", [](const SolutionModel &self) { return self.bounds; })
      .def_property(
          "flags",
          [](const SolutionModel &self) {
            std::vector<std::string> flags;
            for (const auto &flag : self.flags)
              flags.push_back(std::any_cast<std::string>(flag));
            return flags;
          },
          [](SolutionModel &self, const std::vector<std::string> &flags) {
            self.flags.clear();
            for (const auto &flag : flags)
              self.flags.emplace_back(flag);
          })
      .def(
          "function",
          [](SolutionModel &self, const Eigen::VectorXd &x,
             const std::string &flag) { return self.function(x, flag); },
          py::arg("x"), py::arg("flag") = "V")
      .def(
          "normal",
          [](SolutionModel &self, const Eigen::VectorXd &x,
             const std::string &flag) { return self.normal(x, flag); },
          py::arg("x"), py::arg("flag") = "V");
  eos_fitting.def(
      "fit_XPTp_data",
      [](std::shared_ptr<Solution> solution, const py::sequence &fit_params,
         const optim::PropertyFlags &flags, const Eigen::MatrixXd &data,
         const std::vector<Eigen::MatrixXd> &data_covariances,
         const Eigen::VectorXd &mle_tolerances, double param_tolerance,
         const std::optional<Eigen::VectorXd> &delta_params,
         const std::optional<Eigen::MatrixXd> &bounds, int max_lm_iterations,
         const std::optional<Eigen::VectorXd> &param_priors,
         const std::optional<Eigen::MatrixXd> &param_prior_inv_cov_matrix,
         bool first_run_differential_evolution, const py::object &,
         bool verbose) {
        if (first_run_differential_evolution)
          throw exceptions::NotImplementedError(
              "eos_fitting", "fit_XPTp_data",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        return optim::fit_XPTp_data(std::move(solution),
                                    solution_fit_parameters(fit_params), flags,
                                    data, data_covariances, mle_tolerances,
                                    delta_params, bounds, options);
      },
      py::arg("solution"), py::arg("fit_params"), py::arg("flags"),
      py::arg("data"),
      py::arg("data_covariances") = std::vector<Eigen::MatrixXd>{},
      py::arg("mle_tolerances") = Eigen::VectorXd{},
      py::arg("param_tolerance") = 1.e-5, py::arg("delta_params") = py::none(),
      py::arg("bounds") = py::none(), py::arg("max_lm_iterations") = 100,
      py::arg("param_priors") = py::none(),
      py::arg("param_prior_inv_cov_matrix") = py::none(),
      py::arg("first_run_differential_evolution") = false,
      py::arg("differential_evolution_seed") = py::none(),
      py::arg("verbose") = true,
      R"doc(Fit solution parameters to endmember fractions, P, T and a scalar property.

fit_params contains [name, i] for an endmember parameter or [E/S/V, i, j]
for a binary interaction (i < j). flags is a shared property name or one per
row. Omitted covariances give unit pressure variance and exact remaining
coordinates. Empty mle_tolerances use the property defaults at 1e5 Pa, 300 K.
Bounds, Gaussian priors and verbose diagnostics follow fit_PTp_data().
Updates solution in place and returns a SolutionFit. Differential-evolution
initialisation is not yet implemented.
)doc");
}
} // namespace burnman::python
