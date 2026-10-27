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

#include "burnman/tools/thermobarometry.hpp"
#include "bindings.hpp"

namespace burnman::python {
namespace {
std::optional<minerals::CovarianceData>
dataset_covariance(const py::object &data) {
  if (data.is_none())
    return std::nullopt;
  const auto dictionary = data.cast<py::dict>();
  return minerals::CovarianceData{
      dictionary["endmember_names"].cast<std::vector<std::string>>(),
      dictionary["covariance_matrix"].cast<Eigen::MatrixXd>()};
}
} // namespace

void bind_thermobarometry(py::module_ &m) {
  namespace thermo = burnman::thermobarometry;
  auto module = m.def_submodule("thermobarometry",
                                "Optimal thermobarometry from mineral "
                                "compositions and their uncertainties.");
  auto result = py::class_<thermo::EstimateConditionsResult>(
      module, "EstimateConditionsResult",
      "Estimated conditions and reaction statistics. x contains pressure, "
      "temperature and free compositional coordinates. xcov, var and xcorr "
      "describe pressure and temperature only.");
#define RESULT_FIELD(name)                                                     \
  result.def_property_readonly(                                                \
      #name,                                                                   \
      [](const thermo::EstimateConditionsResult &self) { return self.name; })
  RESULT_FIELD(x);
  RESULT_FIELD(jac);
  RESULT_FIELD(fun);
  RESULT_FIELD(xcov);
  RESULT_FIELD(var);
  RESULT_FIELD(xcorr);
  RESULT_FIELD(affinities);
  RESULT_FIELD(acov);
  RESULT_FIELD(weighted_affinities);
  RESULT_FIELD(dadx);
  RESULT_FIELD(n_reactions);
  RESULT_FIELD(n_constraints);
  RESULT_FIELD(n_params);
  RESULT_FIELD(degrees_of_freedom);
  RESULT_FIELD(status);
  RESULT_FIELD(success);
  RESULT_FIELD(message);
  RESULT_FIELD(nfev);
#undef RESULT_FIELD
  result.def_property_readonly("reduced_chisqr", [](const thermo::
                                                        EstimateConditionsResult
                                                            &self) {
    if (self.degrees_of_freedom <= 0)
      throw py::attribute_error(
          "reduced_chisqr is undefined for nonpositive degrees of freedom.");
    return self.reduced_chisqr;
  });
  result.def_property_readonly(
      "fit", [](const thermo::EstimateConditionsResult &self) {
        if (self.degrees_of_freedom <= 0)
          throw py::attribute_error(
              "fit is undefined for nonpositive degrees of freedom.");
        return self.fit;
      });
  module.def("get_reaction_matrix", &thermo::get_reaction_matrix,
             py::arg("assemblage"), py::arg("small_fraction_tol") = 0.,
             "Return independent reactions in the original endmember basis. A "
             "positive "
             "small_fraction_tol selects physical endmembers greedily from the "
             "observed "
             "site occupancies and excludes small contributions.");
  module.def(
      "assemblage_set_state_from_params",
      &thermo::assemblage_set_state_from_params, py::arg("assemblage"),
      py::arg("params"),
      "Set pressure (Pa), temperature (K) and free compositional coordinates. "
      "Solutions with free_compositional_vectors require baseline_composition. "
      "Site occupancies must remain nonnegative.");
  module.def(
      "assemblage_affinity_covariance_matrix",
      [](const Assemblage &assemblage, const Eigen::MatrixXd &R,
         const py::object &data, bool state) {
        return thermo::assemblage_affinity_covariance_matrix(
            assemblage, R, dataset_covariance(data), state);
      },
      py::arg("assemblage"), py::arg("reaction_matrix"),
      py::arg("dataset_covariances") = py::none(),
      py::arg("include_state_uncertainties") = false,
      "Return reaction affinity covariances from compositional_covariances on "
      "solutions and optional dataset covariances. State uncertainties "
      "additionally "
      "require assemblage.state_covariances in pressure (Pa) and temperature "
      "(K).");
  module.def(
      "assemblage_affinity_misfit",
      [](const Assemblage &assemblage, const Eigen::MatrixXd &R,
         const py::object &data, bool state) {
        return thermo::assemblage_affinity_misfit(
            assemblage, R, dataset_covariance(data), state);
      },
      py::arg("assemblage"), py::arg("reaction_matrix"),
      py::arg("dataset_covariances") = py::none(),
      py::arg("include_state_uncertainties") = false,
      "Return the squared reaction affinity misfit weighted by the "
      "pseudoinverse "
      "of its covariance. Set the assemblage state and solution "
      "compositional_covariances first.");
  module.def("assemblage_state_misfit", &thermo::assemblage_state_misfit,
             py::arg("assemblage"),
             "Return the P-T prior misfit. Requires state_priors and "
             "state_inverse_covariances "
             "on the assemblage, using pressure in Pa and temperature in K.");
  module.def(
      "estimate_conditions",
      [](Assemblage &assemblage, const py::object &data, bool prior,
         const std::optional<Eigen::Vector2d> &guess,
         const Eigen::Vector2d &pressure_bounds,
         const Eigen::Vector2d &temperature_bounds, double tolerance,
         int max_it) {
        auto estimate = thermo::estimate_conditions(
            assemblage, dataset_covariance(data), prior, guess, pressure_bounds,
            temperature_bounds, tolerance, max_it);
        if (estimate.degrees_of_freedom <= 0 &&
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Degrees of freedom <= 0, cannot compute reduced "
                         "chi-squared and fit quality.",
                         1) < 0)
          throw py::error_already_set();
        return estimate;
      },
      py::arg("assemblage"), py::arg("dataset_covariances") = py::none(),
      py::arg("include_state_misfit") = false,
      py::arg("guessed_conditions") = Eigen::Vector2d(1.e9, 1000.),
      py::arg("pressure_bounds") = Eigen::Vector2d(1.e5, 400.e9),
      py::arg("temperature_bounds") = Eigen::Vector2d(300., 4000.),
      py::arg("small_fraction_tol") = 0., py::arg("max_it") = 100,
      R"doc(Estimate pressure and temperature by minimizing weighted reaction affinities.

Each solution requires compositional_covariances in endmember order. Optional
free_compositional_vectors contain rows of compositional changes. Their
baseline_composition is set to the composition at the start of the fit.
Dataset covariances are a dictionary with endmember_names and covariance_matrix.
P-T prior misfit additionally requires state_priors and state_inverse_covariances.

guessed_conditions is [pressure (Pa), temperature (K)]; None uses the current
assemblage state. Equal bounds fix a condition. P/T scaling uses initial weighted
affinity sensitivities and state priors. A positive small_fraction_tol reduces the reaction basis
by excluding small endmember contributions. max_it limits optimizer evaluations.

The assemblage is left at the returned conditions. Result x contains P, T and
free compositional coordinates. xcov, var and xcorr describe P and T only;
Fixed P/T conditions have zero covariance and undefined correlations.
dadx contains affinity derivatives with respect to P and T at the optimized
compositions. Reduced chi-squared and fit are available only for positive
degrees of freedom. Native Nelder-Mead and SLSQP supply success, status,
message and nfev; inspect success before using an estimate.
)doc");
}
} // namespace burnman::python
