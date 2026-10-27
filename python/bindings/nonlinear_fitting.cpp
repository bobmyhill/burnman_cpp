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

#include "burnman/optim/nonlinear_fitting.hpp"
#include "bindings.hpp"
#include "burnman/utils/exceptions.hpp"

namespace burnman::python {
namespace {
std::any band_flag(const py::object &flag) {
  if (flag.is_none())
    return {};
  if (py::isinstance<py::str>(flag))
    return flag.cast<std::string>();
  return flag;
}
// Keep arbitrary Python flags in the adapter; the numerical core treats them
// as opaque values and is also usable with native C++ models.
class PythonModel final : public optim::NonLinearModel {
public:
  explicit PythonModel(py::object model) : object(std::move(model)) {}

  Eigen::VectorXd get_params() const override {
    return object.attr("get_params")().cast<Eigen::VectorXd>();
  }
  void set_params(const Eigen::VectorXd &params) override {
    object.attr("set_params")(py::cast(params, py::return_value_policy::copy));
  }
  Eigen::VectorXd function(const Eigen::VectorXd &x,
                           const std::any &flag) override {
    return evaluate("function", x, flag);
  }
  Eigen::VectorXd normal(const Eigen::VectorXd &x,
                         const std::any &flag) override {
    return evaluate("normal", x, flag);
  }
  void load_data(bool differences) {
    data = object.attr("data").cast<Eigen::MatrixXd>();
    for (const auto &cov : object.attr("data_covariances"))
      data_covariances.push_back(py::cast<Eigen::MatrixXd>(cov));
    mle_tolerances = object.attr("mle_tolerances").cast<Eigen::VectorXd>();
    if (differences)
      delta_params = object.attr("delta_params").cast<Eigen::VectorXd>();
    if (py::hasattr(object, "flags")) {
      for (const auto &flag : object.attr("flags"))
        flags.emplace_back(py::reinterpret_borrow<py::object>(flag));
      if (flags.size() != static_cast<std::size_t>(data.rows()))
        throw std::invalid_argument("Require one flag per observation.");
    } else
      flags.resize(static_cast<std::size_t>(data.rows()));
  }
  void store_results() const {
    object.attr("dof") = dof;
    object.attr("n_iterations") = n_iterations;
    object.attr("data_mle") = py::cast(data_mle);
    object.attr("jacobian") = py::cast(jacobian);
    object.attr("weighted_residuals") = py::cast(weighted_residuals);
    object.attr("weights") = py::cast(weights);
    object.attr("WSS") = WSS;
    object.attr("popt") = py::cast(popt);
    object.attr("pcov") = py::cast(pcov);
    object.attr("goodness_of_fit") = goodness_of_fit;
    object.attr("noise_variance") = noise_variance;
  }
  bool flags_equal(const std::any &a, const std::any &b) const override {
    return python_flag(a).equal(python_flag(b));
  }
  void load_band_data() {
    data = object.attr("data").cast<Eigen::MatrixXd>();
    delta_params = object.attr("delta_params").cast<Eigen::VectorXd>();
    pcov = object.attr("pcov").cast<Eigen::MatrixXd>();
    dof = object.attr("dof").cast<Eigen::Index>();
    if (py::hasattr(object, "flags"))
      for (const auto &flag : object.attr("flags"))
        flags.emplace_back(band_flag(py::reinterpret_borrow<py::object>(flag)));
  }

private:
  static py::object python_flag(const std::any &flag) {
    if (!flag.has_value())
      return py::none();
    if (flag.type() == typeid(std::string))
      return py::str(std::any_cast<const std::string &>(flag));
    return std::any_cast<py::object>(flag);
  }
  Eigen::VectorXd evaluate(const char *method, const Eigen::VectorXd &x,
                           const std::any &flag) const {
    const py::object value = python_flag(flag);
    return object
        .attr(method)(py::cast(x, py::return_value_policy::copy), value)
        .cast<Eigen::VectorXd>();
  }
  py::object object;
};
} // namespace

void bind_nonlinear_fitting(py::module_ &m) {
  auto nonlinear =
      m.attr("optimize").cast<py::module_>().def_submodule("nonlinear_fitting");
  nonlinear.def(
      "extreme_values",
      [](const Eigen::VectorXd &weighted_residuals,
         double confidence_interval) {
        if (weighted_residuals.size() == 0) {
          PyErr_SetString(PyExc_ZeroDivisionError, "float division by zero");
          throw py::error_already_set();
        }
        return optim::extreme_values(weighted_residuals, confidence_interval);
      },
      py::arg("weighted_residuals"), py::arg("confidence_interval"),
      R"doc(Return the extreme-residual threshold, indices and two-tailed probabilities.

Uses Python BurnMan's normal/generalised-extreme-value approximation, intended
for more than ten observations. Residuals are standardised by their uncertainty.
The numerical calculation uses GSL. Indices retain observation order.
)doc");
  using Model = optim::NonLinearModel;
  auto native_model = py::class_<Model>(nonlinear, "_NonLinearModel");
  native_model.def("get_params", &Model::get_params)
      .def("set_params", &Model::set_params, py::arg("param_values"))
      .def("validate", &Model::validate);
  // Return snapshots, as for the Python-model adapter's result attributes.
#define ATTRIBUTE(name)                                                        \
  native_model.def_property(                                                   \
      #name, [](const Model &self) { return self.name; },                      \
      [](Model &self, const decltype(Model::name) &value) {                    \
        self.name = value;                                                     \
      })
  ATTRIBUTE(data);
  ATTRIBUTE(data_covariances);
  ATTRIBUTE(mle_tolerances);
  ATTRIBUTE(delta_params);
  ATTRIBUTE(dof);
  ATTRIBUTE(n_iterations);
  ATTRIBUTE(data_mle);
  ATTRIBUTE(jacobian);
  ATTRIBUTE(weighted_residuals);
  ATTRIBUTE(weights);
  ATTRIBUTE(WSS);
  ATTRIBUTE(popt);
  ATTRIBUTE(pcov);
  ATTRIBUTE(goodness_of_fit);
  ATTRIBUTE(noise_variance);
#undef ATTRIBUTE
  nonlinear.def(
      "confidence_prediction_bands",
      [](py::object model, const Eigen::MatrixXd &x_array,
         double confidence_interval, const py::function &f,
         const py::object &flag) {
        const optim::ScalarModelFunction function = [&](const Eigen::VectorXd
                                                            &x) {
          return f(py::cast(x, py::return_value_policy::copy)).cast<double>();
        };
        const std::any mode = band_flag(flag);
        if (py::isinstance<Model>(model))
          return optim::confidence_prediction_bands(
              model.cast<Model &>(), x_array, confidence_interval, function,
              mode);
        PythonModel adapter(std::move(model));
        adapter.load_band_data();
        return optim::confidence_prediction_bands(
            adapter, x_array, confidence_interval, function, mode);
      },
      py::arg("model"), py::arg("x_array"), py::arg("confidence_interval"),
      py::arg("f"), py::arg("flag") = py::none(),
      R"doc(Return pointwise confidence lower/upper and prediction lower/upper rows.

The centre is f(model.function(x, flag)). Parameter uncertainty uses forward
finite differences and model.pcov, with a positive Student-t quantile and
model.dof degrees of freedom. Parameters are restored on success or error.
Prediction noise uses matching response residuals divided by their residual
degrees of freedom, converted to output units using the existing parameter
trials. Prediction rows are NaN when trials do not identify a proportional
response-to-output conversion or there are insufficient matching observations.
These follow Python's pointwise approximation. Empty queries return shape (4, 0).
)doc");
  nonlinear.def("abs_line_project", &optim::abs_line_project, py::arg("M"),
                py::arg("n"),
                "Project a covariance onto the unit direction n.");
  nonlinear.def(
      "mle_estimate",
      [](py::object model, const Eigen::VectorXd &x, const Eigen::VectorXd &x_m,
         const Eigen::MatrixXd &cov, const py::object &flag) {
        if (py::isinstance<Model>(model)) {
          const std::any native_flag =
              flag.is_none() ? std::any{} : std::any(flag.cast<std::string>());
          return optim::mle_estimate(model.cast<Model &>(), x, x_m, cov,
                                     native_flag);
        }
        PythonModel adapter(std::move(model));
        return optim::mle_estimate(adapter, x, x_m, cov, flag);
      },
      py::arg("model"), py::arg("x"), py::arg("x_m"), py::arg("cov"),
      py::arg("flag") = py::none(),
      "Return the tangent-plane MLE position, residual and normal variance.");
  nonlinear.def(
      "find_mle",
      [](py::object model) {
        if (py::isinstance<Model>(model))
          return optim::find_mle(model.cast<Model &>());
        PythonModel adapter(std::move(model));
        adapter.load_data(false);
        return optim::find_mle(adapter);
      },
      py::arg("model"),
      "Return MLE positions, standardised residuals and inverse variances.");
  nonlinear.def(
      "calculate_jacobian",
      [](py::object model) {
        if (py::isinstance<Model>(model)) {
          optim::calculate_jacobian(model.cast<Model &>());
          return;
        }
        PythonModel adapter(model);
        adapter.load_data(true);
        optim::calculate_jacobian(adapter);
        model.attr("jacobian") = py::cast(adapter.jacobian);
      },
      py::arg("model"),
      "Set model.jacobian using central differences; restore its parameters.");
  nonlinear.def(
      "nonlinear_least_squares_fit",
      [](py::object model, double lm_damping, double param_tolerance,
         int max_lm_iterations,
         const std::optional<Eigen::VectorXd> &param_priors,
         const std::optional<Eigen::MatrixXd> &param_prior_inv_cov_matrix,
         bool first_run_differential_evolution, const py::object &,
         bool verbose) {
        if (first_run_differential_evolution)
          throw exceptions::NotImplementedError(
              "nonlinear_fitting", "nonlinear_least_squares_fit",
              "Differential evolution is not yet implemented.");
        optim::NonlinearFitOptions options;
        options.lm_damping = lm_damping;
        options.param_tolerance = param_tolerance;
        options.max_lm_iterations = max_lm_iterations;
        options.param_priors = param_priors;
        options.param_prior_inv_cov_matrix = param_prior_inv_cov_matrix;
        options.verbose = verbose;
        if (py::isinstance<Model>(model)) {
          optim::nonlinear_least_squares_fit(model.cast<Model &>(), options);
          return;
        }
        if (py::hasattr(model, "validate"))
          model.attr("validate")();
        PythonModel adapter(model);
        adapter.load_data(true);
        if (!py::hasattr(model, "flags")) {
          py::list flags;
          for (Eigen::Index i = 0; i < adapter.data.rows(); ++i)
            flags.append(py::none());
          model.attr("flags") = flags;
        }
        optim::nonlinear_least_squares_fit(adapter, options);
        adapter.store_results();
      },
      py::arg("model"), py::arg("lm_damping") = 0.,
      py::arg("param_tolerance") = 1.e-7, py::arg("max_lm_iterations") = 100,
      py::arg("param_priors") = py::none(),
      py::arg("param_prior_inv_cov_matrix") = py::none(),
      py::arg("first_run_differential_evolution") = false,
      py::arg("differential_evolution_seed") = 42, py::arg("verbose") = false,
      R"doc(Fit an errors-in-variables model with adaptive LM damping.

The model supplies get_params(), set_params(), function(x, flag), normal(x,
flag), data, data_covariances, mle_tolerances and delta_params. normal() must
return a unit vector. Optional per-observation flags may be arbitrary objects;
missing flags default to None. set_params() may enforce bounds.

Updates the model parameters, data_mle, weighted_residuals, weights, jacobian,
WSS, popt, pcov, dof, goodness_of_fit and noise_variance. n_iterations counts
accepted LM updates. Gaussian priors require both means and a positive
semidefinite precision matrix. WSS includes their penalty, but dof remains
n_data - n_params and noise_variance uses only data residuals.

Requires n_data > n_params and nonsingular fitting normal matrices. MLE
projection fails after 10000 iterations rather than looping indefinitely.
Differential-evolution initialisation is not yet implemented.
)doc");
}
} // namespace burnman::python
