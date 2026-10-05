#include "bindings.hpp"
#include "burnman/core/solution.hpp"
#include "burnman/tools/equilibration/equality_constraints.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"

namespace burnman::python {
namespace {
void check_phases(const Assemblage &assemblage) {
  for (Eigen::Index i = 0; i < assemblage.get_n_phases(); ++i) {
    const auto phase = assemblage.get_phase(static_cast<std::size_t>(i));
    if (!dynamic_cast<const Mineral *>(phase.get()) &&
        !dynamic_cast<const Solution *>(phase.get())) {
      throw std::invalid_argument("Equilibration requires a flat assemblage of "
                                  "Mineral and Solution phases.");
    }
  }
}

void check_scalar(double value, const char *name, bool positive = false) {
  if (!std::isfinite(value) || (positive && value <= 0.0)) {
    throw std::invalid_argument(std::string(name) + " must be finite" +
                                (positive ? " and positive." : "."));
  }
}

void check_composition(
    const Assemblage &assemblage, const types::FormulaMap &composition,
    const std::vector<equilibration::FreeVectorMap> &free_vectors) {
  check_phases(assemblage);
  double total = 0.0;
  const auto &elements = assemblage.get_elements();
  for (const auto &[element, amount] : composition) {
    if (!std::isfinite(amount) || amount < 0.0)
      throw std::invalid_argument(
          "Bulk composition must be finite and nonnegative.");
    if (std::find(elements.begin(), elements.end(), element) ==
        elements.end()) {
      throw std::invalid_argument(
          "Bulk composition contains an element absent from the assemblage: " +
          element);
    }
    total += amount;
  }
  if (total <= 0.0)
    throw std::invalid_argument(
        "Bulk composition must have a positive total amount.");
  for (const auto &vec : free_vectors) {
    for (const auto &[element, amount] : vec) {
      check_scalar(amount, "Free compositional vector values");
      if (std::find(elements.begin(), elements.end(), element) ==
          elements.end()) {
        throw std::invalid_argument(
            "Free compositional vector contains an unknown element: " +
            element);
      }
    }
  }
}
} // namespace

void bind_equilibration(py::module_ &m) {
  using namespace equilibration;
  using optim::roots::DampedNewtonResult;
  py::class_<EqualityConstraint, std::shared_ptr<EqualityConstraint>>(
      m, "EqualityConstraint");
#define SCALAR_CONSTRAINT(type, positive)                                      \
  py::class_<type, EqualityConstraint, std::shared_ptr<type>>(m, #type).def(   \
      py::init([](double value) {                                              \
        check_scalar(value, #type, positive);                                  \
        return std::make_shared<type>(value);                                  \
      }),                                                                      \
      py::arg("value"))
  SCALAR_CONSTRAINT(PressureConstraint, false);
  SCALAR_CONSTRAINT(TemperatureConstraint, true);
  SCALAR_CONSTRAINT(EntropyConstraint, false);
  SCALAR_CONSTRAINT(VolumeConstraint, true);
#undef SCALAR_CONSTRAINT
  py::class_<PTEllipseConstraint, EqualityConstraint,
             std::shared_ptr<PTEllipseConstraint>>(m, "PTEllipseConstraint")
      .def(py::init([](const Eigen::Vector2d &centre,
                       const Eigen::Vector2d &scaling) {
             if (!centre.allFinite() || !scaling.allFinite() ||
                 (scaling.array() <= 0.0).any()) {
               throw std::invalid_argument(
                   "Ellipse centre must be finite and both scales must be "
                   "positive and finite.");
             }
             return std::make_shared<PTEllipseConstraint>(centre, scaling);
           }),
           py::arg("centre"), py::arg("scaling"));
  py::class_<LinearXConstraint, EqualityConstraint,
             std::shared_ptr<LinearXConstraint>>(m, "LinearXConstraint")
      .def(py::init([](const Eigen::VectorXd &A, double b) {
             if (!A.allFinite() || A.size() == 0)
               throw std::invalid_argument(
                   "A must be a nonempty finite vector.");
             check_scalar(b, "b");
             return std::make_shared<LinearXConstraint>(A, b);
           }),
           py::arg("A"), py::arg("b"));
  py::class_<PhaseFractionConstraint, LinearXConstraint,
             std::shared_ptr<PhaseFractionConstraint>>(
      m, "PhaseFractionConstraint")
      .def(py::init([](Eigen::Index phase_index, double fraction,
                       const EquilibrationParameters &parameters) {
             if (phase_index < 0 ||
                 phase_index >= parameters.phase_amount_indices.size())
               throw py::index_error("Phase index out of range.");
             if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
               throw std::invalid_argument(
                   "Phase fraction must be between zero and one.");
             return std::make_shared<PhaseFractionConstraint>(
                 phase_index, fraction, parameters);
           }),
           py::arg("phase_index"), py::arg("phase_fraction"),
           py::arg("parameters"));
  py::class_<PhaseCompositionConstraint, LinearXConstraint,
             std::shared_ptr<PhaseCompositionConstraint>>(
      m, "PhaseCompositionConstraint")
      .def(py::init([](Eigen::Index phase_index,
                       const std::vector<std::string> &sites,
                       const Eigen::VectorXd &numerator,
                       const Eigen::VectorXd &denominator, double value,
                       const Assemblage &assemblage,
                       const EquilibrationParameters &parameters) {
             if (phase_index < 0 || phase_index >= assemblage.get_n_phases())
               throw py::index_error("Phase index out of range.");
             if (sites.empty() ||
                 numerator.size() != static_cast<Eigen::Index>(sites.size()) ||
                 denominator.size() != numerator.size() ||
                 !numerator.allFinite() || !denominator.allFinite()) {
               throw std::invalid_argument(
                   "site_names, numerator and denominator must have matching "
                   "nonzero lengths and finite values.");
             }
             check_scalar(value, "Phase composition value");
             return std::make_shared<PhaseCompositionConstraint>(
                 phase_index, sites, numerator, denominator, value, assemblage,
                 parameters);
           }),
           py::arg("phase_index"), py::arg("site_names"), py::arg("numerator"),
           py::arg("denominator"), py::arg("value"), py::arg("assemblage"),
           py::arg("parameters"));

  auto parameters_class =
      py::class_<EquilibrationParameters>(m, "EquilibrationParameters");
#define PARAMETER(name)                                                        \
  parameters_class.def_property_readonly(                                      \
      #name, [](const EquilibrationParameters &self) { return self.name; })
  PARAMETER(parameter_names);
  PARAMETER(bulk_composition_vector);
  PARAMETER(reduced_composition_vector);
  PARAMETER(free_compositional_vectors);
  PARAMETER(reduced_free_composition_vectors);
  PARAMETER(constraint_vector);
  PARAMETER(constraint_matrix);
  PARAMETER(phase_amount_indices);
  PARAMETER(n_parameters);
#undef PARAMETER
  py::class_<DampedNewtonResult::Iterates>(m, "IterationHistory")
      .def_property_readonly(
          "x", [](const DampedNewtonResult::Iterates &self) { return self.x; })
      .def_property_readonly(
          "F", [](const DampedNewtonResult::Iterates &self) { return self.F; })
      .def_property_readonly(
          "lambdas",
          [](const DampedNewtonResult::Iterates &self) { return self.lambda; });
  auto solver_result = py::class_<DampedNewtonResult>(m, "DampedNewtonResult");
#define RESULT_FIELD(name)                                                     \
  solver_result.def_property_readonly(                                         \
      #name, [](const DampedNewtonResult &self) { return self.name; })
  RESULT_FIELD(x);
  RESULT_FIELD(F);
  RESULT_FIELD(J);
  RESULT_FIELD(F_norm);
  RESULT_FIELD(n_iterations);
  RESULT_FIELD(success);
  RESULT_FIELD(code);
  RESULT_FIELD(message);
  RESULT_FIELD(iteration_history);
#undef RESULT_FIELD
  py::class_<EquilibrateResult>(m, "EquilibrateResult")
      .def_property_readonly(
          "prm", [](const EquilibrateResult &self) { return self.prm; })
      .def_property_readonly("sol_array", [](const EquilibrateResult &self) {
        // NumPy owns solver-result objects, so the grid remains valid after
        // self is deleted.
        py::module_ numpy = py::module_::import("numpy");
        py::object flat = numpy.attr("array")(py::cast(self.sol_array.data()),
                                              py::arg("dtype") = "object");
        return flat.attr("reshape")(
            py::tuple(py::cast(self.sol_array.shape())));
      });

  m.def(
      "get_equilibration_parameters",
      [](const Assemblage &assemblage, const types::FormulaMap &composition,
         const std::vector<FreeVectorMap> &free_vectors) {
        check_composition(assemblage, composition, free_vectors);
        return get_equilibration_parameters(assemblage, composition,
                                            free_vectors);
      },
      py::arg("assemblage"), py::arg("composition"),
      py::arg("free_compositional_vectors") = std::vector<FreeVectorMap>{});
  m.def(
      "get_parameter_vector",
      [](const Assemblage &assemblage, int n_free) {
        check_phases(assemblage);
        if (n_free < 0)
          throw std::invalid_argument(
              "n_free_compositional_vectors cannot be negative.");
        return get_parameter_vector(assemblage, n_free);
      },
      py::arg("assemblage"), py::arg("n_free_compositional_vectors") = 0);
  m.def(
      "get_endmember_amounts",
      [](const Assemblage &assemblage) {
        check_phases(assemblage);
        return get_endmember_amounts(assemblage);
      },
      py::arg("assemblage"));
  m.def(
      "set_composition_and_state_from_parameters",
      [](Assemblage &assemblage, const Eigen::VectorXd &values) {
        check_phases(assemblage);
        const Eigen::Index required = 2 + assemblage.get_n_endmembers();
        if (values.size() < required || !values.allFinite())
          throw std::invalid_argument(
              "Parameter vector is too short or contains nonfinite values.");
        check_pt(values(0), values(1));
        Eigen::Index offset = 2;
        double total = 0.0;
        for (Eigen::Index i = 0; i < assemblage.get_n_phases(); ++i) {
          if (values(offset) < -1.0e-8)
            throw std::invalid_argument("Phase amounts must be nonnegative.");
          total += values(offset);
          const auto phase = assemblage.get_phase(static_cast<std::size_t>(i));
          if (const auto *solution =
                  dynamic_cast<const Solution *>(phase.get())) {
            const Eigen::Index n = solution->get_n_endmembers();
            Eigen::ArrayXd fractions(n);
            fractions.tail(n - 1) = values.segment(offset + 1, n - 1);
            fractions(0) = 1.0 - fractions.tail(n - 1).sum();
            check_site_fractions(fractions,
                                 solution->get_endmember_occupancies());
            offset += n;
          } else {
            ++offset;
          }
        }
        if (total <= 0.0)
          throw std::invalid_argument(
              "Phase amounts must have a positive sum.");
        set_composition_and_state_from_parameters(assemblage, values);
      },
      py::arg("assemblage"), py::arg("parameters"));

  m.def(
      "_equilibrate",
      [](const types::FormulaMap &composition, Assemblage &assemblage,
         const std::vector<std::vector<std::shared_ptr<EqualityConstraint>>>
             &groups,
         const std::vector<FreeVectorMap> &free_vectors,
         const py::object &tolerance, bool store_iterates, int max_iterations,
         bool verbose) {
        check_composition(assemblage, composition, free_vectors);
        if (max_iterations <= 0)
          throw std::invalid_argument("max_iterations must be positive.");
        if (groups.size() != free_vectors.size() + 2)
          throw std::invalid_argument(
              "Exactly 2 + len(free_compositional_vectors) constraint groups "
              "are required.");
        const auto parameters =
            get_equilibration_parameters(assemblage, composition, free_vectors);
        double tol = 1.0e-3;
        Eigen::VectorXd parameter_tolerances;
        if (py::isinstance<py::float_>(tolerance) ||
            py::isinstance<py::int_>(tolerance)) {
          tol = py::cast<double>(tolerance);
          check_scalar(tol, "tol", true);
        } else {
          parameter_tolerances = py::cast<Eigen::VectorXd>(tolerance);
          if (parameter_tolerances.size() != parameters.n_parameters ||
              !parameter_tolerances.allFinite() ||
              (parameter_tolerances.array() <= 0.0).any()) {
            throw std::invalid_argument(
                "tol must be a positive scalar or a finite positive vector of "
                "length n_parameters.");
          }
        }
        const Eigen::VectorXd zero =
            Eigen::VectorXd::Zero(parameters.n_parameters);
        ConstraintList constraints;
        for (const auto &group : groups) {
          if (group.empty())
            throw std::invalid_argument("Constraint groups cannot be empty.");
          ConstraintGroup cloned;
          for (const auto &constraint : group) {
            if (!constraint)
              throw std::invalid_argument("Constraints cannot be None.");
            if (dynamic_cast<const LinearXConstraint *>(constraint.get()) &&
                constraint
                        ->derivative(zero, assemblage, parameters.n_parameters)
                        .size() != parameters.n_parameters) {
              throw std::invalid_argument(
                  "Linear constraint vector length must match n_parameters.");
            }
            cloned.push_back(constraint->clone());
          }
          constraints.push_back(std::move(cloned));
        }
        // The native store_assemblage option is currently a placeholder. Return
        // solver vectors instead; callers can apply a selected vector with the
        // helper above.
        return equilibrate(composition, assemblage, constraints, free_vectors,
                           tol, store_iterates, false, max_iterations, verbose,
                           parameter_tolerances);
      },
      py::arg("composition"), py::arg("assemblage"),
      py::arg("equality_constraints"), py::arg("free_compositional_vectors"),
      py::arg("tol"), py::arg("store_iterates"), py::arg("max_iterations"),
      py::arg("verbose"));
}
} // namespace burnman::python
