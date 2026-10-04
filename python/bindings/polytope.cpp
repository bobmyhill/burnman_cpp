#include "burnman/tools/polytope.hpp"
#include "bindings.hpp"

namespace burnman::python {
void bind_polytope(py::module_ &m) {
  using namespace polytope;
  auto cls = py::class_<MaterialPolytope>(m, "MaterialPolytope");
  cls.def(
      py::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &, double>(),
      py::arg("equalities"), py::arg("inequalities"),
      py::arg("rational_tolerance") = 1.e-12,
      "Enumerate [b, a...] constraints b + a*x == 0 or >= 0 using cddlib/GMP.",
      py::call_guard<py::gil_scoped_release>());
#define POLYTOPE_PROPERTY(name)                                                \
  cls.def_property_readonly(                                                   \
      #name, [](const MaterialPolytope &self) { return self.get_##name(); },   \
      py::return_value_policy::copy)
  POLYTOPE_PROPERTY(equalities);
  POLYTOPE_PROPERTY(inequalities);
  POLYTOPE_PROPERTY(vertices);
  POLYTOPE_PROPERTY(rays);
  POLYTOPE_PROPERTY(lineality);
  POLYTOPE_PROPERTY(endmember_occupancies);
  POLYTOPE_PROPERTY(endmembers_as_independent_endmember_amounts);
#undef POLYTOPE_PROPERTY
  cls.def_property_readonly("is_empty", &MaterialPolytope::is_empty)
      .def_property_readonly("is_bounded", &MaterialPolytope::is_bounded);
  m.def("solution_polytope_from_endmember_occupancies",
        &solution_polytope_from_endmember_occupancies,
        py::arg("endmember_occupancies"),
        py::arg("rational_tolerance") = 1.e-12,
        py::call_guard<py::gil_scoped_release>());
  m.def("composite_polytope_at_constrained_composition",
        &composite_polytope_at_constrained_composition, py::arg("composite"),
        py::arg("composition"), py::arg("rational_tolerance") = 1.e-12,
        py::call_guard<py::gil_scoped_release>());
  m.def("transform_solution_to_new_basis", &transform_solution_to_new_basis,
        py::arg("solution"), py::arg("new_basis"),
        py::arg("molar_fractions") = Eigen::ArrayXd(),
        py::arg("solution_name") = "",
        py::call_guard<py::gil_scoped_release>());
  m.def("simplify_composite_with_composition",
        &simplify_composite_with_composition, py::arg("composite"),
        py::arg("composition"), py::arg("tolerance") = 1.e-10,
        py::arg("rational_tolerance") = 1.e-12,
        "Return independent native phases reduced to the feasible "
        "site-occupancy faces.",
        py::call_guard<py::gil_scoped_release>());
}
} // namespace burnman::python
