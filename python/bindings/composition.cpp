#include "burnman/core/composition.hpp"
#include "bindings.hpp"
#include <pybind11/operators.h>

namespace burnman::python {
namespace {
Composition::ComponentAmounts parse_amounts(const py::dict &input) {
  Composition::ComponentAmounts result;
  for (const auto &item : input)
    result.emplace_back(py::cast<std::string>(item.first),
                        py::cast<double>(item.second));
  return result;
}

py::dict as_dict(const Composition::ComponentAmounts &amounts) {
  py::dict result;
  for (const auto &[component, amount] : amounts)
    result[py::str(component)] = amount;
  return result;
}
} // namespace

void bind_composition(py::module_ &m) {
  py::class_<Composition> cls(m, "Composition", R"doc(
Chemical composition with native conversions, normalization and basis changes.

Components are formula strings, e.g. MgO, Fe2O3 or MgSiO3. Mass amounts are in
kg; molar amounts are in mol of components; atomic amounts are in mol of atoms.
"weight" aliases "mass". Signed amounts are supported. Dictionaries returned by
properties are independent snapshots. All calculations, including NNLS basis
changes, run in C++ without importing Python BurnMan or SciPy.
)doc");
  cls.def(py::init([](const py::dict &composition_dictionary,
                      const std::string &unit_type, bool normalize) {
            return Composition(parse_amounts(composition_dictionary), unit_type,
                               normalize);
          }),
          py::arg("composition_dictionary"), py::arg("unit_type") = "mass",
          py::arg("normalize") = false);

  const auto mass = [](const Composition &self) {
    return as_dict(self.get_mass_composition());
  };
  const auto moles = [](const Composition &self) {
    return as_dict(self.get_molar_composition());
  };
  const auto atoms = [](const Composition &self) {
    return as_dict(self.get_atomic_composition());
  };
  const auto formulae = [](const Composition &self) {
    py::dict result;
    for (const auto &[component, formula] : self.get_component_formulae())
      result[py::str(component)] = py::cast(formula);
    return result;
  };
  cls.def_property_readonly("mass_composition", mass)
      .def("get_mass_composition", mass)
      .def_property_readonly("weight_composition", mass)
      .def("get_weight_composition", mass)
      .def_property_readonly("molar_composition", moles)
      .def("get_molar_composition", moles)
      .def_property_readonly("atomic_composition", atoms)
      .def("get_atomic_composition", atoms)
      .def_property_readonly("component_formulae", formulae)
      .def("get_component_formulae", formulae)
      .def_property_readonly("element_list", &Composition::get_element_list,
                             py::return_value_policy::copy)
      .def("get_element_list", &Composition::get_element_list,
           py::return_value_policy::copy)
      .def(
          "composition",
          [](const Composition &self, const std::string &unit_type) {
            return as_dict(self.composition(unit_type));
          },
          py::arg("unit_type"))
      .def("renormalize", &Composition::renormalize, py::arg("unit_type"),
           py::arg("normalization_component"), py::arg("normalization_amount"))
      .def(
          "add_components",
          [](Composition &self, const py::dict &amounts,
             const std::string &unit_type) {
            self.add_components(parse_amounts(amounts), unit_type);
          },
          py::arg("composition_dictionary"), py::arg("unit_type"))
      .def("change_component_set", &Composition::change_component_set,
           py::arg("new_component_list"),
           "Preserve the elemental inventory in a nonnegative new basis "
           "(native NNLS; relative residual <= 1e-10). Invalid bases leave the "
           "object unchanged.")
      .def("remove_null_components", &Composition::remove_null_components,
           py::arg("tol") = 1.0e-12)
      .def("format", &Composition::format, py::arg("unit_type"),
           py::arg("significant_figures") = 1,
           py::arg("normalization_component") = "total",
           py::arg("normalization_amount") = py::none())
      .def(
          "print",
          [](const Composition &self, const std::string &unit_type, int digits,
             const std::string &component, std::optional<double> amount) {
            py::print(self.format(unit_type, digits, component, amount),
                      py::arg("end") = "");
          },
          py::arg("unit_type"), py::arg("significant_figures") = 1,
          py::arg("normalization_component") = "total",
          py::arg("normalization_amount") = py::none())
      .def("__repr__", &Composition::repr)
      .def("__str__", &Composition::repr)
      .def("__copy__", [](const Composition &self) { return self; })
      .def(
          "__deepcopy__",
          [](const Composition &self, const py::dict &) { return self; },
          py::arg("memo"))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * double())
      .def(double() * py::self)
      .def(py::self / double())
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= double())
      .def(py::self /= double());

  m.def("file_to_composition_list", &file_to_composition_list, py::arg("fname"),
        py::arg("unit_type"), py::arg("normalize"),
        "Read a whitespace-separated composition table with a component header "
        "ending in 'Comment'. Return compositions and lists of comment words.");
}
} // namespace burnman::python
