#include "bindings.hpp"
#include "burnman/utils/exceptions.hpp"

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native BurnMan C++ bindings. All quantities use SI units.";
  m.attr("__version__") = BURNMAN_VERSION;
  py::register_exception<burnman::exceptions::NotImplementedError>(
      m, "NotImplementedError", PyExc_NotImplementedError);
  burnman::python::bind_params(m);
  burnman::python::bind_materials(m);
  burnman::python::bind_combined(m);
  burnman::python::bind_solutions(m);
  burnman::python::bind_minerals(m);
  burnman::python::bind_equilibration(m);
}
