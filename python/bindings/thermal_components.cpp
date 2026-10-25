// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "bindings.hpp"
#include "burnman/eos/components/anharmonic_debye.hpp"
#include "burnman/eos/components/bukowinski_electronic.hpp"
#include "burnman/eos/components/debye.hpp"
#include "burnman/eos/components/einstein.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using ThermalFunction = double (*)(double, double, double);
void bind_thermal_functions(
    py::module_ &m, const char *theta_name,
    std::initializer_list<std::pair<const char *, ThermalFunction>> functions) {
  for (const auto &entry : functions)
    m.def(entry.first,
          py::vectorize(
              [function = entry.second](double t, double theta, double n) {
                if (!std::isfinite(t) || t < 0. || !std::isfinite(theta) ||
                    theta <= 0. || !std::isfinite(n) || n < 0.)
                  throw std::invalid_argument(
                      "Thermal components require finite T>=0, characteristic "
                      "temperature>0 and n>=0.");
                return function(t, theta, n);
              }),
          py::arg("T"), py::arg(theta_name), py::arg("n"));
}
types::MineralParams electronic_params(double t, double v, double t0, double v0,
                                       double b, double g) {
  if (!std::isfinite(t) || t < 0. || !std::isfinite(t0) || t0 < 0. ||
      !std::isfinite(v) || v <= 0. || !std::isfinite(v0) || v0 <= 0. ||
      !std::isfinite(b) || !std::isfinite(g))
    throw std::invalid_argument(
        "Invalid electronic thermal-component arguments.");
  types::MineralParams p;
  p.T_0 = t0;
  p.V_0 = v0;
  p.bel_0 = b;
  p.gel = g;
  return p;
}
} // namespace

void bind_thermal_components(py::module_ &m) {
  namespace d = eos::debye;
  namespace e = eos::einstein;
  namespace b = eos::bukowinski;
  auto debye = m.def_submodule("debye");
  bind_thermal_functions(
      debye, "debye_T",
      {{"thermal_energy", &d::compute_thermal_energy},
       {"molar_heat_capacity_v", &d::compute_molar_heat_capacity_v},
       {"helmholtz_energy", &d::compute_helmholtz_free_energy},
       {"entropy", &d::compute_entropy},
       {"dmolar_heat_capacity_v_dT", &d::compute_dmolar_heat_capacity_v_dT},
       {"dhelmholtz_dTheta", &d::compute_dhelmholtz_dTheta},
       {"d2helmholtz_dTheta2", &d::compute_d2helmholtz_dTheta2},
       {"dentropy_dTheta", &d::compute_dentropy_dTheta}});
  const auto bind_debye_function = [&](const char *name, auto function) {
    debye.def(name, py::vectorize([function](double x) {
                if (!std::isfinite(x) || x < 0.)
                  throw std::invalid_argument(
                      "Debye argument must be finite and nonnegative.");
                return function(x);
              }),
              py::arg("x"));
  };
  bind_debye_function("debye_fn", &d::debye_fn_quad);
  bind_debye_function("debye_fn_cheb", &d::debye_fn_cheb);
  auto einstein = m.def_submodule("einstein");
  bind_thermal_functions(
      einstein, "einstein_T",
      {{"thermal_energy", &e::compute_thermal_energy},
       {"molar_heat_capacity_v", &e::compute_molar_heat_capacity_v},
       {"helmholtz_energy", &e::compute_helmholtz_free_energy},
       {"entropy", &e::compute_entropy},
       {"dmolar_heat_capacity_v_dT", &e::compute_dmolar_heat_capacity_v_dT},
       {"dhelmholtz_dTheta", &e::compute_dhelmholtz_dTheta},
       {"d2helmholtz_dTheta2", &e::compute_d2helmholtz_dTheta2},
       {"dentropy_dTheta", &e::compute_dentropy_dTheta}});

  auto electronic = m.def_submodule("bukowinski_electronic");
  electronic.def("helmholtz",
                 py::vectorize([](double t, double v, double t0, double v0,
                                  double bel, double gel) {
                   return b::compute_helmholtz_el(
                       t, v, electronic_params(t, v, t0, v0, bel, gel));
                 }),
                 py::arg("temperature"), py::arg("volume"), py::arg("T_0"),
                 py::arg("V_0"), py::arg("bel_0"), py::arg("gel"));
  electronic.def("pressure",
                 py::vectorize([](double t, double v, double t0, double v0,
                                  double bel, double gel) {
                   return b::compute_pressure_el(
                       t, v, electronic_params(t, v, t0, v0, bel, gel));
                 }),
                 py::arg("temperature"), py::arg("volume"), py::arg("T_0"),
                 py::arg("V_0"), py::arg("bel_0"), py::arg("gel"));
  electronic.def("KToverV",
                 py::vectorize([](double t, double v, double t0, double v0,
                                  double bel, double gel) {
                   return b::compute_KT_over_V(
                       t, v, electronic_params(t, v, t0, v0, bel, gel));
                 }),
                 py::arg("temperature"), py::arg("volume"), py::arg("T_0"),
                 py::arg("V_0"), py::arg("bel_0"), py::arg("gel"));
  electronic.def(
      "entropy",
      py::vectorize([](double t, double v, double v0, double bel, double gel) {
        return b::compute_entropy_el(t, v,
                                     electronic_params(t, v, 0., v0, bel, gel));
      }),
      py::arg("temperature"), py::arg("volume"), py::arg("V_0"),
      py::arg("bel_0"), py::arg("gel"));
  electronic.def(
      "CVoverT", py::vectorize([](double v, double v0, double bel, double gel) {
        return b::compute_CV_over_T(v,
                                    electronic_params(0., v, 0., v0, bel, gel));
      }),
      py::arg("volume"), py::arg("V_0"), py::arg("bel_0"), py::arg("gel"));
  electronic.def(
      "aKT",
      py::vectorize([](double t, double v, double v0, double bel, double gel) {
        return b::compute_alpha_KT(t, v,
                                   electronic_params(t, v, 0., v0, bel, gel));
      }),
      py::arg("temperature"), py::arg("volume"), py::arg("V_0"),
      py::arg("bel_0"), py::arg("gel"));

  auto anharmonic = m.def_submodule("anharmonic_debye");
  using Anharmonic = eos::AnharmonicDebye;
  auto cls =
      py::class_<Anharmonic>(anharmonic, "AnharmonicDebye").def(py::init<>());
  cls.def_static("validate_parameters", &Anharmonic::validate_parameters,
                 py::arg("params"));
  const auto bind_anharmonic = [&](const char *name, auto function) {
    cls.def_static(
        name,
        [function](double t, double v, types::MineralParams p) {
          Anharmonic::validate_parameters(p);
          return function(t, v, p);
        },
        py::arg("temperature"), py::arg("volume"), py::arg("params"));
  };
  bind_anharmonic("helmholtz_energy", &Anharmonic::helmholtz_energy);
  bind_anharmonic("pressure", &Anharmonic::pressure);
  bind_anharmonic("isothermal_bulk_modulus",
                  &Anharmonic::isothermal_bulk_modulus);
  bind_anharmonic("entropy", &Anharmonic::entropy);
  bind_anharmonic("heat_capacity_v", &Anharmonic::heat_capacity_v);
  bind_anharmonic("dSdV", &Anharmonic::dSdV);
}
} // namespace burnman::python
