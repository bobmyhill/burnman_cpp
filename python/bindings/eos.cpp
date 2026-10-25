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

#include "bindings.hpp"
#include "burnman/core/equation_of_state.hpp"
#include "burnman/eos/components/anharmonic_prefactor_models.hpp"
#include "burnman/eos/components/anharmonic_thermal_models.hpp"
#include "burnman/eos/components/debye_temperature_models.hpp"
#include "burnman/eos/make_eos.hpp"

namespace burnman::python {
namespace {
void bind_eos_components(py::module_ &m) {
  namespace dt = eos::debye_temperature_models;
  namespace ap = eos::anharmonic_prefactor_models;
  namespace at = eos::anharmonic_thermal_models;
  auto debye = m.def_submodule("debye_temperature_models");
  py::class_<dt::DebyeTemperatureModelBase,
             std::shared_ptr<dt::DebyeTemperatureModelBase>>(
      debye, "DebyeTemperatureModelBase")
      .def("value", &dt::DebyeTemperatureModelBase::value)
      .def("dVrel", &dt::DebyeTemperatureModelBase::dVrel)
      .def("dVrel2", &dt::DebyeTemperatureModelBase::dVrel2)
      .def("_gamma", &dt::DebyeTemperatureModelBase::gamma)
      .def("_gamma_prime", &dt::DebyeTemperatureModelBase::gamma_prime)
      .def("validate_parameters",
           &dt::DebyeTemperatureModelBase::validate_parameters);
  py::class_<dt::SLB, dt::DebyeTemperatureModelBase, std::shared_ptr<dt::SLB>>(
      debye, "SLB")
      .def(py::init<>());
  py::class_<dt::PowerLawGammaSimple, dt::DebyeTemperatureModelBase,
             std::shared_ptr<dt::PowerLawGammaSimple>>(debye,
                                                       "PowerLawGammaSimple")
      .def(py::init<>());
  py::class_<dt::PowerLawGamma, dt::DebyeTemperatureModelBase,
             std::shared_ptr<dt::PowerLawGamma>>(debye, "PowerLawGamma")
      .def(py::init<>());
  auto prefactor = m.def_submodule("anharmonic_prefactor_models");
  py::class_<ap::AnharmonicPrefactorModel,
             std::shared_ptr<ap::AnharmonicPrefactorModel>>(
      prefactor, "AnharmonicPrefactorModel")
      .def("value", &ap::AnharmonicPrefactorModel::value)
      .def("dVrel", &ap::AnharmonicPrefactorModel::dVrel)
      .def("dVrel2", &ap::AnharmonicPrefactorModel::dVrel2)
      .def("validate_parameters",
           &ap::AnharmonicPrefactorModel::validate_parameters);
  py::class_<ap::PowerLaw, ap::AnharmonicPrefactorModel,
             std::shared_ptr<ap::PowerLaw>>(prefactor, "PowerLaw")
      .def(py::init<>());
  py::class_<ap::Sigmoid, ap::AnharmonicPrefactorModel,
             std::shared_ptr<ap::Sigmoid>>(prefactor, "Sigmoid")
      .def(py::init<>());
  auto thermal = m.def_submodule("anharmonic_thermal_models");
  auto base = py::class_<at::AnharmonicThermalModelBase,
                         std::shared_ptr<at::AnharmonicThermalModelBase>>(
                  thermal, "AnharmonicThermalModelBase")
                  .def("validate_parameters",
                       &at::AnharmonicThermalModelBase::validate_parameters);
  base.def("nondimensional_helmholtz_energy",
           [](const at::AnharmonicThermalModelBase &self, double temperature,
              double theta, const types::MineralParams &params) {
             return self.evaluate(temperature, theta, params).helmholtz;
           });
  base.def("nondimensional_entropy",
           [](const at::AnharmonicThermalModelBase &self, double temperature,
              double theta, const types::MineralParams &params) {
             return self.evaluate(temperature, theta, params).entropy;
           });
  base.def("nondimensional_heat_capacity",
           [](const at::AnharmonicThermalModelBase &self, double temperature,
              double theta, const types::MineralParams &params) {
             return self.evaluate(temperature, theta, params).heat_capacity;
           });
  base.def("nondimensional_dhelmholtz_dTheta",
           [](const at::AnharmonicThermalModelBase &self, double temperature,
              double theta, const types::MineralParams &params) {
             return self.evaluate(temperature, theta, params).dhelmholtz_dTheta;
           });
  base.def(
      "nondimensional_d2helmholtz_dTheta2",
      [](const at::AnharmonicThermalModelBase &self, double temperature,
         double theta, const types::MineralParams &params) {
        return self.evaluate(temperature, theta, params).d2helmholtz_dTheta2;
      });
  base.def("nondimensional_dentropy_dTheta",
           [](const at::AnharmonicThermalModelBase &self, double temperature,
              double theta, const types::MineralParams &params) {
             return self.evaluate(temperature, theta, params).dentropy_dTheta;
           });
  py::class_<at::Pade, at::AnharmonicThermalModelBase,
             std::shared_ptr<at::Pade>>(thermal, "Pade")
      .def(py::init<>());
  py::class_<at::LogNormal, at::AnharmonicThermalModelBase,
             std::shared_ptr<at::LogNormal>>(thermal, "LogNormal")
      .def(py::init<>());
}
} // namespace

// All concrete equations of state use this interface and the make_eos factory.
void bind_eos(py::module_ &m) {
  bind_eos_components(m);
  bind_thermal_components(m);
  auto eos = py::class_<EquationOfState, std::shared_ptr<EquationOfState>>(
      m, "EquationOfState");
  eos.def("validate_parameters", &EquationOfState::validate_parameters,
          py::arg("params"))
      .def(
          "compute_volume",
          [](EquationOfState &self, double pressure, double temperature,
             types::MineralParams params) {
            check_pt(pressure, temperature);
            self.validate_parameters(params);
            return self.compute_volume(pressure, temperature, params);
          },
          py::arg("pressure"), py::arg("temperature"), py::arg("params"))
      .def(
          "compute_pressure",
          [](EquationOfState &self, double temperature, double volume,
             types::MineralParams params) {
            check_pt(0.0, temperature);
            if (!std::isfinite(volume) || volume <= 0.0)
              throw std::invalid_argument(
                  "Volume must be positive and finite.");
            self.validate_parameters(params);
            return self.compute_pressure(temperature, volume, params);
          },
          py::arg("temperature"), py::arg("volume"), py::arg("params"));
  // Direct volume-based properties let examples evaluate isochores without
  // solving V(P,T) again or selecting a different stable volume branch.
  const auto bind_volume_property = [&](const char *name, auto method) {
    eos.def(
        name,
        [method](EquationOfState &self, double pressure, double temperature,
                 double volume, types::MineralParams params) {
          check_pt(pressure, temperature);
          if (!std::isfinite(volume) || volume <= 0.0)
            throw std::invalid_argument("Volume must be positive and finite.");
          self.validate_parameters(params);
          return (self.*method)(pressure, temperature, volume, params);
        },
        py::arg("pressure"), py::arg("temperature"), py::arg("volume"),
        py::arg("params"));
  };
  bind_volume_property("compute_molar_heat_capacity_v",
                       &EquationOfState::compute_molar_heat_capacity_v);
  bind_volume_property("compute_molar_heat_capacity_p",
                       &EquationOfState::compute_molar_heat_capacity_p);
  bind_volume_property("compute_thermal_expansivity",
                       &EquationOfState::compute_thermal_expansivity);
  bind_volume_property("compute_grueneisen_parameter",
                       &EquationOfState::compute_grueneisen_parameter);
  bind_volume_property("compute_isothermal_bulk_modulus_reuss",
                       &EquationOfState::compute_isothermal_bulk_modulus_reuss);
  bind_volume_property("compute_isentropic_bulk_modulus_reuss",
                       &EquationOfState::compute_isentropic_bulk_modulus_reuss);
  bind_volume_property("compute_entropy", &EquationOfState::compute_entropy);
  bind_volume_property("compute_helmholtz_free_energy",
                       &EquationOfState::compute_helmholtz_free_energy);
  bind_volume_property("compute_molar_internal_energy",
                       &EquationOfState::compute_molar_internal_energy);
  bind_volume_property("compute_gibbs_free_energy",
                       &EquationOfState::compute_gibbs_free_energy);
  bind_volume_property("compute_enthalpy", &EquationOfState::compute_enthalpy);
  m.def("make_eos", &eos::make_eos, py::arg("method"));
}
} // namespace burnman::python
