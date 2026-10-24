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
#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::python {
void bind_params(py::module_ &m) {
  using namespace types;
  py::enum_<EOSType>(m, "EOSType")
      .value("Auto", EOSType::Auto)
      .value("Vinet", EOSType::Vinet)
      .value("MT", EOSType::MT)
      .value("BM2", EOSType::BM2)
      .value("BM3", EOSType::BM3)
      .value("BM4", EOSType::BM4)
      .value("SPOCK", EOSType::SPOCK)
      .value("DKS_S", EOSType::DKS_S)
      .value("DKS_L", EOSType::DKS_L)
      .value("MGD2", EOSType::MGD2)
      .value("MGD3", EOSType::MGD3)
      .value("ModularMGD", EOSType::ModularMGD)
      .value("ModularMGDWithAnharmonicity",
             EOSType::ModularMGDWithAnharmonicity)
      .value("HPTMT", EOSType::HPTMT)
      .value("HPTMTL", EOSType::HPTMTL)
      .value("CORK", EOSType::CORK)
      .value("SLB2", EOSType::SLB2)
      .value("SLB3", EOSType::SLB3)
      .value("SLB3Conductive", EOSType::SLB3Conductive)
      .value("SLB3Stishovite", EOSType::SLB3Stishovite)
      .value("Custom", EOSType::Custom);
  py::enum_<AveragingType>(m, "AveragingType")
      .value("Voigt", AveragingType::Voigt)
      .value("Reuss", AveragingType::Reuss)
      .value("VRH", AveragingType::VRH)
      .value("HashinShtrikmanLower", AveragingType::HashinShtrikmanLower)
      .value("HashinShtrikmanUpper", AveragingType::HashinShtrikmanUpper)
      .value("HashinShtrikman", AveragingType::HashinShtrikman);
  py::enum_<FractionType>(m, "FractionType")
      .value("Molar", FractionType::Molar)
      .value("Mass", FractionType::Mass);

  py::class_<CpParams>(m, "CpParams")
      .def(py::init<double, double, double, double>(), py::arg("a"),
           py::arg("b"), py::arg("c"), py::arg("d"))
      .def_readwrite("a", &CpParams::a)
      .def_readwrite("b", &CpParams::b)
      .def_readwrite("c", &CpParams::c)
      .def_readwrite("d", &CpParams::d);
  py::class_<CorkParams>(m, "CorkParams")
      .def(py::init<double, double, double, double, double, double, double>())
      .def_readwrite("a_0", &CorkParams::a_0)
      .def_readwrite("a_1", &CorkParams::a_1)
      .def_readwrite("b", &CorkParams::b)
      .def_readwrite("c_0", &CorkParams::c_0)
      .def_readwrite("c_1", &CorkParams::c_1)
      .def_readwrite("d_0", &CorkParams::d_0)
      .def_readwrite("d_1", &CorkParams::d_1);
  auto params = py::class_<MineralParams>(m, "MineralParams").def(py::init<>());
#define PARAM(name) params.def_readwrite(#name, &MineralParams::name)
  PARAM(name);
  PARAM(formula);
  PARAM(equation_of_state);
  PARAM(napfu);
  PARAM(molar_mass);
  PARAM(Z);
  PARAM(P_0);
  PARAM(T_0);
  PARAM(E_0);
  PARAM(F_0);
  PARAM(V_0);
  PARAM(K_0);
  PARAM(Kprime_0);
  PARAM(Kdprime_0);
  PARAM(G_0);
  PARAM(Gprime_0);
  PARAM(Kprime_inf);
  PARAM(Gprime_inf);
  PARAM(S_0);
  PARAM(H_0);
  PARAM(Cv);
  PARAM(Cp);
  PARAM(debye_0);
  PARAM(grueneisen_0);
  PARAM(q_0);
  PARAM(a_0);
  PARAM(dKdT_0);
  PARAM(m);
  PARAM(a);
  PARAM(eta_s_0);
  PARAM(T_einstein);
  PARAM(bel_0);
  PARAM(gel);
  PARAM(reference_eos);
  PARAM(debye_temperature_model);
  PARAM(anharmonic_prefactor_model);
  PARAM(anharmonic_thermal_model);
  PARAM(c_1);
  PARAM(c_2);
  PARAM(q_1);
  PARAM(q_2);
  PARAM(a_anh);
  PARAM(b_anh);
  PARAM(c_anh);
  PARAM(m_anh);
  PARAM(mu_anh);
  PARAM(sigma_anh);
  PARAM(order_theta);
  PARAM(order_f);
  PARAM(Tel_0);
  PARAM(zeta_0);
  PARAM(xi);
  PARAM(eta);
  PARAM(el_V_0);
  PARAM(spin_a);
  PARAM(spin_b);
  PARAM(dks_s_power_law_grueneisen);
  params.def_readwrite("O_theta", &MineralParams::order_theta);
  params.def_readwrite("O_f", &MineralParams::order_f);
  PARAM(delta_0);
  PARAM(delta_1);
  PARAM(b_0);
  PARAM(b_1);
  PARAM(cork_T);
  PARAM(cork_P);
  PARAM(cork_params);
  PARAM(dcorkdT);
#undef PARAM
}
} // namespace burnman::python
