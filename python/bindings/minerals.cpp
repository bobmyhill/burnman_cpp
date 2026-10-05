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
#include "burnman/minerals/datasets.hpp"

namespace burnman::python {
void bind_minerals(py::module_ &m) {
  auto catalog = m.def_submodule(
      "minerals", "Native mineral factories for the equilibration examples.");
  auto HP_2011_ds62 = catalog.def_submodule("HP_2011_ds62");
  HP_2011_ds62.def("sill", &minerals::HP_2011_ds62::sill);
  HP_2011_ds62.def("andalusite", &minerals::HP_2011_ds62::andalusite);
  HP_2011_ds62.def("ky", &minerals::HP_2011_ds62::ky);
  HP_2011_ds62.def("q", &minerals::HP_2011_ds62::q);
  HP_2011_ds62.def("law", &minerals::HP_2011_ds62::law);
  HP_2011_ds62.def("zo", &minerals::HP_2011_ds62::zo);
  HP_2011_ds62.def("ru", &minerals::HP_2011_ds62::ru);
  HP_2011_ds62.def("sph", &minerals::HP_2011_ds62::sph);
  HP_2011_ds62.def("ab", &minerals::HP_2011_ds62::ab);
  HP_2011_ds62.def("ta", &minerals::HP_2011_ds62::ta);
  HP_2011_ds62.def("pre", &minerals::HP_2011_ds62::pre);
  HP_2011_ds62.def("pa", &minerals::HP_2011_ds62::pa);
  HP_2011_ds62.def("ma", &minerals::HP_2011_ds62::ma);
  HP_2011_ds62.def("coe", &minerals::HP_2011_ds62::coe);
  auto HGP_2018_ds633 = catalog.def_submodule("HGP_2018_ds633");
  HGP_2018_ds633.def("silicate_melt", &minerals::HGP_2018_ds633::silicate_melt);
  HGP_2018_ds633.def("iron", &minerals::HGP_2018_ds633::iron);
  HGP_2018_ds633.def("wu", &minerals::HGP_2018_ds633::wu);
  HGP_2018_ds633.def("mt", &minerals::HGP_2018_ds633::mt);
  HGP_2018_ds633.def("hem", &minerals::HGP_2018_ds633::hem);
  auto mb50NCKFMASHTO = catalog.def_submodule("mb50NCKFMASHTO");
  mb50NCKFMASHTO.def("hb", &minerals::mb50NCKFMASHTO::hb);
  mb50NCKFMASHTO.def("aug", &minerals::mb50NCKFMASHTO::aug);
  mb50NCKFMASHTO.def("dio", &minerals::mb50NCKFMASHTO::dio);
  mb50NCKFMASHTO.def("opx", &minerals::mb50NCKFMASHTO::opx);
  mb50NCKFMASHTO.def("g", &minerals::mb50NCKFMASHTO::g);
  mb50NCKFMASHTO.def("ol", &minerals::mb50NCKFMASHTO::ol);
  mb50NCKFMASHTO.def("pl4tr", &minerals::mb50NCKFMASHTO::pl4tr);
  mb50NCKFMASHTO.def("abc", &minerals::mb50NCKFMASHTO::abc);
  mb50NCKFMASHTO.def("k4tr", &minerals::mb50NCKFMASHTO::k4tr);
  mb50NCKFMASHTO.def("ksp", &minerals::mb50NCKFMASHTO::ksp);
  mb50NCKFMASHTO.def("plc", &minerals::mb50NCKFMASHTO::plc);
  mb50NCKFMASHTO.def("pli", &minerals::mb50NCKFMASHTO::pli);
  mb50NCKFMASHTO.def("sp", &minerals::mb50NCKFMASHTO::sp);
  mb50NCKFMASHTO.def("ilm", &minerals::mb50NCKFMASHTO::ilm);
  mb50NCKFMASHTO.def("ilmm", &minerals::mb50NCKFMASHTO::ilmm);
  mb50NCKFMASHTO.def("ep", &minerals::mb50NCKFMASHTO::ep);
  mb50NCKFMASHTO.def("bi", &minerals::mb50NCKFMASHTO::bi);
  mb50NCKFMASHTO.def("mu", &minerals::mb50NCKFMASHTO::mu);
  mb50NCKFMASHTO.def("chl", &minerals::mb50NCKFMASHTO::chl);
  auto SLB_2011 = catalog.def_submodule("SLB_2011");
  SLB_2011.def("mg_fe_olivine", &minerals::SLB_2011::mg_fe_olivine);
  SLB_2011.def("mg_fe_wadsleyite", &minerals::SLB_2011::mg_fe_wadsleyite);
  SLB_2011.def("mg_fe_ringwoodite", &minerals::SLB_2011::mg_fe_ringwoodite);
  SLB_2011.def("mg_fe_bridgmanite", &minerals::SLB_2011::mg_fe_bridgmanite);
  SLB_2011.def("post_perovskite", &minerals::SLB_2011::post_perovskite);
  SLB_2011.def("ferropericlase", &minerals::SLB_2011::ferropericlase);
  SLB_2011.def("orthopyroxene", &minerals::SLB_2011::orthopyroxene);
  SLB_2011.def("garnet", &minerals::SLB_2011::garnet);
  SLB_2011.def("ca_perovskite", &minerals::SLB_2011::ca_perovskite);
  SLB_2011.def("pyrope_grossular", &minerals::SLB_2011::pyrope_grossular);
  SLB_2011.def("mg_fe_bridgmanite_binary",
               &minerals::SLB_2011::mg_fe_bridgmanite_binary);
  SLB_2011.def("periclase", &minerals::SLB_2011::periclase);
  SLB_2011.def("mg_perovskite", &minerals::SLB_2011::mg_perovskite);
  SLB_2011.def("mg_akimotoite", &minerals::SLB_2011::mg_akimotoite);
  SLB_2011.def("mg_ringwoodite", &minerals::SLB_2011::mg_ringwoodite);
  auto JH_2015 = catalog.def_submodule("JH_2015");
  JH_2015.def("orthopyroxene", &minerals::JH_2015::orthopyroxene);
  JH_2015.def("mg_fe_orthopyroxene", &minerals::JH_2015::mg_fe_orthopyroxene);
  auto mp50NCKFMASHTO = catalog.def_submodule("mp50NCKFMASHTO");
  mp50NCKFMASHTO.def("g", &minerals::mp50NCKFMASHTO::g);
  mp50NCKFMASHTO.def("pl4tr", &minerals::mp50NCKFMASHTO::pl4tr);
  mp50NCKFMASHTO.def("k4tr", &minerals::mp50NCKFMASHTO::k4tr);
  mp50NCKFMASHTO.def("plc", &minerals::mp50NCKFMASHTO::plc);
  mp50NCKFMASHTO.def("ksp", &minerals::mp50NCKFMASHTO::ksp);
  mp50NCKFMASHTO.def("ep", &minerals::mp50NCKFMASHTO::ep);
  mp50NCKFMASHTO.def("ma", &minerals::mp50NCKFMASHTO::ma);
  mp50NCKFMASHTO.def("mu", &minerals::mp50NCKFMASHTO::mu);
  mp50NCKFMASHTO.def("bi", &minerals::mp50NCKFMASHTO::bi);
  mp50NCKFMASHTO.def("opx", &minerals::mp50NCKFMASHTO::opx);
  mp50NCKFMASHTO.def("sa", &minerals::mp50NCKFMASHTO::sa);
  mp50NCKFMASHTO.def("cd", &minerals::mp50NCKFMASHTO::cd);
  mp50NCKFMASHTO.def("st", &minerals::mp50NCKFMASHTO::st);
  mp50NCKFMASHTO.def("chl", &minerals::mp50NCKFMASHTO::chl);
  mp50NCKFMASHTO.def("ctd", &minerals::mp50NCKFMASHTO::ctd);
  mp50NCKFMASHTO.def("sp", &minerals::mp50NCKFMASHTO::sp);
  mp50NCKFMASHTO.def("ilmm", &minerals::mp50NCKFMASHTO::ilmm);
  mp50NCKFMASHTO.def("ilm", &minerals::mp50NCKFMASHTO::ilm);
  mp50NCKFMASHTO.def("mt1", &minerals::mp50NCKFMASHTO::mt1);
  auto SLB_2024 = catalog.def_submodule("SLB_2024");
  SLB_2024.def("c2c_pyroxene", &minerals::SLB_2024::c2c_pyroxene);
  SLB_2024.def("calcium_ferrite_structured_phase",
               &minerals::SLB_2024::calcium_ferrite_structured_phase);
  SLB_2024.def("clinopyroxene", &minerals::SLB_2024::clinopyroxene);
  SLB_2024.def("garnet", &minerals::SLB_2024::garnet);
  SLB_2024.def("ilmenite", &minerals::SLB_2024::ilmenite);
  SLB_2024.def("ferropericlase", &minerals::SLB_2024::ferropericlase);
  SLB_2024.def("new_aluminous_phase", &minerals::SLB_2024::new_aluminous_phase);
  SLB_2024.def("olivine", &minerals::SLB_2024::olivine);
  SLB_2024.def("orthopyroxene", &minerals::SLB_2024::orthopyroxene);
  SLB_2024.def("plagioclase", &minerals::SLB_2024::plagioclase);
  SLB_2024.def("post_perovskite", &minerals::SLB_2024::post_perovskite);
  SLB_2024.def("bridgmanite", &minerals::SLB_2024::bridgmanite);
  SLB_2024.def("ringwoodite", &minerals::SLB_2024::ringwoodite);
  SLB_2024.def("mg_fe_aluminous_spinel",
               &minerals::SLB_2024::mg_fe_aluminous_spinel);
  SLB_2024.def("wadsleyite", &minerals::SLB_2024::wadsleyite);
  SLB_2024.def("ab", &minerals::SLB_2024::ab);
  SLB_2024.def("acm", &minerals::SLB_2024::acm);
  SLB_2024.def("al", &minerals::SLB_2024::al);
  SLB_2024.def("alpv", &minerals::SLB_2024::alpv);
  SLB_2024.def("an", &minerals::SLB_2024::an);
  SLB_2024.def("anao", &minerals::SLB_2024::anao);
  SLB_2024.def("andr", &minerals::SLB_2024::andr);
  SLB_2024.def("apbo", &minerals::SLB_2024::apbo);
  SLB_2024.def("appv", &minerals::SLB_2024::appv);
  SLB_2024.def("capv", &minerals::SLB_2024::capv);
  SLB_2024.def("cats", &minerals::SLB_2024::cats);
  SLB_2024.def("cen", &minerals::SLB_2024::cen);
  SLB_2024.def("co", &minerals::SLB_2024::co);
  SLB_2024.def("coes", &minerals::SLB_2024::coes);
  SLB_2024.def("cppv", &minerals::SLB_2024::cppv);
  SLB_2024.def("crcf", &minerals::SLB_2024::crcf);
  SLB_2024.def("crpv", &minerals::SLB_2024::crpv);
  SLB_2024.def("di", &minerals::SLB_2024::di);
  SLB_2024.def("en", &minerals::SLB_2024::en);
  SLB_2024.def("esk", &minerals::SLB_2024::esk);
  SLB_2024.def("fa", &minerals::SLB_2024::fa);
  SLB_2024.def("fapv", &minerals::SLB_2024::fapv);
  SLB_2024.def("fea", &minerals::SLB_2024::fea);
  SLB_2024.def("fec2", &minerals::SLB_2024::fec2);
  SLB_2024.def("fecf", &minerals::SLB_2024::fecf);
  SLB_2024.def("fee", &minerals::SLB_2024::fee);
  SLB_2024.def("feg", &minerals::SLB_2024::feg);
  SLB_2024.def("feil", &minerals::SLB_2024::feil);
  SLB_2024.def("fepv", &minerals::SLB_2024::fepv);
  SLB_2024.def("feri", &minerals::SLB_2024::feri);
  SLB_2024.def("fewa", &minerals::SLB_2024::fewa);
  SLB_2024.def("fnal", &minerals::SLB_2024::fnal);
  SLB_2024.def("fo", &minerals::SLB_2024::fo);
  SLB_2024.def("fppv", &minerals::SLB_2024::fppv);
  SLB_2024.def("fs", &minerals::SLB_2024::fs);
  SLB_2024.def("gr", &minerals::SLB_2024::gr);
  SLB_2024.def("hc", &minerals::SLB_2024::hc);
  SLB_2024.def("he", &minerals::SLB_2024::he);
  SLB_2024.def("hem", &minerals::SLB_2024::hem);
  SLB_2024.def("hepv", &minerals::SLB_2024::hepv);
  SLB_2024.def("hlpv", &minerals::SLB_2024::hlpv);
  SLB_2024.def("hmag", &minerals::SLB_2024::hmag);
  SLB_2024.def("hppv", &minerals::SLB_2024::hppv);
  SLB_2024.def("jd", &minerals::SLB_2024::jd);
  SLB_2024.def("knor", &minerals::SLB_2024::knor);
  SLB_2024.def("ky", &minerals::SLB_2024::ky);
  SLB_2024.def("lppv", &minerals::SLB_2024::lppv);
  SLB_2024.def("mag", &minerals::SLB_2024::mag);
  SLB_2024.def("mgc2", &minerals::SLB_2024::mgc2);
  SLB_2024.def("mgcf", &minerals::SLB_2024::mgcf);
  SLB_2024.def("mgil", &minerals::SLB_2024::mgil);
  SLB_2024.def("mgmj", &minerals::SLB_2024::mgmj);
  SLB_2024.def("mgpv", &minerals::SLB_2024::mgpv);
  SLB_2024.def("mgri", &minerals::SLB_2024::mgri);
  SLB_2024.def("mgts", &minerals::SLB_2024::mgts);
  SLB_2024.def("mgwa", &minerals::SLB_2024::mgwa);
  SLB_2024.def("mnal", &minerals::SLB_2024::mnal);
  SLB_2024.def("mppv", &minerals::SLB_2024::mppv);
  SLB_2024.def("nacf", &minerals::SLB_2024::nacf);
  SLB_2024.def("namj", &minerals::SLB_2024::namj);
  SLB_2024.def("neph", &minerals::SLB_2024::neph);
  SLB_2024.def("nnal", &minerals::SLB_2024::nnal);
  SLB_2024.def("odi", &minerals::SLB_2024::odi);
  SLB_2024.def("pe", &minerals::SLB_2024::pe);
  SLB_2024.def("picr", &minerals::SLB_2024::picr);
  SLB_2024.def("pwo", &minerals::SLB_2024::pwo);
  SLB_2024.def("py", &minerals::SLB_2024::py);
  SLB_2024.def("qtz", &minerals::SLB_2024::qtz);
  SLB_2024.def("smag", &minerals::SLB_2024::smag);
  SLB_2024.def("sp", &minerals::SLB_2024::sp);
  SLB_2024.def("st", &minerals::SLB_2024::st);
  SLB_2024.def("wo", &minerals::SLB_2024::wo);
  SLB_2024.def("wu", &minerals::SLB_2024::wu);
  SLB_2024.def("wuls", &minerals::SLB_2024::wuls);
}
} // namespace burnman::python
