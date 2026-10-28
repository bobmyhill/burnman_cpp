// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once
#include "burnman/calibrants/catalogue.hpp"
namespace burnman::python {
inline void bind_calibrant_catalogue(py::module_ &m) {
  auto Anderson_1989 = m.def_submodule("Anderson_1989");
  Anderson_1989.def("Au", &calibrants::Anderson_1989::Au);
  auto Armentrout_2015 = m.def_submodule("Armentrout_2015");
  Armentrout_2015.def("Co_fcc", &calibrants::Armentrout_2015::Co_fcc);
  auto Campbell_2009 = m.def_submodule("Campbell_2009");
  Campbell_2009.def("Ni_fcc", &calibrants::Campbell_2009::Ni_fcc);
  auto Chidester_2021 = m.def_submodule("Chidester_2021");
  Chidester_2021.def("KCl_B2", &calibrants::Chidester_2021::KCl_B2);
  auto Decker_1971 = m.def_submodule("Decker_1971");
  Decker_1971.def("NaCl_B1", &calibrants::Decker_1971::NaCl_B1);
  auto Dewaele_2008 = m.def_submodule("Dewaele_2008");
  Dewaele_2008.def("Ni", &calibrants::Dewaele_2008::Ni);
  auto Dewaele_2012 = m.def_submodule("Dewaele_2012");
  Dewaele_2012.def("KBr_B2", &calibrants::Dewaele_2012::KBr_B2);
  Dewaele_2012.def("KCl_B2", &calibrants::Dewaele_2012::KCl_B2);
  auto Dewaele_2013 = m.def_submodule("Dewaele_2013");
  Dewaele_2013.def("Al2O3_corundum", &calibrants::Dewaele_2013::Al2O3_corundum);
  auto Dewaele_2020 = m.def_submodule("Dewaele_2020");
  Dewaele_2020.def("CsCl", &calibrants::Dewaele_2020::CsCl);
  auto Dorogokupets_2007 = m.def_submodule("Dorogokupets_2007");
  Dorogokupets_2007.def("Ag", &calibrants::Dorogokupets_2007::Ag);
  Dorogokupets_2007.def("Al", &calibrants::Dorogokupets_2007::Al);
  Dorogokupets_2007.def("Au", &calibrants::Dorogokupets_2007::Au);
  Dorogokupets_2007.def("Cu", &calibrants::Dorogokupets_2007::Cu);
  Dorogokupets_2007.def("diamond", &calibrants::Dorogokupets_2007::diamond);
  Dorogokupets_2007.def("MgO", &calibrants::Dorogokupets_2007::MgO);
  Dorogokupets_2007.def("Pt", &calibrants::Dorogokupets_2007::Pt);
  Dorogokupets_2007.def("Ta", &calibrants::Dorogokupets_2007::Ta);
  Dorogokupets_2007.def("W", &calibrants::Dorogokupets_2007::W);
  auto Dorogokupets_2017 = m.def_submodule("Dorogokupets_2017");
  Dorogokupets_2017.def("Fe_bcc", &calibrants::Dorogokupets_2017::Fe_bcc);
  Dorogokupets_2017.def("Fe_fcc", &calibrants::Dorogokupets_2017::Fe_fcc);
  Dorogokupets_2017.def("Fe_hcp", &calibrants::Dorogokupets_2017::Fe_hcp);
  auto Dubrovinsky_1998 = m.def_submodule("Dubrovinsky_1998");
  Dubrovinsky_1998.def("Al2O3_corundum",
                       &calibrants::Dubrovinsky_1998::Al2O3_corundum);
  auto Fei_2007 = m.def_submodule("Fei_2007");
  Fei_2007.def("Pt", &calibrants::Fei_2007::Pt);
  Fei_2007.def("Au", &calibrants::Fei_2007::Au);
  auto Fei_2016 = m.def_submodule("Fei_2016");
  Fei_2016.def("Fe_hcp", &calibrants::Fei_2016::Fe_hcp);
  auto Holmes_1989 = m.def_submodule("Holmes_1989");
  Holmes_1989.def("Pt", &calibrants::Holmes_1989::Pt);
  auto Huang_2016 = m.def_submodule("Huang_2016");
  Huang_2016.def("Mo", &calibrants::Huang_2016::Mo);
  auto LeGodec_2000 = m.def_submodule("LeGodec_2000");
  LeGodec_2000.def("hBN", &calibrants::LeGodec_2000::hBN);
  auto Litasov_2013 = m.def_submodule("Litasov_2013");
  Litasov_2013.def("Mo_bcc", &calibrants::Litasov_2013::Mo_bcc);
  Litasov_2013.def("W_bcc", &calibrants::Litasov_2013::W_bcc);
  auto Matsui_2009 = m.def_submodule("Matsui_2009");
  Matsui_2009.def("Pt", &calibrants::Matsui_2009::Pt);
  auto Matsui_2010 = m.def_submodule("Matsui_2010");
  Matsui_2010.def("Au", &calibrants::Matsui_2010::Au);
  auto Matsui_2012 = m.def_submodule("Matsui_2012");
  Matsui_2012.def("NaCl_B1", &calibrants::Matsui_2012::NaCl_B1);
  auto Miozzi_2020 = m.def_submodule("Miozzi_2020");
  Miozzi_2020.def("Fe_hcp", &calibrants::Miozzi_2020::Fe_hcp);
  auto Ono_2022 = m.def_submodule("Ono_2022");
  Ono_2022.def("Re", &calibrants::Ono_2022::Re);
  auto Pigott_2015 = m.def_submodule("Pigott_2015");
  Pigott_2015.def("Ni", &calibrants::Pigott_2015::Ni);
  auto Shi_2022 = m.def_submodule("Shi_2022");
  Shi_2022.def("Al2O3_corundum", &calibrants::Shi_2022::Al2O3_corundum);
  auto Shim_2002 = m.def_submodule("Shim_2002");
  Shim_2002.def("Au", &calibrants::Shim_2002::Au);
  auto Sokolova_2013 = m.def_submodule("Sokolova_2013");
  Sokolova_2013.def("Mo", &calibrants::Sokolova_2013::Mo);
  auto Speziale_2001 = m.def_submodule("Speziale_2001");
  Speziale_2001.def("MgO", &calibrants::Speziale_2001::MgO);
  auto Tange_2009 = m.def_submodule("Tange_2009");
  Tange_2009.def("MgO_BM3", &calibrants::Tange_2009::MgO_BM3);
  Tange_2009.def("MgO_Vinet", &calibrants::Tange_2009::MgO_Vinet);
  auto Tateno_2019 = m.def_submodule("Tateno_2019");
  Tateno_2019.def("KCl_B2", &calibrants::Tateno_2019::KCl_B2);
  auto Tsuchiya_2003 = m.def_submodule("Tsuchiya_2003");
  Tsuchiya_2003.def("Au", &calibrants::Tsuchiya_2003::Au);
  auto Walker_2002 = m.def_submodule("Walker_2002");
  Walker_2002.def("KCl_B2", &calibrants::Walker_2002::KCl_B2);
  auto Zeng_2010 = m.def_submodule("Zeng_2010");
  Zeng_2010.def("Mo", &calibrants::Zeng_2010::Mo);
  auto Zha_2004 = m.def_submodule("Zha_2004");
  Zha_2004.def("Re", &calibrants::Zha_2004::Re);
  auto Zha_2008 = m.def_submodule("Zha_2008");
  Zha_2008.def("Pt", &calibrants::Zha_2008::Pt);
  auto Zhang_1999 = m.def_submodule("Zhang_1999");
  Zhang_1999.def("Fe_bcc", &calibrants::Zhang_1999::Fe_bcc);
  auto Zhao_1997 = m.def_submodule("Zhao_1997");
  Zhao_1997.def("hBN", &calibrants::Zhao_1997::hBN);
  auto Zhao_2000 = m.def_submodule("Zhao_2000");
  Zhao_2000.def("Mo", &calibrants::Zhao_2000::Mo);
}
} // namespace burnman::python
