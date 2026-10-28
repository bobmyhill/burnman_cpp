// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#pragma once
#include "burnman/core/calibrant.hpp"
#include <vector>
namespace burnman::calibrants {
namespace Anderson_1989 {
Calibrant Au();
}
namespace Armentrout_2015 {
Calibrant Co_fcc();
}
namespace Campbell_2009 {
Calibrant Ni_fcc();
}
namespace Chidester_2021 {
Calibrant KCl_B2();
}
namespace Decker_1971 {
Calibrant NaCl_B1();
}
namespace Dewaele_2008 {
Calibrant Ni();
}
namespace Dewaele_2012 {
Calibrant KBr_B2();
Calibrant KCl_B2();
} // namespace Dewaele_2012
namespace Dewaele_2013 {
Calibrant Al2O3_corundum();
}
namespace Dewaele_2020 {
Calibrant CsCl();
}
namespace Dorogokupets_2007 {
Calibrant Ag();
Calibrant Al();
Calibrant Au();
Calibrant Cu();
Calibrant diamond();
Calibrant MgO();
Calibrant Pt();
Calibrant Ta();
Calibrant W();
} // namespace Dorogokupets_2007
namespace Dorogokupets_2017 {
Calibrant Fe_bcc();
Calibrant Fe_fcc();
Calibrant Fe_hcp();
} // namespace Dorogokupets_2017
namespace Dubrovinsky_1998 {
Calibrant Al2O3_corundum();
}
namespace Fei_2007 {
Calibrant Pt();
Calibrant Au();
} // namespace Fei_2007
namespace Fei_2016 {
Calibrant Fe_hcp();
}
namespace Holmes_1989 {
Calibrant Pt();
}
namespace Huang_2016 {
Calibrant Mo();
}
namespace LeGodec_2000 {
Calibrant hBN();
}
namespace Litasov_2013 {
Calibrant Mo_bcc();
Calibrant W_bcc();
} // namespace Litasov_2013
namespace Matsui_2009 {
Calibrant Pt();
}
namespace Matsui_2010 {
Calibrant Au();
}
namespace Matsui_2012 {
Calibrant NaCl_B1();
}
namespace Miozzi_2020 {
Calibrant Fe_hcp();
}
namespace Ono_2022 {
Calibrant Re();
}
namespace Pigott_2015 {
Calibrant Ni();
}
namespace Shi_2022 {
Calibrant Al2O3_corundum();
}
namespace Shim_2002 {
Calibrant Au();
}
namespace Sokolova_2013 {
Calibrant Mo();
}
namespace Speziale_2001 {
Calibrant MgO();
}
namespace Tange_2009 {
Calibrant MgO_BM3();
Calibrant MgO_Vinet();
} // namespace Tange_2009
namespace Tateno_2019 {
Calibrant KCl_B2();
}
namespace Tsuchiya_2003 {
Calibrant Au();
}
namespace Walker_2002 {
Calibrant KCl_B2();
}
namespace Zeng_2010 {
Calibrant Mo();
}
namespace Zha_2004 {
Calibrant Re();
}
namespace Zha_2008 {
Calibrant Pt();
}
namespace Zhang_1999 {
Calibrant Fe_bcc();
}
namespace Zhao_1997 {
Calibrant hBN();
}
namespace Zhao_2000 {
Calibrant Mo();
}
using Factory = Calibrant (*)();
const std::vector<std::pair<std::string, Factory>> &catalogue();
} // namespace burnman::calibrants
