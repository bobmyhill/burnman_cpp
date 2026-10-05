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

#include "burnman/eos/make_eos.hpp"
#include "burnman/eos/birch_murnaghan.hpp"
#include "burnman/eos/hp.hpp"
#include "burnman/eos/hp_liquid.hpp"
#include "burnman/eos/mie_grueneisen_debye.hpp"
#include "burnman/eos/modified_tait.hpp"
#include "burnman/eos/slb.hpp"
#include "burnman/eos/vinet.hpp"
#include <stdexcept>

namespace burnman::eos {
// TODO: Add HP types here

std::shared_ptr<EquationOfState> make_eos(types::EOSType eos_type) {
  switch (eos_type) {
  case types::EOSType::Vinet:
    return std::make_shared<Vinet>();
  case types::EOSType::BM2:
    return std::make_shared<BM2>();
  case types::EOSType::BM3:
    return std::make_shared<BM3>();
  case types::EOSType::MT:
    return std::make_shared<MT>();
  case types::EOSType::MGD2:
    return std::make_shared<MGD2>();
  case types::EOSType::MGD3:
    return std::make_shared<MGD3>();
  case types::EOSType::SLB2:
    return std::make_shared<SLB2>();
  case types::EOSType::SLB3:
    return std::make_shared<SLB3>();
  case types::EOSType::SLB3Conductive:
    return std::make_shared<SLB3Conductive>();
  case types::EOSType::SLB3Stishovite:
    return std::make_shared<SLB3Stishovite>();
  case types::EOSType::HPTMTL:
    return std::make_shared<HP_TMTL>();
  case types::EOSType::HPTMT:
    return std::make_shared<HP_TMT>();
  default:
    throw std::invalid_argument("Unknown EOS type!");
  }
}

} // namespace burnman::eos
