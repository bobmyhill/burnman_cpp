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

#ifndef TESTS_EOS_ENUM_STRING_HPP
#define TESTS_EOS_ENUM_STRING_HPP

#include "burnman/utils/types/simple_types.hpp"
#include <string>

inline constexpr const char *eos_string(burnman::types::EOSType eos_type) {
  switch (eos_type) {
  case burnman::types::EOSType::BM2:
    return "BM2";
  case burnman::types::EOSType::BM3:
    return "BM3";
  case burnman::types::EOSType::MT:
    return "MT";
  case burnman::types::EOSType::Vinet:
    return "Vinet";
  case burnman::types::EOSType::MGD2:
    return "MGD2";
  case burnman::types::EOSType::MGD3:
    return "MGD3";
  case burnman::types::EOSType::SLB2:
    return "SLB2";
  case burnman::types::EOSType::SLB3:
    return "SLB3";
  case burnman::types::EOSType::SLB3Conductive:
    return "SLB3Conductive";
  case burnman::types::EOSType::HPTMT:
    return "HPTMT";
  default:
    return "Unknown";
  }
}

#endif // TESTS_EOS_ENUM_STRING_HPP
