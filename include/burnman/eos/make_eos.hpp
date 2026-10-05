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

#ifndef BURNMAN_EOS_MAKE_EOS_HPP_INCLUDED
#define BURNMAN_EOS_MAKE_EOS_HPP_INCLUDED

#include "burnman/core/equation_of_state.hpp"
#include "burnman/utils/types/simple_types.hpp"
#include <memory>

namespace burnman {
namespace eos {
/**
 * @brief Makes a pointer to a predefined EOS class
 * @param eos_type types::EOSType enum specifying equation of state
 * @return std::shared_ptr to an instance of the specific EOS
 */
std::shared_ptr<EquationOfState> make_eos(types::EOSType eos_type);
} // namespace eos
} // namespace burnman

#endif // BURNMAN_EOS_MAKE_EOS_HPP_INCLUDED
