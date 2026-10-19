// ------------------------------------------------------
//
// SPDX-License-Identifier: GPL-3.0-or-later
//
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
//
// This file is part of BurnMan.
//
// Detailed license information governing the source code
// and contributions can be found in the LICENSE document.
//
// ------------------------------------------------------

#include "burnman/minerals/phase_aliases.hpp"
#include "burnman/minerals/datasets.hpp"

namespace burnman::minerals::SLB24 {
const std::map<std::string, std::string> &phase_aliases() {
  static const std::map<std::string, std::string> aliases{
      {olivine()->get_name(), "ol"},
      {wadsleyite()->get_name(), "wa"},
      {ringwoodite()->get_name(), "ri"},
      {orthopyroxene()->get_name(), "opx"},
      {clinopyroxene()->get_name(), "cpx"},
      {c2c_pyroxene()->get_name(), "hpcpx"},
      {garnet()->get_name(), "gt"},
      {mg_fe_aluminous_spinel()->get_name(), "sp"},
      {plagioclase()->get_name(), "pl"},
      {ilmenite()->get_name(), "ak"},
      {ferropericlase()->get_name(), "fp"},
      {bridgmanite()->get_name(), "bg"},
      {post_perovskite()->get_name(), "ppv"},
      {calcium_ferrite_structured_phase()->get_name(), "cf"},
      {new_aluminous_phase()->get_name(), "nal"},
      {capv()->get_name(), "capv"},
      {neph()->get_name(), "neph"},
      {ky()->get_name(), "ky"},
      {qtz()->get_name(), "q"},
      {coes()->get_name(), "coe"},
      {st()->get_name(), "st"},
      {apbo()->get_name(), "seif"},
      {wo()->get_name(), "wo"},
      {pwo()->get_name(), "pwo"},
      {lppv()->get_name(), "lppv"},
      {fea()->get_name(), "Fe-bcc"},
      {feg()->get_name(), "Fe-fcc"},
      {fee()->get_name(), "Fe-hcp"},
  };
  return aliases;
}
} // namespace burnman::minerals::SLB24
