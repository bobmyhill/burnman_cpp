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

#include "burnman/minerals/model_sets.hpp"
#include "burnman/minerals/datasets.hpp"
#include "burnman/minerals/water.hpp"
#include <stdexcept>

namespace burnman::minerals::model_sets {
namespace {
std::shared_ptr<Material> named(std::shared_ptr<Material> phase,
                                const std::string &name) {
  phase->set_name(name);
  return phase;
}
} // namespace

ModelSet metapelite() {
  return {"HPx metapelite",
          "2022-01-23",
          "HP11",
          "https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/"
          "metapelite-set/",
          "https://doi.org/10.1111/jmg.12071",
          1.5e9,
          2.e9,
          {named(MP14::mu(), "ms"),
           named(MP14::bi(), "bi"),
           named(MP14::chl(), "chl"),
           named(MP14::ctd(), "ctd"),
           named(MP14::st(), "st"),
           named(MP14::cd(), "cd"),
           named(MP14::g(), "g"),
           named(MP14::pl4tr(), "fsp"),
           named(MP14::ep(), "ep"),
           named(MP14::opx(), "opx"),
           named(MP14::sa(), "spr"),
           named(MP14::sp(), "sp"),
           named(MP14::ilmm(), "ilm"),
           named(MP14::mt1(), "mt"),
           named(MP14::liq(), "melt"),
           named(HP11::q(), "q"),
           named(HP11::andalusite(), "and"),
           named(HP11::ky(), "ky"),
           named(HP11::sill(), "sill"),
           named(HP11::law(), "law"),
           named(HP11::ru(), "ru"),
           named(water_fluid(), "H2O"),
           named(MP14::ma(), "ma"),
           named(HP11::sph(), "sph"),
           named(HP11::cor(), "cor")}};
}

ModelSet metabasite(const std::string &clinopyroxene) {
  if (clinopyroxene != "dio" && clinopyroxene != "aug")
    throw std::invalid_argument(
        "Clinopyroxene must be 'dio' or 'aug'; use one model.");
  return {"HPx metabasite (" + clinopyroxene + ")",
          "2022-01-30",
          "HP11",
          "https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/"
          "metabasite-set/",
          "https://doi.org/10.1111/jmg.12211",
          2.e9,
          3.e9,
          {named(MB16::hb(), "hb"),
           named(clinopyroxene == "dio" ? MB16::dio() : MB16::aug(), "cpx"),
           named(MB16::opx(), "opx"),
           named(MB16::g(), "g"),
           named(MB16::ol(), "ol"),
           named(MB16::pl4tr(), "pl"),
           named(MB16::ep(), "ep"),
           named(MB16::chl(), "chl"),
           named(MB16::bi(), "bi"),
           named(MB16::ilmm(), "ilm"),
           named(MB16::L(), "melt"),
           named(HP11::q(), "q"),
           named(HP11::law(), "law"),
           named(HP11::ru(), "ru"),
           named(HP11::sph(), "sph"),
           named(water_fluid(), "H2O"),
           named(MB16::mu(), "ms"),
           named(MB16::sp(), "sp"),
           named(MB16::abc(), "ab"),
           named(HP11::andalusite(), "and"),
           named(HP11::ky(), "ky"),
           named(HP11::sill(), "sill"),
           named(HP11::cor(), "cor")}};
}

ModelSet igneous(const std::string &calibration) {
  if (calibration != "G25" && calibration != "W24")
    throw std::invalid_argument("Igneous calibration must be 'G25' or 'W24'.");
  const bool dry = calibration == "W24";
  ModelSet models{
      "HPx igneous (" + calibration + ")",
      dry ? "2025-06-28 (W24)" : "2024-12-21 (G25)",
      "HPx_ds636",
      "https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/"
      "hpx-eos-igneous-sets/",
      dry ? "https://doi.org/10.1093/petrology/egae098"
          : "https://doi.org/10.1093/petrology/egae079",
      2.e9,
      3.e9,
      {named(dry ? IG24::ol_H18() : IG25::ol_H18(), "ol"),
       named(dry ? IG24::cpx_W24() : IG25::cpx_W24(), "cpx"),
       named(dry ? IG24::opx_W24() : IG25::opx_W24(), "opx"),
       named(dry ? IG24::g_W24() : IG25::g_W24(), "g"),
       named(dry ? IG24::spl_T21() : IG25::spl_T21(), "sp"),
       named(dry ? IG24::fsp_H22() : IG25::fsp_H22(), "fsp"),
       named(dry ? IG24::ilm_W24() : IG25::ilm_W24(), "ilm"),
       named(dry ? IG24::liq_W24d() : IG25::liq_G25w(), "melt"),
       named(HPx_ds636::q(), "q"), named(HPx_ds636::trd(), "trd"),
       named(HPx_ds636::crst(), "crst"), named(HPx_ds636::coe(), "coe"),
       named(HPx_ds636::andalusite(), "and"), named(HPx_ds636::ky(), "ky"),
       named(HPx_ds636::sill(), "sill"), named(HPx_ds636::ru(), "ru"),
       named(HPx_ds636::sph(), "sph"), named(HPx_ds636::cor(), "cor")},
      dry ? "Anhydrous alkaline and subalkaline systems; no aqueous fluid."
          : "Hydrous and anhydrous subalkaline systems, including the "
            "G25 silicate-bearing aqueous fluid with PS94 H2O."};
  if (dry) {
    models.phases.insert(models.phases.end(), {named(IG24::nph_W24(), "nph"),
                                               named(IG24::kals_W24(), "kals"),
                                               named(IG24::lct_W24(), "lct"),
                                               named(IG24::mel_W24(), "mel")});
  } else {
    models.phases.insert(
        models.phases.end(),
        {named(IG25::fl_G25(), "fluid"), named(IG25::amp_G16(), "hb"),
         named(IG25::bi_G25(), "bi"), named(IG25::mu_W14(), "ms"),
         named(IG25::ep_H11(), "ep"), named(IG25::cd_G25(), "cd")});
  }
  return models;
}
} // namespace burnman::minerals::model_sets
