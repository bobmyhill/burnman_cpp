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
constexpr char hp_water[] = "PS94 H2O; Holland-Powell 2011 thermal reference";
std::shared_ptr<Material> named(std::shared_ptr<Material> phase,
                                const std::string &name) {
  phase->set_name(name);
  return phase;
}
} // namespace

ModelSet metapelite() {
  return {"HPx metapelite",
          "2022-01-23",
          "HP_2011_ds62",
          "https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/"
          "metapelite-set/",
          "https://doi.org/10.1111/jmg.12071",
          1.5e9,
          2.e9,
          {named(mp50NCKFMASHTO::mu(), "ms"),
           named(mp50NCKFMASHTO::bi(), "bi"),
           named(mp50NCKFMASHTO::chl(), "chl"),
           named(mp50NCKFMASHTO::ctd(), "ctd"),
           named(mp50NCKFMASHTO::st(), "st"),
           named(mp50NCKFMASHTO::cd(), "cd"),
           named(mp50NCKFMASHTO::g(), "g"),
           named(mp50NCKFMASHTO::pl4tr(), "fsp"),
           named(mp50NCKFMASHTO::ep(), "ep"),
           named(mp50NCKFMASHTO::opx(), "opx"),
           named(mp50NCKFMASHTO::sa(), "spr"),
           named(mp50NCKFMASHTO::sp(), "sp"),
           named(mp50NCKFMASHTO::ilmm(), "ilm"),
           named(mp50NCKFMASHTO::mt1(), "mt"),
           named(mp50NCKFMASHTO::liq(), "melt"),
           named(HP_2011_ds62::q(), "q"),
           named(HP_2011_ds62::trd(), "trd"),
           named(HP_2011_ds62::crst(), "crst"),
           named(HP_2011_ds62::coe(), "coe"),
           named(HP_2011_ds62::stv(), "stv"),
           named(HP_2011_ds62::andalusite(), "and"),
           named(HP_2011_ds62::ky(), "ky"),
           named(HP_2011_ds62::sill(), "sill"),
           named(HP_2011_ds62::law(), "law"),
           named(HP_2011_ds62::ru(), "ru"),
           named(water_fluid(), "H2O"),
           named(mp50NCKFMASHTO::ma(), "ma"),
           named(HP_2011_ds62::sph(), "sph"),
           named(HP_2011_ds62::cor(), "cor")},
          "",
          hp_water};
}

ModelSet metabasite(const std::string &clinopyroxene) {
  if (clinopyroxene != "dio" && clinopyroxene != "aug")
    throw std::invalid_argument(
        "Clinopyroxene must be 'dio' or 'aug'; use one model.");
  return {"HPx metabasite (" + clinopyroxene + ")",
          "2022-01-30",
          "HP_2011_ds62",
          "https://hpxeosandthermocalc.org/the-hpx-eos/the-hpx-eos-families/"
          "metabasite-set/",
          "https://doi.org/10.1111/jmg.12211",
          2.e9,
          3.e9,
          {named(mb50NCKFMASHTO::hb(), "hb"),
           named(clinopyroxene == "dio" ? mb50NCKFMASHTO::dio()
                                        : mb50NCKFMASHTO::aug(),
                 "cpx"),
           named(mb50NCKFMASHTO::opx(), "opx"),
           named(mb50NCKFMASHTO::g(), "g"),
           named(mb50NCKFMASHTO::ol(), "ol"),
           named(mb50NCKFMASHTO::pl4tr(), "pl"),
           named(mb50NCKFMASHTO::ep(), "ep"),
           named(mb50NCKFMASHTO::chl(), "chl"),
           named(mb50NCKFMASHTO::bi(), "bi"),
           named(mb50NCKFMASHTO::ilmm(), "ilm"),
           named(mb50NCKFMASHTO::L(), "melt"),
           named(HP_2011_ds62::q(), "q"),
           named(HP_2011_ds62::trd(), "trd"),
           named(HP_2011_ds62::crst(), "crst"),
           named(HP_2011_ds62::coe(), "coe"),
           named(HP_2011_ds62::stv(), "stv"),
           named(HP_2011_ds62::law(), "law"),
           named(HP_2011_ds62::ru(), "ru"),
           named(HP_2011_ds62::sph(), "sph"),
           named(water_fluid(), "H2O"),
           named(mb50NCKFMASHTO::mu(), "ms"),
           named(mb50NCKFMASHTO::sp(), "sp"),
           named(mb50NCKFMASHTO::abc(), "ab"),
           named(HP_2011_ds62::andalusite(), "and"),
           named(HP_2011_ds62::ky(), "ky"),
           named(HP_2011_ds62::sill(), "sill"),
           named(HP_2011_ds62::cor(), "cor")},
          "",
          hp_water};
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
      {named(dry ? ig51W24::ol_H18() : ig51G25::ol_H18(), "ol"),
       named(dry ? ig51W24::cpx_W24() : ig51G25::cpx_W24(), "cpx"),
       named(dry ? ig51W24::opx_W24() : ig51G25::opx_W24(), "opx"),
       named(dry ? ig51W24::g_W24() : ig51G25::g_W24(), "g"),
       named(dry ? ig51W24::spl_T21() : ig51G25::spl_T21(), "sp"),
       named(dry ? ig51W24::fsp_H22() : ig51G25::fsp_H22(), "fsp"),
       named(dry ? ig51W24::ilm_W24() : ig51G25::ilm_W24(), "ilm"),
       named(dry ? ig51W24::liq_W24d() : ig51G25::liq_G25w(), "melt"),
       named(HPx_ds636::q(), "q"), named(HPx_ds636::trd(), "trd"),
       named(HPx_ds636::crst(), "crst"), named(HPx_ds636::coe(), "coe"),
       named(HPx_ds636::stv(), "stv"), named(HPx_ds636::andalusite(), "and"),
       named(HPx_ds636::ky(), "ky"), named(HPx_ds636::sill(), "sill"),
       named(HPx_ds636::ru(), "ru"), named(HPx_ds636::sph(), "sph"),
       named(HPx_ds636::cor(), "cor")},
      dry ? "Anhydrous alkaline and subalkaline systems; no aqueous fluid."
          : "Hydrous and anhydrous subalkaline systems, including the "
            "G25 silicate-bearing aqueous fluid with PS94 H2O.",
      dry ? "none (anhydrous W24)" : hp_water};
  if (dry) {
    models.phases.insert(
        models.phases.end(),
        {named(ig51W24::nph_W24(), "nph"), named(ig51W24::kals_W24(), "kals"),
         named(ig51W24::lct_W24(), "lct"), named(ig51W24::mel_W24(), "mel")});
  } else {
    models.phases.insert(
        models.phases.end(),
        {named(ig51G25::fl_G25(), "fluid"), named(ig51G25::amp_G16(), "hb"),
         named(ig51G25::bi_G25(), "bi"), named(ig51G25::mu_W14(), "ms"),
         named(ig51G25::ep_H11(), "ep"), named(ig51G25::cd_G25(), "cd")});
  }
  return models;
}

ModelSet slb2024() {
  namespace slb = SLB_2024;
  ModelSet models{
      "SLB2024 mantle solids",
      "010123",
      "SLB_2024",
      "https://github.com/stixrude/HeFESTo_Parameters_010123/tree/"
      "ff99dd0cc6a183d0ae6ea402dad61e1d2093c918/phase",
      "https://doi.org/10.1093/gji/ggae126",
      std::nullopt,
      std::nullopt,
      {named(slb::olivine(), "ol"),
       named(slb::wadsleyite(), "wa"),
       named(slb::ringwoodite(), "ri"),
       named(slb::orthopyroxene(), "opx"),
       named(slb::clinopyroxene(), "cpx"),
       named(slb::c2c_pyroxene(), "hpcpx"),
       named(slb::garnet(), "gt"),
       named(slb::mg_fe_aluminous_spinel(), "sp"),
       named(slb::plagioclase(), "pl"),
       named(slb::ilmenite(), "ak"),
       named(slb::ferropericlase(), "fp"),
       named(slb::bridgmanite(), "bg"),
       named(slb::post_perovskite(), "ppv"),
       named(slb::calcium_ferrite_structured_phase(), "cf"),
       named(slb::new_aluminous_phase(), "nal"),
       named(slb::capv(), "capv"),
       named(slb::neph(), "neph"),
       named(slb::ky(), "ky"),
       named(slb::qtz(), "q"),
       named(slb::coes(), "coe"),
       named(slb::st(), "st"),
       named(slb::apbo(), "seif"),
       named(slb::wo(), "wo"),
       named(slb::pwo(), "pwo"),
       named(slb::fea(), "Fe-bcc"),
       named(slb::feg(), "Fe-fcc"),
       named(slb::fee(), "Fe-hcp")},
      "The 27 active HeFESTo phases; no melts or fluids. Ordinary solution "
      "models retain their spin endmembers for equilibration. The unused "
      "lppv definition is excluded. EOS domains depend on pressure, "
      "temperature and composition."};
  models.phase_glossary = {
      {"ol", "Olivine"},
      {"wa", "Wadsleyite"},
      {"ri", "Ringwoodite"},
      {"opx", "Orthopyroxene"},
      {"cpx", "Clinopyroxene"},
      {"hpcpx", "High-pressure C2/c pyroxene"},
      {"gt", "Garnet / majorite"},
      {"sp", "Spinel"},
      {"pl", "Plagioclase"},
      {"ak", "Akimotoite / corundum / hematite solution"},
      {"fp", "Ferropericlase, including high- and low-spin iron"},
      {"bg", "Bridgmanite, including high- and low-spin ferric iron"},
      {"ppv", "Post-perovskite"},
      {"cf", "Calcium-ferrite-structured solution"},
      {"nal", "New aluminous phase"},
      {"capv", "CaSiO3 perovskite"},
      {"neph", "Nepheline"},
      {"ky", "Kyanite"},
      {"q", "Quartz"},
      {"coe", "Coesite"},
      {"st", "Stishovite"},
      {"seif", "Seifertite"},
      {"wo", "Wollastonite"},
      {"pwo", "Pseudowollastonite"},
      {"Fe-bcc", "Alpha (bcc) metallic iron"},
      {"Fe-fcc", "Gamma (fcc) metallic iron"},
      {"Fe-hcp", "Epsilon (hcp) metallic iron"}};
  return models;
}

ModelSet hhph2013() {
  namespace hp = HHPH_2013;
  ModelSet models{
      "HHPH2013 dry NCFMAS solids",
      "authors' archived tc-NCFMAS.txt and tc-ds62.txt",
      "HHPH_2013",
      "https://hpxeosandthermocalc.org/wp-content/uploads/2022/06/"
      "hollandetal2013.zip",
      "https://doi.org/10.1093/petrology/egt035",
      30.0e9,
      std::nullopt,
      {hp::g_solution(), hp::fp_solution(), hp::mpv_solution(),
       hp::cpv_solution(), hp::cor_solution(), hp::cf_solution(),
       hp::nal_solution(), hp::aki_solution(), hp::ol_solution(),
       hp::wad_solution(), hp::ring_solution(), hp::cpx_solution(),
       hp::opx_solution(), hp::hpx_solution(), named(hp::coe(), "coe"),
       named(hp::stv(), "stv")},
      "Dry, all ferrous iron; no melt or fluid. Pyroxene Ca bounds are not "
      "intrinsic to these solution models."};
  models.phase_glossary = {{"g", "Garnet, including majorite and Na-garnet"},
                           {"fp", "Ferropericlase"},
                           {"mpv", "Mg-rich silicate perovskite (bridgmanite)"},
                           {"cpv", "Ca-rich silicate perovskite"},
                           {"cor", "Corundum solution"},
                           {"cf", "Calcium-ferrite-structured solution"},
                           {"nal", "NAL (new aluminous phase)"},
                           {"aki", "Akimotoite"},
                           {"ol", "Olivine"},
                           {"wad", "Wadsleyite"},
                           {"ring", "Ringwoodite"},
                           {"cpx", "Clinopyroxene"},
                           {"opx", "Orthopyroxene"},
                           {"hpx", "High-pressure clinoenstatite solution"},
                           {"coe", "Coesite"},
                           {"stv", "Stishovite"}};
  return models;
}
} // namespace burnman::minerals::model_sets
