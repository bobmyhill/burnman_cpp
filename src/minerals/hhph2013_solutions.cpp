// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

// HHPH2013 NCFMAS activities and the authors' archived tc-NCFMAS.txt values.
// Energies below are kJ/mol and volume interactions are kJ/(mol kbar);
// the helpers convert both to SI. The archived fscf reference, cats ordering
// and coesite enthalpy are handled here or in the HHPH_2013 catalogue.

#include "burnman/core/combined_mineral.hpp"
#include "burnman/core/solution_models/asymmetric_regular_solution.hpp"
#include "burnman/core/solution_models/symmetric_regular_solution.hpp"
#include "burnman/minerals/datasets.hpp"
#include <utility>

namespace burnman::minerals::HHPH_2013 {
namespace {
using Rows = std::vector<std::vector<double>>;
using Members = types::PairedEndmemberList;

Mineral raw(std::shared_ptr<Mineral> (*factory)()) { return *factory(); }

Mineral combined(std::vector<Mineral> minerals,
                 std::initializer_list<double> amounts, const std::string &name,
                 double energy = 0.0, double entropy = 0.0,
                 double volume = 0.0) {
  Eigen::ArrayXd coefficients(static_cast<Eigen::Index>(amounts.size()));
  Eigen::Index i = 0;
  for (double amount : amounts)
    coefficients[i++] = amount;
  return make_combined_mineral(minerals, coefficients,
                               Eigen::Vector3d(energy, entropy, volume), name);
}

Mineral adjusted(Mineral mineral, const std::string &name, double energy,
                 double entropy = 0.0, double volume = 0.0) {
  return combined({mineral}, {1.0}, name, energy, entropy, volume);
}

Rows si(Rows rows, double factor) {
  for (auto &row : rows)
    for (double &value : row)
      value *= factor;
  return rows;
}

std::shared_ptr<Solution> solution(const std::string &name, Members members,
                                   Rows energies, Rows volumes = {},
                                   std::vector<double> alphas = {}) {
  std::shared_ptr<solution_models::SolutionModel> model;
  if (alphas.empty())
    model = std::make_shared<solution_models::SymmetricRegularSolution>(
        members, si(std::move(energies), 1.0e3),
        si(std::move(volumes), 1.0e-5));
  else
    model = std::make_shared<solution_models::AsymmetricRegularSolution>(
        members, std::move(alphas), si(std::move(energies), 1.0e3),
        si(std::move(volumes), 1.0e-5));
  auto phase = std::make_shared<Solution>();
  phase->set_solution_model(model);
  phase->set_name(name);
  phase->set_composition(
      Eigen::ArrayXd::Constant(static_cast<Eigen::Index>(members.size()),
                               1.0 / static_cast<double>(members.size())));
  return phase;
}

std::shared_ptr<Solution> perovskite(const std::string &name,
                                     std::initializer_list<double> offsets) {
  const std::vector<Mineral> minerals = {raw(mpv), raw(fpv), raw(cpv), raw(apv),
                                         raw(npv)};
  const std::vector<std::string> sites = {"[Mg][Si]O3", "[Fe][Si]O3",
                                          "[Ca][Si]O3", "[Al][Al]O3",
                                          "[Na1/2Al1/2][Si]O3"};
  Members members;
  std::size_t i = 0;
  for (double offset : offsets) {
    const std::string names[] = {"mpv", "fpv", "cpv", "apv", "npv"};
    members.emplace_back(adjusted(minerals[i], names[i], offset * 1.0e3),
                         sites[i]);
    ++i;
  }
  return solution(name, std::move(members),
                  {{12, 15, 20, 22}, {10.5, 14, 15.4}, {5, 7.5}, {2.5}});
}

std::shared_ptr<Solution> olivine_family(const std::string &name,
                                         std::shared_ptr<Mineral> (*mg)(),
                                         std::shared_ptr<Mineral> (*fe)(),
                                         double interaction) {
  return solution(name, {{raw(mg), "[Mg]2SiO4"}, {raw(fe), "[Fe]2SiO4"}},
                  {{interaction}});
}

std::shared_ptr<Solution> pyroxene(const std::string &name,
                                   std::shared_ptr<Mineral> (*mg)(),
                                   std::shared_ptr<Mineral> (*fe)()) {
  const Mineral mg_mineral = raw(mg);
  const Mineral fe_mineral = raw(fe);
  Mineral tschermak = name == "opx"
                          ? raw(mgts)
                          : combined({raw(mgts), raw(hen), raw(en)},
                                     {1.0, 1.0, -1.0}, "hmts", -1000.0);
  Members members = {
      {mg_mineral, "[Mg][Mg][Si]1/2O6"},
      {fe_mineral, "[Fe][Fe][Si]1/2O6"},
      {combined({mg_mineral, fe_mineral}, {0.5, 0.5}, "fm", -6000.0),
       "[Mg][Fe][Si]1/2O6"},
      {adjusted(raw(di), "odi", -100.0, -0.211, 5.0e-8), "[Mg][Ca][Si]1/2O6"},
      {tschermak, "[Al][Mg][Al1/2Si1/2]1/2O6"}};
  return solution(name, std::move(members),
                  {{5.2, 4, 32.2, 13}, {4, 24, 7}, {18, 2}, {75.4}},
                  {{0, 0, 0.12, -0.15}, {0, 0, -0.15}, {0, -0.15}, {-0.94}},
                  {1, 1, 1, 1.2, 1});
}
} // namespace

std::shared_ptr<Solution> g_solution() {
  Members members = {{raw(py), "[Mg]3[Al]2Si3O12"},
                     {raw(alm), "[Fe]3[Al]2Si3O12"},
                     {raw(gr), "[Ca]3[Al]2Si3O12"},
                     {raw(maj), "[Mg]3[Mg1/2Si1/2]2Si3O12"},
                     {combined({raw(alm), raw(maj), raw(py)},
                               {1.0 / 3.0, 1.0, -1.0 / 3.0}, "gfm", -10000.0),
                      "[Mg]3[Fe1/2Si1/2]2Si3O12"},
                     {raw(nagt), "[Mg2/3Na1/3]3[Al1/2Si1/2]2Si3O12"}};
  return solution("g", std::move(members),
                  {{3, 33, 15, 13.5, 14},
                   {5, 18, 16.5, 11.2},
                   {48, 46.5, 30},
                   {0.5, 8.5},
                   {7}});
}

std::shared_ptr<Solution> fp_solution() {
  return solution("fp", {{raw(per), "[Mg]O"}, {raw(fper), "[Fe]O"}}, {{18}});
}

std::shared_ptr<Solution> mpv_solution() {
  return perovskite("mpv", {0, -9.5, 60, 0, 16});
}

std::shared_ptr<Solution> cpv_solution() {
  return perovskite("cpv", {35, 24, 0, 45, 25});
}

std::shared_ptr<Solution> cor_solution() {
  Members members = {
      {raw(cor), "[Al][Al]O3"},
      {raw(mcor), "[Mg][Si]O3"},
      {combined({raw(mcor), raw(fak), raw(mak)}, {1, 1, -1}, "fcor", -15000.0),
       "[Fe][Si]O3"}};
  return solution("cor", std::move(members), {{12, 10}, {4}});
}

std::shared_ptr<Solution> cf_solution() {
  Members members = {
      {raw(macf), "[Mg][Al]O4"},
      {raw(cacf), "[Ca][Al]O4"},
      {raw(mscf), "[Mg][Mg1/2Si1/2]O4"},
      {raw(fscf), "[Fe][Fe1/2Si1/2]O4"},
      {combined({raw(fscf), raw(mscf)}, {0.5, 0.5}, "oscf", -3500.0),
       "[Fe][Mg1/2Si1/2]O4"},
      {adjusted(raw(nacf), "nacf", -9000.0), "[Na][Al1/2Si1/2]O4"}};
  return solution("cf", std::move(members),
                  {{11, 7.5, 10.75, 11.5, 24.5},
                   {18.5, 14.45, 15.2, 18.5},
                   {5, 4, 15.5},
                   {1, 9.95},
                   {10.7}});
}

std::shared_ptr<Solution> nal_solution() {
  Members members = {
      {raw(nanal), "[Na][Mg]2[Al5/6Si1/6]3O12"},
      {raw(canal), "[Ca][Mg]2[Al]3O12"},
      {raw(manal), "[Mg][Mg]2[Al]3O12"},
      {raw(msnal), "[Mg][Mg]2[Mg1/2Si1/2]3O12"},
      {raw(fsnal), "[Fe][Fe]2[Fe1/2Si1/2]3O12"},
      {combined({raw(fsnal), raw(msnal)}, {1.0 / 6.0, 5.0 / 6.0}, "o1nal",
                2000.0),
       "[Fe][Mg]2[Mg1/2Si1/2]3O12"},
      {combined({raw(fsnal), raw(msnal)}, {0.5, 0.5}, "o2nal", 6500.0),
       "[Fe][Fe]2[Mg1/2Si1/2]3O12"}};
  return solution("nal", std::move(members),
                  {{14.2, 20.2, 20.2, 21.2, 15.4, 23.4},
                   {11, 33.5, 36, 30.2, 38.2},
                   {22.5, 32.2, 26.5, 34.5},
                   {15, 4, 12},
                   {11, 3},
                   {8}});
}

std::shared_ptr<Solution> aki_solution() {
  return solution("aki",
                  {{raw(cor), "[Al]1/2[Al]1/2O3"},
                   {raw(mak), "[Mg]1/2[Si]1/2O3"},
                   {raw(fak), "[Fe]1/2[Si]1/2O3"}},
                  {{8, 6}, {4}});
}

std::shared_ptr<Solution> ol_solution() {
  return olivine_family("ol", fo, fa, 9);
}

std::shared_ptr<Solution> wad_solution() {
  return olivine_family("wad", mwd, fwd, 13);
}

std::shared_ptr<Solution> ring_solution() {
  return olivine_family("ring", mrw, frw, 4);
}

std::shared_ptr<Solution> cpx_solution() {
  Members members = {
      {raw(di), "[Mg][Ca][Si]1/2O6"},
      {adjusted(raw(fs), "cfs", 3800.0, 3.0, 3.0e-7), "[Fe][Fe][Si]1/2O6"},
      {raw(cats), "[Al][Ca][Si1/2Al1/2]1/2O6"},
      {raw(jd), "[Al][Na][Si]1/2O6"},
      {adjusted(raw(en), "cen", 3500.0, 2.0, 4.8e-7), "[Mg][Mg][Si]1/2O6"},
      {combined({raw(en), raw(fs)}, {0.5, 0.5}, "cfm", -3000.0),
       "[Mg][Fe][Si]1/2O6"}};
  return solution(
      "cpx", std::move(members),
      {{20, 12.3, 26, 29.8, 18}, {25, 36, 7, 4}, {6, 45.7, 27}, {40, 40}, {4}},
      {{0, -0.1, 0, -0.03, 0}, {-0.1, 0, 0, 0}, {0, -0.29, -0.1}, {0, 0}, {0}},
      {1.2, 1, 1.9, 1.2, 1, 1});
}

std::shared_ptr<Solution> opx_solution() { return pyroxene("opx", en, fs); }

std::shared_ptr<Solution> hpx_solution() { return pyroxene("hpx", hen, hfs); }
} // namespace burnman::minerals::HHPH_2013
