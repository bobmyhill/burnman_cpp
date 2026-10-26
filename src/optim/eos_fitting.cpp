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

#include "burnman/optim/eos_fitting.hpp"
#include "burnman/core/material.hpp"
#include <stdexcept>
#include <unordered_set>

namespace burnman::optim {
Eigen::VectorXd default_mle_tolerances(Material &material,
                                       const std::vector<std::string> &flags) {
  static const std::unordered_set<std::string> energy_flags = {
      "H",           "energy",          "molar_internal_energy",
      "helmholtz",   "molar_helmholtz", "gibbs",
      "molar_gibbs", "enthalpy",        "molar_enthalpy"};
  material.set_state(1.e5, 300.);
  Eigen::VectorXd tolerances(static_cast<Eigen::Index>(flags.size()));
  for (Eigen::Index i = 0; i < tolerances.size(); ++i) {
    const auto &flag = flags[static_cast<std::size_t>(i)];
    if (energy_flags.count(flag))
      tolerances[i] = 1.;
    else if (flag == "P" || flag == "pressure")
      tolerances[i] = 100.;
    else
      tolerances[i] = 1.e-5 * material.get_property(flag);
  }
  if (!(tolerances.array() > 0.).all())
    throw std::invalid_argument("All MLE tolerances should be > 0.");
  return tolerances;
}
} // namespace burnman::optim
