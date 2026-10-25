// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/minerals/other.hpp"
#include "burnman/utils/constants.hpp"

namespace burnman::minerals::other {
std::shared_ptr<Mineral> liquid_iron() {
  auto mineral = std::make_shared<Mineral>();
  auto &p = mineral->params;
  const double mass = constants::chemistry::atomic_masses.at("Fe");
  p.name = "liquid iron";
  p.formula = types::FormulaMap{{"Fe", 1.}};
  p.equation_of_state = types::EOSType::AA;
  p.napfu = 1.;
  p.molar_mass = mass;
  p.P_0 = 1.e5;
  p.T_0 = 1811.;
  p.S_0 = 100.;
  p.E_0 = 0.;
  p.V_0 = mass / 7019.;
  p.K_0 = 109.7e9;
  p.Kprime_0 = 4.661;
  p.Kdprime_0 = -.043e-9;
  p.grueneisen_0 = 1.735;
  p.grueneisen_prime = -.130e-6 / mass;
  p.grueneisen_n = -1.870;
  p.aa_a = std::array<double, 2>{248.92 * mass, 289.48 * mass};
  p.aa_b = std::array<double, 2>{.4057 * mass, -1.1499 * mass};
  p.aa_Theta = std::array<double, 2>{1747.3, 1.537};
  p.theta = 5000.;
  p.aa_lmda =
      std::array<double, 3>{-325.23 * mass, 302.07 * mass, 30.45 * mass};
  p.xi_0 = 282.67 * mass;
  p.aa_F = std::array<double, 2>{7766. / 7019., 1146. / 7019.};
  mineral->set_method(types::EOSType::Auto);
  return mineral;
}
} // namespace burnman::minerals::other
