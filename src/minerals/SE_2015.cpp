// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/minerals/SE_2015.hpp"
#include "burnman/utils/constants.hpp"
#include <utility>

namespace burnman::minerals::SE_2015 {
namespace {
std::shared_ptr<Mineral> iron(const char *name,
                              types::GibbsCoefficients coefficients) {
  auto mineral = std::make_shared<Mineral>();
  auto &p = mineral->params;
  p.name = name;
  p.formula = types::FormulaMap{{"Fe", 1.}};
  p.napfu = 1.;
  p.molar_mass = constants::chemistry::atomic_masses.at("Fe");
  p.equation_of_state = types::EOSType::BroshCalphad;
  p.gibbs_coefficients = std::move(coefficients);
  return mineral;
}
} // namespace

std::shared_ptr<Mineral> bcc_iron() {
  auto mineral =
      iron("BCC iron", {
                           {1811.,
                            {1225.7, 124.134, -23.5143, 77359., 0., 0., 0.,
                             -0.00439752, -5.8927e-08, 0., 0., 0., 0.}},
                           {6000.,
                            {-25383.6, 299.3126, -46., 0., 0., 0., 2.29603e+31,
                             0., 0., 0., 0., 0., 0.}},
                           {12000.,
                            {-25383.4, 299.3122, -45.99997, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.}},
                       });
  auto &p = mineral->params;
  p.V_0 = 7.05e-06;
  p.K_0 = 170000000000.;
  p.Kprime_0 = 6.2;
  p.theta_0 = 300.;
  p.grueneisen_0 = 1.55;
  p.delta_0 = 6.;
  p.delta_1 = 15.;
  p.b_0 = 1.;
  p.b_1 = 3.;
  mineral->set_property_modifier_params(
      {eos::excesses::MagneticChsParams{0.4, 1043., 0., 2.22, 0.}});
  mineral->set_method(types::EOSType::Auto);
  return mineral;
}

std::shared_ptr<Mineral> fcc_iron() {
  auto mineral =
      iron("FCC iron", {
                           {1811.,
                            {-236.7, 132.416, -24.6643, 77359., 0., 0., 0.,
                             -0.00375752, -5.8927e-08, 0., 0., 0., 0.}},
                           {6000.,
                            {-27097.4, 300.2526, -46., 0., 0., 0., 2.78854e+31,
                             0., 0., 0., 0., 0., 0.}},
                           {12000.,
                            {-27097.1, 300.2522, -45.99996, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.}},
                       });
  auto &p = mineral->params;
  p.V_0 = 6.826e-06;
  p.K_0 = 140000000000.;
  p.Kprime_0 = 8.;
  p.theta_0 = 250.;
  p.grueneisen_0 = 2.;
  p.delta_0 = 4.;
  p.delta_1 = 10.;
  p.b_0 = 1.;
  p.b_1 = 3.;
  mineral->set_property_modifier_params(
      {eos::excesses::MagneticChsParams{0.28, 67., 0., 0.7, 0.}});
  mineral->set_method(types::EOSType::Auto);
  return mineral;
}

std::shared_ptr<Mineral> hcp_iron() {
  auto mineral =
      iron("HCP iron", {
                           {1811.,
                            {-2480.08, 136.725, -24.6643, 77359., 0., 0., 0.,
                             -0.00375752, -5.8927e-08, 0., 0., 0., 0.}},
                           {6000.,
                            {-29340.8, 304.5616, -46., 0., 0., 0., 2.78854e+31,
                             0., 0., 0., 0., 0., 0.}},
                           {12000.,
                            {-29340.5, 304.5612, -45.99996, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.}},
                       });
  auto &p = mineral->params;
  p.V_0 = 6.677e-06;
  p.K_0 = 170000000000.;
  p.Kprime_0 = 5.5;
  p.theta_0 = 250.;
  p.grueneisen_0 = 2.85;
  p.delta_0 = 6.;
  p.delta_1 = 10.;
  p.b_0 = 0.7;
  p.b_1 = 2.49614;
  mineral->set_method(types::EOSType::Auto);
  return mineral;
}

std::shared_ptr<Mineral> liquid_iron() {
  auto mineral = iron(
      "Liquid iron",
      {
          {1811.,
           {13265.87, 117.5756, -23.5143, 77359., 0., 0., 0., -0.00439752,
            -5.8927e-08, 0., -3.675155e-21, 0., 0.}},
          {12000.,
           {-10838.8, 291.302, -46., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}},
      });
  auto &p = mineral->params;
  p.V_0 = 7.4602e-06;
  p.K_0 = 165000000000.;
  p.Kprime_0 = 4.4729;
  p.theta_0 = 250.;
  p.grueneisen_0 = 2.;
  p.delta_0 = 6.;
  p.delta_1 = 4.;
  p.b_0 = 1.;
  p.b_1 = 5.10624;
  mineral->set_method(types::EOSType::Auto);
  return mineral;
}
} // namespace burnman::minerals::SE_2015
