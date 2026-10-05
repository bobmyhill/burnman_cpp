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

#include "bindings.hpp"
#include <limits>

namespace burnman::python {
eos::excesses::ExcessParamVector
parse_modifiers(const py::iterable &modifiers) {
  using namespace eos::excesses;
  ExcessParamVector result;
  for (const auto item : modifiers) {
    const auto pair = py::cast<py::sequence>(item);
    if (pair.size() != 2)
      throw std::invalid_argument(
          "Modifiers must be (name, parameters) pairs.");
    const auto name = py::cast<std::string>(pair[0]);
    const auto p = py::cast<py::dict>(pair[1]);
    auto value = [&p](const char *key) {
      if (!p.contains(key))
        throw std::invalid_argument(
            std::string("Missing modifier parameter: ") + key);
      double x = py::cast<double>(p[key]);
      if (!std::isfinite(x))
        throw std::invalid_argument("Modifier parameters must be finite.");
      return x;
    };
    if (name == "linear") {
      result.emplace_back(
          LinearParams{value("delta_V"), value("delta_S"), value("delta_E")});
    } else if (name == "bragg_williams") {
      double n = value("n");
      if (n < 1.0 || n > std::numeric_limits<int>::max() ||
          n != std::floor(n)) {
        throw std::invalid_argument(
            "Bragg-Williams n must be a positive integer.");
      }
      result.emplace_back(
          BraggWilliamsParams{static_cast<int>(n), value("factor"), value("Wh"),
                              value("Wv"), value("deltaH"), value("deltaV")});
    } else if (name == "landau") {
      result.emplace_back(
          LandauParams{value("Tc_0"), value("V_D"), value("S_D")});
    } else if (name == "landau_slb_2022") {
      result.emplace_back(
          LandauSLB2022Params{value("Tc_0"), value("V_D"), value("S_D")});
    } else if (name == "landau_hp") {
      result.emplace_back(LandauHPParams{value("T_0"), value("P_0"),
                                         value("Tc_0"), value("V_D"),
                                         value("S_D")});
    } else if (name == "magnetic_chs") {
      auto finite_pair = [&p](const char *key) {
        if (!p.contains(key))
          throw std::invalid_argument(
              std::string("Missing modifier parameter: ") + key);
        auto xs = py::cast<std::vector<double>>(p[key]);
        if (xs.size() != 2 || !std::isfinite(xs[0]) || !std::isfinite(xs[1])) {
          throw std::invalid_argument(
              "Magnetic modifier parameters require two finite values.");
        }
        return xs;
      };
      // Pure Python BurnMan calls this parameter curie_temperature.
      auto tc = finite_pair(p.contains("curie_T") ? "curie_T"
                                                  : "curie_temperature"),
           moment = finite_pair("magnetic_moment");
      result.emplace_back(MagneticChsParams{
          value("structural_parameter"), tc[0], tc[1], moment[0], moment[1]});
    } else if (name == "debye") {
      result.emplace_back(DebyeParams{value("Cv_inf"), value("Theta_0")});
    } else if (name == "debye_delta") {
      result.emplace_back(DebyeDeltaParams{value("S_inf"), value("Theta_0")});
    } else if (name == "einstein") {
      result.emplace_back(EinsteinParams{value("Cv_inf"), value("Theta_0")});
    } else if (name == "einstein_delta") {
      result.emplace_back(
          EinsteinDeltaParams{value("S_inf"), value("Theta_0")});
    } else {
      throw std::invalid_argument("Unknown property modifier: " + name);
    }
  }
  return result;
}
} // namespace burnman::python
