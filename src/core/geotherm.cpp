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

#include "burnman/core/geotherm.hpp"
#include "burnman/core/material.hpp"
#include "burnman/optim/roots/brent.hpp"
#include "geotherm_data.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace burnman::geotherm {
namespace {
using Array = Eigen::ArrayXd;

void check_table(const Array &coordinates, const Array &temperatures) {
  if (coordinates.size() == 0 || coordinates.size() != temperatures.size() ||
      !coordinates.isFinite().all() || !temperatures.isFinite().all())
    throw std::invalid_argument("Geotherm coordinates and temperatures must be "
                                "nonempty, finite and have matching lengths.");
  for (Eigen::Index i = 1; i < coordinates.size(); ++i)
    if (coordinates[i] < coordinates[i - 1])
      throw std::invalid_argument("Geotherm coordinates must be increasing.");
}

Array interpolate(const Array &query, const Array &x, const Array &y,
                  double minimum, double maximum,
                  const std::string &coordinate) {
  check_table(x, y);
  if (!query.isFinite().all())
    throw std::invalid_argument("Requested geotherm " + coordinate +
                                " must be finite.");
  Array result(query.size());
  for (Eigen::Index i = 0; i < query.size(); ++i) {
    if (query[i] < minimum || query[i] < x[0])
      throw std::invalid_argument("Requested " + coordinate +
                                  " is below the minimum geotherm " +
                                  coordinate + ".");
    if (query[i] > maximum || query[i] > x[x.size() - 1])
      throw std::invalid_argument("Requested " + coordinate +
                                  " is above the maximum geotherm " +
                                  coordinate + ".");
    const auto upper =
        std::lower_bound(x.data(), x.data() + x.size(), query[i]);
    const Eigen::Index j = upper - x.data();
    if (j == 0)
      result[i] = y[0];
    else {
      const double fraction = (query[i] - x[j - 1]) / (x[j] - x[j - 1]);
      result[i] = y[j - 1] * (1.0 - fraction) + y[j] * fraction;
    }
  }
  return result;
}

Eigen::ArrayXXd read_table(const char *text) {
  std::istringstream input(text);
  std::vector<std::pair<double, double>> values;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line.substr(0, line.find('#')));
    double coordinate, temperature;
    if (row >> coordinate >> temperature)
      values.emplace_back(coordinate, temperature);
  }
  Eigen::ArrayXXd table(static_cast<Eigen::Index>(values.size()), 2);
  for (Eigen::Index i = 0; i < table.rows(); ++i) {
    table(i, 0) = values[static_cast<std::size_t>(i)].first;
    table(i, 1) = values[static_cast<std::size_t>(i)].second;
  }
  return table;
}

std::shared_ptr<seismic::Seismic1DModel>
pressure_model(std::shared_ptr<seismic::Seismic1DModel> model) {
  if (model)
    return model;
  return std::shared_ptr<seismic::Seismic1DModel>(&seismic::prem_model(),
                                                  [](auto *) {});
}

void check_anchor(double temperature, double pressure) {
  if (!std::isfinite(temperature) || temperature <= 0.0 ||
      !std::isfinite(pressure))
    throw std::invalid_argument("An adiabat requires a finite positive anchor "
                                "temperature and a finite anchor pressure.");
}

struct EntropyResidual {
  Material &rock;
  double pressure, entropy;
  std::exception_ptr error;

  double operator()(double temperature) const {
    rock.set_state(pressure, temperature);
    const double residual = entropy - rock.get_molar_entropy();
    if (!std::isfinite(residual))
      throw std::runtime_error("Nonfinite entropy while solving the adiabat.");
    return residual;
  }
};

// Exceptions must not unwind through GSL's C callbacks. Save the original
// exception and rethrow after the native Brent wrapper has freed its solver.
double entropy_residual(double temperature, void *context) {
  auto &residual = **static_cast<EntropyResidual **>(context);
  try {
    return residual(temperature);
  } catch (...) {
    if (!residual.error)
      residual.error = std::current_exception();
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double solve_temperature(EntropyResidual &residual, double estimate) {
  const double floor = 1.0e-12;
  const double centre = std::max(floor, estimate);
  const double f0 = residual(centre);
  if (f0 == 0.0)
    return centre;
  double step = 1.0;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double left = std::max(floor, centre - step);
    const double right = centre + step;
    const double fl = residual(left);
    const double fr = residual(right);
    if (fl == 0.0)
      return left;
    if (fr == 0.0)
      return right;
    if (std::signbit(fl) != std::signbit(fr)) {
      double root;
      try {
        root = optim::roots::brent(&entropy_residual, &residual, left, right,
                                   2.0e-12, 1.0e-14);
      } catch (...) {
        if (residual.error)
          std::rethrow_exception(residual.error);
        throw;
      }
      if (residual.error)
        std::rethrow_exception(residual.error);
      return root;
    }
    step *= 1.618;
  }
  throw std::runtime_error(
      "Cannot bracket a constant-entropy temperature for this adiabat.");
}
} // namespace

Geotherm::Geotherm(const Array &depths, const Array &temperatures)
    : depths_array(depths), temperatures_array(temperatures) {
  check_table(depths_array, temperatures_array);
  minimum_depth = depths_array[0];
  maximum_depth = depths_array[depths_array.size() - 1];
}
Geotherm::Geotherm(const Eigen::ArrayXXd &table)
    : Geotherm(table.col(0), table.col(1)) {}
Array Geotherm::temperatures(const Array &depths) const {
  return interpolate(depths, depths_array, temperatures_array, minimum_depth,
                     maximum_depth, "depth");
}
double Geotherm::temperatures(double depth) const {
  return temperatures(Array::Constant(1, depth))[0];
}

BrownShankland::BrownShankland() : Geotherm(read_table(data::brown_81)) {}
Anderson::Anderson() : Geotherm(read_table(data::anderson_82)) {}
StaceyContinental::StaceyContinental()
    : Geotherm(read_table(data::Stacey_1977_continents)) {}
StaceyOceanic::StaceyOceanic()
    : Geotherm(read_table(data::Stacey_1977_oceans)) {}
Katsura2022::Katsura2022() : Geotherm(read_table(data::Katsura_2022)) {}
Plesa2022Mars6cm3::Plesa2022Mars6cm3()
    : Geotherm(read_table(data::Plesa_2022_Mars_V_6cm3)) {}

GeothermFromPressures::GeothermFromPressures(
    const Array &pressures, const Array &temperatures,
    std::shared_ptr<seismic::Seismic1DModel> model)
    : GeothermFromPressures(std::move(model)) {
  check_table(pressures, temperatures);
  pressures_array = pressures;
  temperatures_array = temperatures;
  minimum_pressure = pressures[0];
  maximum_pressure = pressures[pressures.size() - 1];
}
GeothermFromPressures::GeothermFromPressures(
    std::shared_ptr<seismic::Seismic1DModel> model)
    : depth_to_pressure_model(pressure_model(std::move(model))) {}
GeothermFromPressures::GeothermFromPressures(
    const Eigen::ArrayXXd &table,
    std::shared_ptr<seismic::Seismic1DModel> model)
    : GeothermFromPressures(table.col(0), table.col(1), std::move(model)) {}
Array GeothermFromPressures::temperatures(const Array &depths) const {
  if (!depth_to_pressure_model)
    throw std::invalid_argument(
        "A geotherm requires a depth-to-pressure model.");
  if (!depths.isFinite().all())
    throw std::invalid_argument("Requested geotherm depths must be finite.");
  const Array pressures = depth_to_pressure_model->pressure(depths);
  if (pressures.size() != depths.size())
    throw std::invalid_argument(
        "Depth-to-pressure model returned the wrong number of pressures.");
  return temperatures_from_pressures(pressures);
}
Array GeothermFromPressures::temperatures_from_pressures(
    const Array &pressures) const {
  return interpolate(pressures, pressures_array, temperatures_array,
                     minimum_pressure, maximum_pressure, "pressure");
}
double
GeothermFromPressures::temperatures_from_pressures(double pressure) const {
  return temperatures_from_pressures(Array::Constant(1, pressure))[0];
}
Anzellini2013::Anzellini2013(std::shared_ptr<seismic::Seismic1DModel> model)
    : GeothermFromPressures(read_table(data::Anzellini_2013_PT),
                            std::move(model)) {}

AdiabaticGeotherm::AdiabaticGeotherm(
    std::shared_ptr<Material> rock, double temperature, double pressure,
    std::shared_ptr<seismic::Seismic1DModel> model)
    : GeothermFromPressures(std::move(model)), material(std::move(rock)),
      T_0(temperature), P_0(pressure) {
  if (!material)
    throw std::invalid_argument("An adiabatic geotherm requires a material.");
  check_anchor(T_0, P_0);
}
Array AdiabaticGeotherm::temperatures_from_pressures(
    const Array &pressures) const {
  if (!material)
    throw std::invalid_argument("An adiabatic geotherm requires a material.");
  return adiabatic_profile(pressures, *material, T_0, P_0);
}

Array adiabatic_profile(const Array &pressures, Material &rock, double T_0,
                        std::optional<double> P_0) {
  if (!pressures.isFinite().all())
    throw std::invalid_argument("Adiabat pressures must be finite.");
  check_anchor(T_0, P_0.value_or(pressures.size() > 0 ? pressures[0] : 0.0));
  if (pressures.size() == 0)
    return Array(0);
  const double anchor = P_0.value_or(pressures[0]);
  rock.set_state(anchor, T_0);
  const double entropy = rock.get_molar_entropy();
  const double ratio = rock.get_grueneisen_parameter() /
                       rock.get_isentropic_bulk_modulus_reuss();
  if (!std::isfinite(entropy) || !std::isfinite(ratio))
    throw std::runtime_error("The material has nonfinite entropy or adiabatic "
                             "gradient at the anchor.");
  Array temperatures(pressures.size());
  try {
    double previous_pressure = anchor;
    double previous_temperature = T_0;
    for (Eigen::Index i = 0; i < pressures.size(); ++i) {
      if (pressures[i] == anchor) {
        temperatures[i] = T_0;
      } else {
        double estimate = previous_temperature *
                          (1.0 + ratio * (pressures[i] - previous_pressure));
        if (!std::isfinite(estimate) || estimate <= 0.0)
          estimate = previous_temperature;
        EntropyResidual residual{rock, pressures[i], entropy, nullptr};
        temperatures[i] = solve_temperature(residual, estimate);
      }
      previous_pressure = pressures[i];
      previous_temperature = temperatures[i];
    }
  } catch (...) {
    rock.set_state(anchor, T_0);
    throw;
  }
  rock.set_state(anchor, T_0);
  return temperatures;
}
} // namespace burnman::geotherm
