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

#include "burnman/core/seismic.hpp"
#include "../data/read_table.hpp"
#include "../data/seismic_data.hpp"
#include "burnman/utils/constants.hpp"
#include "burnman/utils/exceptions.hpp"
#include "burnman/utils/warnings.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace burnman::seismic {
namespace {
using Array = Eigen::ArrayXd;
using burnman::data::read_table;

void check_coordinates(const Array &x) {
  if (x.size() == 0 || !x.isFinite().all())
    throw std::invalid_argument(
        "Seismic table coordinates must be nonempty and finite.");
  for (Eigen::Index i = 1; i < x.size(); ++i)
    if (x[i] < x[i - 1])
      throw std::invalid_argument(
          "Seismic table coordinates must be increasing.");
}

Array interpolate(const Array &query, const Array &x, const Array &y) {
  check_coordinates(x);
  if (y.size() != x.size())
    throw std::invalid_argument(
        "Seismic property table is missing or has the wrong length.");
  Array result(query.size());
  for (Eigen::Index i = 0; i < query.size(); ++i) {
    const double value = query[i];
    if (std::isnan(value)) {
      result[i] = value;
    } else if (value < x[0]) {
      result[i] = y[0];
    } else if (value >= x[x.size() - 1]) {
      result[i] = y[y.size() - 1];
    } else {
      const auto upper = std::upper_bound(x.data(), x.data() + x.size(), value);
      const Eigen::Index j = upper - x.data() - 1;
      const double fraction = (value - x[j]) / (x[j + 1] - x[j]);
      result[i] = y[j] + fraction * (y[j + 1] - y[j]);
    }
  }
  return result;
}

// The three-point formula used by numpy.gradient(..., edge_order=2),
// including second-order one-sided endpoints on nonuniform coordinates.
Array gradient(const Array &y, const Array &x) {
  Array result(x.size());
  if (!x.isFinite().all())
    throw std::invalid_argument("Bullen coordinates must be finite.");
  const bool increasing = x[1] > x[0];
  for (Eigen::Index i = 1; i < x.size(); ++i)
    if (increasing ? x[i] <= x[i - 1] : x[i] >= x[i - 1])
      throw std::invalid_argument(
          "Bullen coordinates must be strictly monotonic.");
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const Eigen::Index j = std::clamp<Eigen::Index>(i - 1, 0, x.size() - 3);
    const double h1 = x[j + 1] - x[j];
    const double h2 = x[j + 2] - x[j + 1];
    const double t = x[i] - x[j];
    const double a = (2.0 * t - 2.0 * h1 - h2) / (h1 * (h1 + h2));
    const double b = (h1 + h2 - 2.0 * t) / (h1 * h2);
    const double c = (2.0 * t - h1) / (h2 * (h1 + h2));
    result[i] = a * y[j] + b * y[j + 1] + c * y[j + 2];
  }
  return result;
}

void load_prem(SeismicTable &model) {
  const auto table = read_table(data::prem, 8);
  model.table_depth = table.col(0);
  model.table_radius = table.col(1);
  model.table_pressure = table.col(2);
  model.table_density = table.col(3);
  model.table_vp = table.col(4);
  model.table_vs = table.col(5);
  model.table_QK = table.col(6);
  model.table_QG = table.col(7);
}

void load_regional(SeismicTable &model, const char *s_data,
                   const char *p_data) {
  const auto prem = read_table(data::prem, 8);
  const Eigen::ArrayXXd s = read_table(s_data, 2).colwise().reverse();
  const Eigen::ArrayXXd p = read_table(p_data, 2).colwise().reverse();
  std::vector<Eigen::Index> rows;
  const double min_radius = model.earth_radius - s.col(0).maxCoeff();
  const double max_radius = model.earth_radius - s.col(0).minCoeff();
  for (Eigen::Index i = 0; i < prem.rows(); ++i)
    if (prem(i, 1) >= min_radius && prem(i, 1) <= max_radius)
      rows.push_back(i);
  const auto n = static_cast<Eigen::Index>(rows.size());
  model.table_depth.resize(n);
  model.table_radius.resize(n);
  model.table_pressure.resize(n);
  model.table_density.resize(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto row = rows[static_cast<std::size_t>(i)];
    model.table_depth[i] = prem(row, 0);
    model.table_radius[i] = prem(row, 1);
    model.table_pressure[i] = prem(row, 2);
    model.table_density[i] = prem(row, 3);
  }
  model.table_vp = interpolate(model.table_depth, p.col(0), p.col(1));
  model.table_vs = interpolate(model.table_depth, s.col(0), s.col(1));
}
} // namespace

Eigen::ArrayXd Seismic1DModel::internal_depth_list(double, double,
                                                   double) const {
  throw exceptions::NotImplementedError("Seismic1DModel",
                                        "internal_depth_list");
}

#define ABSTRACT_QUERY(name)                                                   \
  Array Seismic1DModel::name(const Array &) const {                            \
    throw exceptions::NotImplementedError("Seismic1DModel", #name);            \
  }
ABSTRACT_QUERY(pressure)
ABSTRACT_QUERY(v_p)
ABSTRACT_QUERY(v_s)
ABSTRACT_QUERY(density)
ABSTRACT_QUERY(QK)
ABSTRACT_QUERY(QG)
ABSTRACT_QUERY(depth)
ABSTRACT_QUERY(gravity)
#undef ABSTRACT_QUERY

#define SCALAR_QUERY(name)                                                     \
  double Seismic1DModel::name(double value) const {                            \
    return name(Array::Constant(1, value))[0];                                 \
  }
SCALAR_QUERY(pressure)
SCALAR_QUERY(v_p)
SCALAR_QUERY(v_s)
SCALAR_QUERY(v_phi)
SCALAR_QUERY(density)
SCALAR_QUERY(G)
SCALAR_QUERY(K)
SCALAR_QUERY(QK)
SCALAR_QUERY(QG)
SCALAR_QUERY(depth)
SCALAR_QUERY(gravity)
#undef SCALAR_QUERY

Array Seismic1DModel::v_phi(const Array &z) const {
  return (v_p(z).square() - (4.0 / 3.0) * v_s(z).square()).sqrt();
}
Array Seismic1DModel::G(const Array &z) const {
  return v_s(z).square() * density(z);
}
Array Seismic1DModel::K(const Array &z) const {
  return v_phi(z).square() * density(z);
}

Eigen::ArrayXXd
Seismic1DModel::evaluate(const std::vector<std::string> &vars_list,
                         const Array &depth_list) const {
  Eigen::ArrayXXd values(static_cast<Eigen::Index>(vars_list.size()),
                         depth_list.size());
  for (std::size_t i = 0; i < vars_list.size(); ++i) {
    Array result;
    const auto &name = vars_list[i];
#define EVALUATE_QUERY(property)                                               \
  if (name == #property)                                                       \
    result = property(depth_list);                                             \
  else
    EVALUATE_QUERY(pressure)
    EVALUATE_QUERY(density)
    EVALUATE_QUERY(gravity)
    EVALUATE_QUERY(v_s)
    EVALUATE_QUERY(v_p)
    EVALUATE_QUERY(v_phi)
    EVALUATE_QUERY(G)
    EVALUATE_QUERY(K)
    EVALUATE_QUERY(QG)
    EVALUATE_QUERY(QK)
#undef EVALUATE_QUERY
    throw std::invalid_argument("Unknown seismic property: " + name);
    if (result.size() != depth_list.size())
      throw std::invalid_argument(
          "Seismic property query returned the wrong length.");
    values.row(static_cast<Eigen::Index>(i)) = result.transpose();
  }
  return values;
}

Eigen::ArrayXXd
Seismic1DModel::evaluate(const std::vector<std::string> &vars_list) const {
  return evaluate(vars_list, internal_depth_list());
}

Array SeismicTable::internal_depth_list(double mindepth, double maxdepth,
                                        double discontinuity_interval) const {
  check_coordinates(table_depth);
  if (!std::isfinite(discontinuity_interval) || discontinuity_interval < 0.0)
    throw std::invalid_argument(
        "discontinuity_interval must be finite and nonnegative.");
  std::vector<double> selected;
  for (Eigen::Index i = 0; i < table_depth.size(); ++i)
    if (table_depth[i] >= mindepth && table_depth[i] <= maxdepth)
      selected.push_back(table_depth[i]);
  const auto n = static_cast<Eigen::Index>(selected.size());
  Array result = Eigen::Map<const Array>(selected.data(), n);
  // Identify all repeated rows before shifting either side, as in Python.
  for (Eigen::Index i = 0; i + 1 < n; ++i)
    if (selected[static_cast<std::size_t>(i)] ==
        selected[static_cast<std::size_t>(i + 1)])
      result[i] -= discontinuity_interval;
  for (Eigen::Index i = 0; i + 1 < n; ++i)
    if (selected[static_cast<std::size_t>(i)] ==
        selected[static_cast<std::size_t>(i + 1)])
      result[i + 1] += discontinuity_interval;
  return result;
}

Array SeismicTable::_lookup(const Array &z, const Array &value_table) const {
  return interpolate(z, table_depth, value_table);
}
Array SeismicTable::pressure(const Array &z) const {
  if (table_pressure.size() == 0) {
    utils::warn("Pressure is not given in this seismic model and is now being "
                "computed. "
                "This requires density for the entire planet.");
    _compute_pressure();
  }
  return _lookup(z, table_pressure);
}
Array SeismicTable::gravity(const Array &z) const {
  if (table_gravity.size() == 0) {
    utils::warn(
        "Gravity is not given in this seismic model and is now being computed. "
        "This requires density for the entire planet.");
    _compute_gravity();
  }
  return _lookup(z, table_gravity);
}
Array SeismicTable::v_p(const Array &z) const { return _lookup(z, table_vp); }
Array SeismicTable::v_s(const Array &z) const { return _lookup(z, table_vs); }
Array SeismicTable::QK(const Array &z) const { return _lookup(z, table_QK); }
Array SeismicTable::QG(const Array &z) const { return _lookup(z, table_QG); }
Array SeismicTable::density(const Array &z) const {
  if (table_density.size() == 0)
    throw std::invalid_argument(
        "Density has not been defined for this seismic model.");
  return _lookup(z, table_density);
}

Array SeismicTable::depth(const Array &p) const {
  if (table_pressure.size() == 0)
    pressure(table_depth);
  check_coordinates(table_pressure);
  for (Eigen::Index i = 0; i < p.size(); ++i)
    if (p[i] < table_pressure[0] ||
        p[i] > table_pressure[table_pressure.size() - 1])
      throw std::invalid_argument("Pressure outside range of SeismicTable.");
  return interpolate(p, table_pressure, table_depth);
}
Array SeismicTable::radius(const Array &p) const {
  return earth_radius - depth(p);
}
double SeismicTable::radius(double p) const { return earth_radius - depth(p); }

Array SeismicTable::bullen(const Array &z) const {
  if (z.size() <= 3)
    throw std::invalid_argument(
        "Bullen parameter requires at least four depths.");
  const Array phi = v_phi(z).square();
  const Array kappa = phi * density(z);
  const Array dkappa = gradient(kappa, pressure(z));
  const Array dphi = gradient(phi, z);
  const Array g = gravity(z);
  return (g.abs() > 1.0e-12)
      .select(dkappa - dphi / g, std::numeric_limits<double>::quiet_NaN());
}

void SeismicTable::_compute_gravity() const {
  if (table_density.size() == 0)
    throw std::invalid_argument(
        "Density profile is an empty list in this SeismicTable instance.");
  if (table_radius.size() != table_density.size() ||
      table_radius.size() != table_depth.size() ||
      !table_radius.isFinite().all() || !table_density.isFinite().all() ||
      (table_radius < 0.0).any())
    throw std::invalid_argument(
        "Density, radius and depth tables must have matching finite entries.");
  const Array radii = table_radius.reverse();
  const Array density = table_density.reverse();
  check_coordinates(radii);
  const Array integrand = constants::physics::gravitation * 4.0 *
                          std::acos(-1.0) * density * radii.square();
  Array g = Array::Zero(radii.size());
  double integral = 0.0;
  for (Eigen::Index i = 1; i < radii.size(); ++i) {
    integral +=
        0.5 * (integrand[i] + integrand[i - 1]) * (radii[i] - radii[i - 1]);
    g[i] = radii[i] > 0.0 ? integral / radii[i] / radii[i] : 0.0;
  }
  table_gravity = g.reverse();
}

void SeismicTable::_compute_pressure() const {
  if (table_radius.size() == 0 || table_density.size() != table_radius.size())
    throw std::invalid_argument(
        "Pressure integration requires matching density and radius tables.");
  const Array z = earth_radius - table_radius;
  check_coordinates(z);
  const Array integrand = gravity(z) * table_density;
  Array p = Array::Zero(z.size());
  for (Eigen::Index i = 1; i < z.size(); ++i)
    p[i] =
        p[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * (z[i] - z[i - 1]);
  table_pressure = std::move(p);
}

PREM::PREM() { load_prem(*this); }
Slow::Slow() { load_regional(*this, data::swave_slow, data::pwave_slow); }
Fast::Fast() { load_regional(*this, data::swave_fast, data::pwave_fast); }

STW105::STW105() {
  const Eigen::ArrayXXd table = read_table(data::STW105, 9).colwise().reverse();
  table_radius = table.col(0);
  table_density = table.col(1);
  table_vpv = table.col(2);
  table_vsv = table.col(3);
  table_QK = table.col(4);
  table_QG = table.col(5);
  table_vph = table.col(6);
  table_vsh = table.col(7);
  table_depth = earth_radius - table_radius;
  table_vs = ((2.0 * table_vsv.square() + table_vsh.square()) / 3.0).sqrt();
  table_vp = ((table_vpv.square() + 4.0 * table_vph.square()) / 5.0).sqrt();
}

IASP91::IASP91() {
  const auto table = read_table(data::iasp91, 4);
  table_depth = table.col(0);
  table_radius = table.col(1);
  table_vp = table.col(2);
  table_vs = table.col(3);
}
AK135::AK135() {
  const auto table = read_table(data::ak135, 7);
  table_depth = table.col(0);
  table_radius = table.col(1);
  table_density = table.col(2);
  table_vp = table.col(3);
  table_vs = table.col(4);
  table_QG = table.col(5);
  table_QK = table.col(6);
}

PREM &prem_model() {
  static PREM model;
  return model;
}
} // namespace burnman::seismic
