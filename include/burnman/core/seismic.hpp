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

#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace burnman::seismic {

/** One-dimensional seismic reference model, following classes/seismic.py.
 * All quantities use SI units. evaluate() returns properties by rows and depths
 * by columns. Derived models may override the vector queries; scalar queries
 * dispatch through the same virtual interface.
 */
class Seismic1DModel {
public:
  virtual ~Seismic1DModel() = default;
  virtual Eigen::ArrayXd
  internal_depth_list(double mindepth = 0.0, double maxdepth = 1.0e99,
                      double discontinuity_interval = 1.0) const;
  Eigen::ArrayXXd evaluate(const std::vector<std::string> &vars_list,
                           const Eigen::ArrayXd &depth_list) const;
  Eigen::ArrayXXd evaluate(const std::vector<std::string> &vars_list) const;

#define BURNMAN_SEISMIC_QUERY(name)                                            \
  virtual Eigen::ArrayXd name(const Eigen::ArrayXd &values) const;             \
  double name(double value) const
  BURNMAN_SEISMIC_QUERY(pressure);
  BURNMAN_SEISMIC_QUERY(v_p);
  BURNMAN_SEISMIC_QUERY(v_s);
  BURNMAN_SEISMIC_QUERY(v_phi);
  BURNMAN_SEISMIC_QUERY(density);
  BURNMAN_SEISMIC_QUERY(G);
  BURNMAN_SEISMIC_QUERY(K);
  BURNMAN_SEISMIC_QUERY(QK);
  BURNMAN_SEISMIC_QUERY(QG);
  BURNMAN_SEISMIC_QUERY(depth);
  BURNMAN_SEISMIC_QUERY(gravity);
#undef BURNMAN_SEISMIC_QUERY
};

/** Tabulated model sorted by increasing depth; repeated depths are allowed.
 * Queries clamp to the table endpoints, and exact discontinuities select the
 * deeper row, matching numpy.interp. internal_depth_list() offsets repeated
 * depths on either side by discontinuity_interval metres.
 *
 * Tables are public to support user-defined models as in Python BurnMan.
 * Missing pressure/gravity tables are computed lazily by trapezoidal
 * integration of density, with zero surface pressure and zero enclosed mass at
 * the centre. These integrations assume a density profile covering the entire
 * planet. After changing density/radii, clear pressure/gravity to recompute
 * them. Concurrent queries that populate these tables require external
 * synchronisation.
 */
class SeismicTable : public Seismic1DModel {
public:
  Eigen::ArrayXd table_depth, table_radius;
  mutable Eigen::ArrayXd table_pressure, table_gravity;
  Eigen::ArrayXd table_density, table_vp, table_vs, table_QG, table_QK;
  double earth_radius = 6371.0e3;

  Eigen::ArrayXd
  internal_depth_list(double mindepth = 0.0, double maxdepth = 1.0e10,
                      double discontinuity_interval = 1.0) const override;

#define BURNMAN_SEISMIC_TABLE_QUERY(name)                                      \
  using Seismic1DModel::name;                                                  \
  Eigen::ArrayXd name(const Eigen::ArrayXd &values) const override
  BURNMAN_SEISMIC_TABLE_QUERY(pressure);
  BURNMAN_SEISMIC_TABLE_QUERY(gravity);
  BURNMAN_SEISMIC_TABLE_QUERY(v_p);
  BURNMAN_SEISMIC_TABLE_QUERY(v_s);
  BURNMAN_SEISMIC_TABLE_QUERY(QK);
  BURNMAN_SEISMIC_TABLE_QUERY(QG);
  BURNMAN_SEISMIC_TABLE_QUERY(density);
  BURNMAN_SEISMIC_TABLE_QUERY(depth);
#undef BURNMAN_SEISMIC_TABLE_QUERY

  /// Inverse pressure coordinate, equivalent to earth_radius - depth(pressure).
  Eigen::ArrayXd radius(const Eigen::ArrayXd &pressure) const;
  double radius(double pressure) const;
  /// Bullen parameter using second-order nonuniform finite differences.
  Eigen::ArrayXd bullen(const Eigen::ArrayXd &depth) const;

  virtual Eigen::ArrayXd _lookup(const Eigen::ArrayXd &depth,
                                 const Eigen::ArrayXd &value_table) const;
  void _compute_gravity() const;
  void _compute_pressure() const;
};

class PREM : public SeismicTable {
public:
  PREM();
};
class Slow : public SeismicTable {
public:
  Slow();
};
class Fast : public SeismicTable {
public:
  Fast();
};
class STW105 : public SeismicTable {
public:
  STW105();
  Eigen::ArrayXd table_vpv, table_vsv, table_vph, table_vsh;
};
class IASP91 : public SeismicTable {
public:
  IASP91();
};
class AK135 : public SeismicTable {
public:
  AK135();
};

/// Shared PREM instance, constructed on first access.
PREM &prem_model();

} // namespace burnman::seismic
