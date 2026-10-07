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

#include "burnman/utils/index_utils.hpp"
#include "internal.hpp"

namespace burnman::pseudosections::detail {
std::unique_ptr<EqualityConstraint>
continuation_plane(const Engine &engine, const Assemblage &a, Eigen::Index n,
                   const Eigen::Vector2d &target,
                   const Eigen::Vector2d &weights) {
  auto axes = diagram_axes(engine.section.type);
  double x = engine.composition_coordinate(a);
  return std::make_unique<SectionConstraint>(
      engine.coordinate_constraint(axes[0], target[0], n, x),
      engine.coordinate_constraint(axes[1], target[1], n, x), weights);
}
Eigen::VectorXd continuation_tangent(const Engine &engine,
                                     const optim::roots::DampedNewtonResult &s,
                                     const Assemblage &a,
                                     const Eigen::Vector2d &range,
                                     int *coordinate_rank) {
  if (s.x.size() != engine.parameters(a).n_parameters) {
    if (coordinate_rank)
      *coordinate_rank = 0;
    return Eigen::VectorXd();
  }
  auto scales = engine.parameter_scales(a, s.x.size(), range);
  // Row 0 fixes the phase amount or contour target; row 1 fixes the moving
  // plane. Keep all remaining equations, including the extra fixed coordinate
  // in X sections. Rescaling balances their sizes without changing equilibrium.
  Eigen::MatrixXd j(s.J.rows() - 1, s.J.cols());
  j.topRows(1) = s.J.topRows(1);
  j.bottomRows(s.J.rows() - 2) = s.J.bottomRows(s.J.rows() - 2);
  j = j * scales.asDiagonal();
  for (int k = 0; k < j.rows(); ++k) {
    double norm = j.row(k).norm();
    if (norm > 0)
      j.row(k) /= norm;
  }
  // Very small phase amounts make some equation directions hard to distinguish.
  // A tighter rank threshold retains their coupling to P,T at empty sites.
  double rank_tolerance =
      engine.settings.active_solution_faces ? 1.e-12 : 1.e-9;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeFullV);
  int rank = burnman::utils::checked_int(
      (svd.singularValues().array() > rank_tolerance).count());
  Eigen::MatrixXd null = svd.matrixV().rightCols(j.cols() - rank);
  // Count independent ways equilibrium can move in the selected diagram.
  // Changing phase amounts can move S/V at constant P,T; other changes may
  // leave both plotted coordinates unchanged. Exactly one direction defines a
  // line.
  Eigen::JacobiSVD<Eigen::MatrixXd> projection(
      range.cwiseInverse().asDiagonal() *
          engine.coordinate_jacobian(a, s.x.size()) * scales.asDiagonal() *
          null,
      Eigen::ComputeThinU | Eigen::ComputeThinV);
  int dim = burnman::utils::checked_int(
      (projection.singularValues().array() > 1.e-7).count());
  if (coordinate_rank)
    *coordinate_rank = dim;
  if (dim != 1)
    return Eigen::VectorXd();
  auto d = (null * projection.matrixV().col(0)).eval();
  auto physical = (scales.asDiagonal() * d).eval();
  return physical /
         engine.project_direction(physical, a).cwiseQuotient(range).norm();
}
} // namespace burnman::pseudosections::detail
