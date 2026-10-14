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
#include "burnman/core/solution_model.hpp"
#include "burnman/tools/pseudosection/common.hpp"
#include <utility>

namespace burnman::pseudosections::detail {
// At an ordinary critical point, two coexisting solution compositions merge.
// Use one copy to avoid a meaningless solution with two identical copies.
// Along the merging direction, the second and third derivatives of Gibbs
// energy must both vanish; these provide the two constraints locating the
// point.
class CriticalConstraint : public EqualityConstraint {
  std::shared_ptr<const solution_models::SolutionModel> model;
  Eigen::MatrixXd basis;
  Eigen::VectorXd reference;
  int start;
  bool third;
  Eigen::VectorXd fractions(const Eigen::VectorXd &) const;
  std::pair<double, Eigen::VectorXd> curvature(double, double,
                                               const Eigen::VectorXd &) const;
  // Gradient of directional curvature (the mixed third derivative), and
  // fourth derivative along mode. Shared by the corrector and certificate.
  std::pair<Eigen::VectorXd, double>
  higher_derivatives(double, double, const Eigen::VectorXd &,
                     const Eigen::VectorXd &) const;

public:
  CriticalConstraint(const Assemblage &, int, const EquilibrationParameters &,
                     const Eigen::VectorXd &, bool);
  std::unique_ptr<EqualityConstraint> clone() const override;
  double evaluate(const Eigen::VectorXd &, const Assemblage &) const override;
  Eigen::VectorXd derivative(const Eigen::VectorXd &, const Assemblage &,
                             Eigen::Index) const override;
  Eigen::VectorXd mode(const Assemblage &, int) const;
  // Call only after both critical constraints have converged. Require one
  // soft mode, stable transverse modes, and positive quartic curvature after
  // relaxing those modes: D4 - 3 T(v,v,.) H_stable^-1 T(v,v,.).
  bool stable(const Assemblage &, int) const;
};
} // namespace burnman::pseudosections::detail
