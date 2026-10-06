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
#include "burnman/core/solution.hpp"

namespace burnman {
/// Minimize internal endmember exchanges at fixed bulk composition and P,T,
/// with the relaxed thermal derivatives used by Python BurnMan.
class RelaxedSolution : public Solution {
public:
  RelaxedSolution(Solution solution, const Eigen::MatrixXd &relaxation_vectors,
                  const Eigen::MatrixXd &unrelaxed_vectors);
  void set_composition(const Eigen::ArrayXd &fractions) override;
  void set_composition(const Eigen::ArrayXd &fractions,
                       const Eigen::VectorXd &q_initial, bool relaxed);
  void set_state(double pressure, double temperature) override;
  void set_state(double pressure, double temperature, bool relaxed);
  Eigen::MatrixXd get_dndq() const { return dndq_; }
  Eigen::MatrixXd get_dndx() const { return dndx_; }
  Eigen::ArrayXd get_unrelaxed_vectors() const { return independent_; }

protected:
  double compute_isothermal_bulk_modulus_reuss() const override;
  double compute_thermal_expansivity() const override;
  double compute_molar_heat_capacity_p() const override;

private:
  Eigen::MatrixXd dndq_, dndx_;
  Eigen::ArrayXd independent_;
  bool state_set_ = false;
  void relax();
  Eigen::Matrix2d relaxed_hessian() const;
};
} // namespace burnman
