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
#include "burnman/core/assemblage.hpp"
#include "burnman/core/solution.hpp"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>

namespace burnman::polytope {

class InfeasibleBulk : public std::domain_error {
public:
  using std::domain_error::domain_error;
};

/** H rows use [b, a...] with b + a*x == 0 or >= 0.
 * cddlib enumerates generators using GMP rational arithmetic. Input doubles
 * are approximated by small rationals within rational_tolerance (absolute);
 * zero tolerance retains their exact binary values. Outputs are doubles.
 * For solution polytopes, vertices are independent-endmember coordinates and
 * endmember_occupancies additionally maps them to site occupancies.
 */
class MaterialPolytope {
public:
  MaterialPolytope(const Eigen::MatrixXd &equalities,
                   const Eigen::MatrixXd &inequalities,
                   double rational_tolerance = 1.e-12);
  const Eigen::MatrixXd &get_equalities() const { return equalities_; }
  const Eigen::MatrixXd &get_inequalities() const { return inequalities_; }
  const Eigen::MatrixXd &get_vertices() const { return vertices_; }
  const Eigen::MatrixXd &get_rays() const { return rays_; }
  const Eigen::MatrixXd &get_lineality() const { return lineality_; }
  const Eigen::MatrixXd &get_endmember_occupancies() const {
    return occupancies_;
  }
  const Eigen::MatrixXd &
  get_endmembers_as_independent_endmember_amounts() const {
    return vertices_;
  }
  bool is_empty() const { return empty_; }
  bool is_bounded() const {
    return rays_.rows() == 0 && lineality_.rows() == 0;
  }
  void map_to_site_occupancies(const Eigen::MatrixXd &occupancies);

private:
  Eigen::MatrixXd equalities_, inequalities_, vertices_, rays_, lineality_,
      occupancies_;
  bool empty_ = false;
};

MaterialPolytope solution_polytope_from_endmember_occupancies(
    const Eigen::MatrixXd &endmember_occupancies,
    double rational_tolerance = 1.e-12);

MaterialPolytope composite_polytope_at_constrained_composition(
    const Assemblage &composite, const types::FormulaMap &composition,
    double rational_tolerance = 1.e-12);

/// Supports ideal and symmetric/asymmetric regular models, including signed
/// basis coordinates. A one-row basis returns a Mineral. Empty fractions use
/// the original composition if representable, otherwise a uniform mixture.
std::shared_ptr<Material> transform_solution_to_new_basis(
    const Solution &solution, const Eigen::MatrixXd &new_basis,
    const Eigen::ArrayXd &molar_fractions = Eigen::ArrayXd(),
    const std::string &solution_name = "");

/** Enumerate feasible assemblage amounts, remove absent phases and restrict
 * solutions to the smallest site-occupancy face containing those amounts.
 * Independent physical vertices of that face form the new basis. Unlike a
 * nonnegative coordinate constraint, this retains ordered/transformed models
 * whose valid independent-endmember coordinates can be negative.
 * Returns independently owned phases and models; never mutates the input.
 */
std::shared_ptr<Assemblage> simplify_composite_with_composition(
    const Assemblage &composite, const types::FormulaMap &composition,
    double tolerance = 1.e-10, double rational_tolerance = 1.e-12);
/// Native finite-pseudocompound Gibbs LP: min g.dot(n), A.transpose()*n=b,
/// n>=0. Returns component chemical potentials and compound amounts.
struct GibbsLPResult {
  Eigen::VectorXd chemical_potentials, amounts;
  double gibbs;
};
GibbsLPResult gibbs_linear_program(const Eigen::MatrixXd &stoichiometry,
                                   const Eigen::VectorXd &gibbs,
                                   const Eigen::VectorXd &bulk);
} // namespace burnman::polytope
