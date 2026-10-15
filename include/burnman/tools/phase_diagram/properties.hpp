// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/tools/phase_diagram/types.hpp"

namespace burnman::phase_diagrams {
/// Scalar Material property names accepted by assemblage_properties/grid.
std::vector<std::string> thermodynamic_property_names();

/// Evaluate an equilibrated assemblage. Relaxation holds elemental amounts
/// fixed and permits all reactions in its active endmember basis. Equations
/// 19–26 of Stixrude & Lithgow-Bertelloni (2022), doi:10.1093/gji/ggab394.
/// Molar responses are extensive derivatives divided by the current number
/// of moles, even when reactions change that number. No equilibrium solve is
/// performed. Supply ordinary Solutions with explicit equilibrated species
/// populations; pre-relaxed Solution derivatives would count a mode twice.
/// Singular/negative reaction curvature raises an exception.
std::map<std::string, double>
assemblage_properties(const Assemblage &, const std::vector<std::string> &,
                      bool relaxed = true);

struct GridSolution {
  BoundaryPoint state;
  double n_moles = 0.;
  Eigen::ArrayXd molar_fractions;
  // The exact active basis and coefficients used by the accepted solve.
  // Original candidate coefficients remain available in state.phases.
  std::vector<Eigen::MatrixXd> phase_bases;
  std::vector<Eigen::VectorXd> phase_compositions;
};

struct PropertyGrid {
  DiagramType diagram_type = DiagramType::PT;
  bool relaxed = true, resolved = true;
  Eigen::VectorXd x, y;
  // Arrays have shape (len(y), len(x)); unavailable properties are NaN.
  std::map<std::string, Eigen::MatrixXd> values;
  // 0 outside identified fields; otherwise the saved one-based Field number.
  Eigen::MatrixXi field_numbers;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> success;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> equilibrium_success;
  std::vector<std::string> diagnostics;
  std::vector<std::string> equilibrium_diagnostics;
  int equilibrium_solves = 0;
  // Diagram metadata without the original samples or boundary geometry.
  Result context;
  // Empty unless requested; otherwise row-major, with nullopt at unsolved
  // points.
  std::vector<std::optional<GridSolution>> solutions;
};

/// Equilibrate only the assemblage identified by each saved closed field.
/// Coordinates use native diagram_axes order and SI units, including total
/// S/V on the saved bulk scale. Candidates must match the source order.
/// Retain holes and exclusions; skip provisional/unidentified/open fields.
/// Points shared by fields use the lowest Field number (a one-sided value).
/// No boundary discovery or global stability search is repeated. The source
/// and candidates are unchanged. Failed solves/properties have diagnostics.
PropertyGrid
phase_diagram_grid(const Result &,
                   const std::vector<std::shared_ptr<Material>> &candidates,
                   const std::vector<std::string> &properties,
                   const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                   bool relaxed = true, bool store_solutions = false);

/// Evaluate more properties from retained states, preserving the exact active
/// faces and both solved/excluded masks. Performs zero equilibrium solves.
PropertyGrid
evaluate_property_grid(const PropertyGrid &,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const std::vector<std::string> &properties,
                       bool relaxed = true);
} // namespace burnman::phase_diagrams
