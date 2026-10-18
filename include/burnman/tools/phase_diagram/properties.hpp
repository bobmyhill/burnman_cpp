// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once
#include "burnman/tools/phase_diagram/consistency.hpp"
#include "burnman/tools/phase_diagram/types.hpp"
#include <limits>

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
/// Negative curvature and undefined singular responses raise exceptions;
/// an unambiguous expansivity divergence returns signed infinity.
std::map<std::string, double>
assemblage_properties(const Assemblage &, const std::vector<std::string> &,
                      bool relaxed = true);

struct GridSolution {
  BoundaryPoint state;
  double n_moles = 0.;
  // Total G for inexpensive consistency checks during reverse/batched retries.
  // Legacy archives without it can reconstruct G from the retained solution.
  double gibbs = std::numeric_limits<double>::quiet_NaN();
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
  // Accepted/attempted saved Field; 0 when no closed loop supplies a number.
  Eigen::MatrixXi field_numbers;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> success;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> equilibrium_success;
  std::vector<std::string> diagnostics;
  std::vector<std::string> equilibrium_diagnostics;
  int equilibrium_solves = 0;
  ThermodynamicValidation thermodynamic_validation;
  // Temporary audit evidence; saved grids reconstruct it from their solutions.
  std::vector<detail::ThermodynamicSample> thermodynamic_samples;
  // Diagram metadata without the original samples or boundary geometry.
  Result context;
  // Empty unless requested; otherwise row-major, with nullopt at unsolved
  // points.
  std::vector<std::optional<GridSolution>> solutions;
};

/// Equilibrate candidate assemblages from the saved fields.
/// Coordinates use native diagram_axes order and SI units, including total
/// S/V on the saved bulk scale. Candidates must match the source order.
/// Try the containing field first, including provisional/open labels, then
/// fields across shared boundaries within one grid spacing only on failure.
/// Successful assigned solves return directly when their extensive energies
/// and conjugate derivatives agree with neighbouring equilibria. Only flagged
/// neighbourhoods require independent phase discovery and root comparison.
/// Retain model-domain exclusions. Warm starts reuse accepted neighbours from
/// this row and the preceding row. Retry failed equilibria in reverse order,
/// using successful neighbours to the right and in the following row.
/// No boundary discovery is repeated. The
/// source and candidates are unchanged; failed solves/properties have
/// diagnostics.
PropertyGrid
phase_diagram_grid(const Result &,
                   const std::vector<std::shared_ptr<Material>> &candidates,
                   const std::vector<std::string> &properties,
                   const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                   bool relaxed = true, bool store_solutions = false);

/// Prepare source geometry once for bounded batches of grid points.
/// Supply the complete grid axes so boundary proximity is independent of batch
/// size. Each evaluate() accepts subsets and returns an independent grid.
/// Cache cloned candidate models, chemical metadata, field bounds and saved
/// field guesses for this gridder's lifetime. Retain only two rows of solved
/// warm starts across batches; no process-wide caches are used.
class PropertyGridder {
public:
  PropertyGridder(Result source,
                  std::vector<std::shared_ptr<Material>> candidates,
                  Eigen::VectorXd x, Eigen::VectorXd y);
  PropertyGrid evaluate(const std::vector<std::string> &properties,
                        const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                        bool relaxed = true, bool store_solutions = false);
  /// Retry failed or thermodynamically inconsistent equilibria in descending
  /// row/column order. Consistent accepted states are preserved. Saved
  /// solutions prime the two warm-start rows;
  /// successive batches must arrive in reverse order with their states
  /// retained.
  void retry(PropertyGrid &);
  /// Audit final retained states and joins with preceding audited batches.
  /// No equilibrium solves. reset starts an independent traversal.
  void audit(PropertyGrid &, bool reverse = false, bool reset = false);

private:
  struct FieldCache {
    Eigen::Vector2d low, high;
    std::optional<State> seed;
    bool seed_ready = false;
  };
  void calculate(PropertyGrid &, bool reverse);
  Result source_;
  std::unique_ptr<detail::Engine> engine_;
  FieldPolygons geometry_;
  Eigen::Vector2d origin_, range_, physical_seed_;
  std::vector<FieldCache> fields_;
  std::array<Eigen::VectorXd, 2> grid_axes_;
  std::map<int, std::vector<std::size_t>> boundary_fields_;
  std::map<double, State> previous_row_, current_row_;
  std::optional<double> row_coordinate_;
  bool reverse_ = false;
  std::map<double, detail::ThermodynamicSample> audit_previous_, audit_current_;
  std::optional<double> audit_row_;
  bool audit_reverse_ = false;
};

/// Evaluate more properties from retained states, preserving the exact active
/// faces and both solved/excluded masks. Performs zero equilibrium solves.
/// averaging_scheme sets the elastic averaging used for acoustic velocities.
PropertyGrid
evaluate_property_grid(const PropertyGrid &,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const std::vector<std::string> &properties,
                       bool relaxed = true,
                       types::AveragingType averaging_scheme =
                           types::AveragingType::VRH);
} // namespace burnman::phase_diagrams
