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
#include "burnman/core/solution_model.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"
#include "burnman/tools/phase_diagram/common.hpp"

namespace burnman::phase_diagrams {
/// Find a stable assemblage at one P (Pa) and T (K) for elemental mole amounts.
/// Search the supplied candidate minerals/solutions, refine reaction
/// equilibrium and test candidate affinities. The returned State contains phase
/// amounts, endmember proportions and validation errors, or a failure message.
State stable_equilibrium(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates, double pressure,
    double temperature, const Settings &settings = Settings{});

/// Single seed equilibrium at two of P,T,S,V, without tracing.
/// Uses exactly the phase-discovery routine used for the initial diagram grid.
/// S (J/K) and V (m^3) are totals on the supplied elemental mole scale.
/// Select PT, PS, PV, TS, TV or SV; P/T ranges guide inverse searches and
/// provide the initial physical seed. PS, TV and SV start from mass-balanced
/// phase amounts and use constrained total enthalpy, Helmholtz energy and
/// internal energy minimisation when the assemblage must change. Physical P-T
/// phase discovery is a fallback. Competing phases can enter the minimisations;
/// accepted states still pass reaction, mass-balance and candidate-affinity
/// checks.
State stable_equilibrium_at(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates,
    const Eigen::Vector2d &coordinates, DiagramType type,
    const Settings &settings = Settings{},
    const std::array<double, 2> &pressure_range = {0., 150.e9},
    const std::array<double, 2> &temperature_range = {1., 6000.});

} // namespace burnman::phase_diagrams

namespace burnman::phase_diagrams::detail {
struct Phase {
  std::shared_ptr<Material> material;
  std::shared_ptr<Solution> solution;
  // Rows of a give endmember formulae; rows of vertices give allowed
  // compositions. occupancies holds the site contents of each endmember.
  Eigen::MatrixXd a, vertices, occupancies;
  Eigen::VectorXd g; // endmember molar Gibbs energies at current P,T (J/mol)
  bool available = true;
  std::string domain_error;
};
struct Minimum {
  Eigen::VectorXd p; // proportions in the endmember basis being minimised
  // Phase molar Gibbs energy minus the value predicted from elemental chemical
  // potentials (J/mol). Negative values favour forming this phase.
  double affinity = 0.;
};
struct WorkState {
  State state;
  std::shared_ptr<Assemblage> assemblage;
  // Instance IDs in assemblage order.
  std::vector<int> ids;
};
// Keep the solved X parameter exactly, rather than recovering it from mass
// residuals. Otherwise finite balance tolerance can leave frame endpoints open.
struct SectionAssemblage : Assemblage {
  double coordinate = 0.;
  bool has_coordinate = false;
};
struct FaceSolution : Solution {
  // Each row expresses a reduced endmember in the original endmembers.
  // Multiplying the transpose by reduced proportions recovers the composition.
  Eigen::MatrixXd original_basis;
  FaceSolution(const Solution &s, const Eigen::MatrixXd &b)
      : Solution(s), original_basis(b) {}
};
struct FaceMineral : Mineral {
  // One row containing the original endmember coefficients of this face.
  Eigen::MatrixXd original_basis;
  FaceMineral(const Mineral &m, const Eigen::MatrixXd &b)
      : Mineral(m), original_basis(b) {}
};
// Thermodynamic workspace for one diagram. Candidate minerals are cloned so
// their state/composition can change without altering the caller's models.
// Phase selection combines a linear-programming bulk balance, solution
// composition minimisation, and constrained equilibrium refinement.
struct Engine {
  types::FormulaMap bulk;
  CompositionSection section;
  types::FormulaMap bulk_start;
  Eigen::VectorXd full_start, bulk_direction;
  std::vector<FreeVectorMap> free_vectors;
  Eigen::Vector2d physical_seed{1.e9, 1000.};
  std::array<double, 2> pressure_bounds{0., 150.e9},
      temperature_bounds{1., 6000.};
  Settings settings;
  std::vector<Phase> phases;
  std::vector<std::string> elements;
  // Elements needed to express the mass-balance equations without redundancy.
  // full_bulk retains every element for the final mass-balance check.
  std::vector<int> components;
  Eigen::VectorXd b, full_bulk, potential_seed;
  int equilibrium_solves = 0, minimization_calls = 0;
  Engine(const types::FormulaMap &,
         const std::vector<std::shared_ptr<Material>> &, const Settings &,
         const CompositionSection & = CompositionSection{});
  /// Elemental mole amounts at bulk mixing fraction X; totals are not
  /// normalised.
  types::FormulaMap composition_at(double) const;
  /// Set the working bulk and its mass-balance vectors to the requested X.
  void set_bulk(double);
  /// Read the assemblage's bulk mixing fraction, distinct from phase
  /// compositions.
  double composition_coordinate(const Assemblage &) const;
  /// Retain a solved X exactly rather than reconstructing it from mass
  /// residuals.
  void set_coordinate(Assemblage &, double) const;
  /// Selected diagram coordinates in native units, e.g. {P,S} for a P-S
  /// section.
  Eigen::Vector2d coordinates(const Assemblage &) const;
  /// P,T starting guess consistent with plotted/fixed P or T where supplied.
  Eigen::Vector2d physical_coordinates(
      const Eigen::Vector2d &,
      const std::optional<Eigen::Vector2d> & = std::nullopt) const;
  /// Equation fixing one physical coordinate; base_x anchors the free X change.
  std::unique_ptr<EqualityConstraint>
  coordinate_constraint(Coordinate, double, Eigen::Index, double base_x = 0.,
                        bool normalized = false) const;
  /// Constraints for an equilibrium at a chosen diagram position, including
  /// the additional fixed thermodynamic coordinate of an X section.
  ConstraintList state_constraints(const Eigen::Vector2d &) const;
  /// Derivatives of the selected coordinates with respect to solve parameters.
  Eigen::MatrixXd coordinate_jacobian(const Assemblage &, Eigen::Index) const;
  /// Parameter scales for conditioning continuation in the rescaled diagram.
  Eigen::VectorXd parameter_scales(const Assemblage &, Eigen::Index,
                                   const Eigen::Vector2d &) const;
  /// Project a change in solve parameters into the selected diagram axes.
  Eigen::Vector2d project_direction(const Eigen::VectorXd &,
                                    const Assemblage &) const;
  /// Whether selected/fixed P,T directly specify a state without an inverse
  /// solve.
  bool direct_coordinates() const;
  /// Save physical P,T and extensive S,V, even when only two of them are
  /// plotted.
  template <typename Point>
  void record_coordinates(Point &p, const Assemblage &a) const {
    p.pressure = a.get_pressure();
    p.temperature = a.get_temperature();
    p.entropy = a.get_n_moles() * a.get_molar_entropy();
    p.volume = a.get_n_moles() * a.get_molar_volume();
    p.composition_coordinate = composition_coordinate(a);
  }
  /// Equilibrium parameter layout and composition bounds for this assemblage.
  /// The free X parameter measures a change from the assemblage's current X.
  EquilibrationParameters parameters(const Assemblage &) const;
  /// Discover an equilibrium at diagram coordinates in native units. S/V
  /// sections also solve for the unplotted P and/or T using physical seed
  /// bounds. A successful neighbour supplies starting phases, compositions
  /// and amounts. Try both supplied neighbours before potential minimisation
  /// or phase discovery, retaining competing-phase and curvature checks.
  WorkState stable_at(const Eigen::Vector2d &, const WorkState *warm = nullptr,
                      const WorkState *alternative = nullptr);
  /// A mass-balanced starting assemblage from one linear balance solve.
  /// This does not solve a separate P-T phase equilibrium.
  WorkState feasible_start(double pressure, double temperature);
  /// Minimise total H (P,S), F (T,V), or U (S,V), with bulk and selected
  /// coordinates constrained. Add negative-affinity candidates between
  /// minimisations, then polish and validate with the equilibrium solver.
  WorkState minimize_potential(const WorkState &, const Eigen::Vector2d &);
  /// Equilibrate supplied phase IDs at a diagram position, using saved amounts
  /// and compositions as starting estimates, then test alternative candidates.
  WorkState fixed_at(const std::vector<int> &, const PhaseStates &,
                     const Eigen::Vector2d &,
                     const std::optional<Eigen::Vector2d> & = std::nullopt);

  /// Build a trial assemblage at a diagram position; no equilibrium is implied.
  /// Optional saved physical P,T supply unplotted coordinates and active faces.
  std::shared_ptr<Assemblage>
  make_at(const std::vector<int> &, const PhaseStates &,
          const Eigen::Vector2d &,
          const std::optional<Eigen::Vector2d> & = std::nullopt);
  /// Whether available EOS models can represent the bulk at a diagram position.
  bool eos_bulk_feasible_at(const Eigen::Vector2d &);
  /// Set all candidate P,T states and record diagnosed EOS-domain exclusions.
  void set_pt(double, double);
  /// Test bulk representability at physical P,T, respecting required EOS
  /// phases.
  bool eos_bulk_feasible(double, double);
  /// Candidate molar Gibbs energy at supplied endmember proportions.
  double energy(int, const Eigen::VectorXd &) const;
  /// Vertices of the allowed composition region, accounting for inactive sites.
  Eigen::MatrixXd feasible_vertices(const Phase &) const;
  /// Minimise one phase's tangent-plane affinity from a composition estimate.
  /// Elemental chemical potentials supply the reference plane.
  Minimum minimize_phase(const Phase &, const Eigen::VectorXd &,
                         const Eigen::VectorXd &);
  /// Best found affinity/composition for one candidate, using multiple starts.
  Minimum minimize(int, const Eigen::VectorXd &,
                   const Eigen::VectorXd &start = Eigen::VectorXd());
  /// Distinct composition minima for one candidate, allowing solvus branches.
  std::vector<Minimum> minima(int, const Eigen::VectorXd &,
                              const Eigen::VectorXd &start = Eigen::VectorXd());
  /// Discover stable phases at physical P,T (Pa, K), including solution copies.
  WorkState stable(double, double);
  /// Refine a specified assemblage at P,T, or at requested diagram coordinates.
  /// Remove phases driven to zero amount and check against other candidates.
  WorkState equilibrate_phase_set(const std::vector<int> &, const PhaseStates &,
                                  double, double,
                                  std::optional<Eigen::Vector2d> = std::nullopt,
                                  bool preserve_composition = false);
  /// Construct trial phases, reducing to occupied solution sites where needed.
  /// Saved zero-amount phases retain compositions to initialise reaction
  /// solves. preserve_composition prevents reduction from discarding positive
  /// trace constituents in an accepted state.
  std::shared_ptr<Assemblage>
  make_assemblage(const std::vector<int> &, const PhaseStates &, double, double,
                  double face_tolerance = 1.e-7,
                  bool preserve_composition = false);
  /// Clone a trial assemblage, including its reduced composition bases and X.
  std::shared_ptr<Assemblage> copy_assemblage(const Assemblage &) const;
  /// Solve reaction equilibrium, bulk balance and the supplied constraints.
  /// Changes the trial assemblage even on failure. Success checks those
  /// equations; stability against competing phases requires a separate
  /// stability() call. tolerance_scale tightens parameter and reaction
  /// tolerances for comparisons below ordinary phase-discovery accuracy.
  EquilibriumSolve solve(Assemblage &, ConstraintList &,
                         bool vary_composition = true,
                         double tolerance_scale = 1.);
  /// Elemental chemical potentials defining this assemblage's tangent plane.
  Eigen::VectorXd potentials(const Assemblage &) const;
  /// Search candidate compositions for negative tangent-plane affinities.
  /// Return the most negative affinity (J/mol), or zero if none is found;
  /// optionally retain each candidate's best found composition and affinity.
  double stability(const Assemblage &, std::vector<Minimum> * = nullptr);
  /// Relative elemental mass-balance error, using the current or solved X bulk.
  double mass_error(const Assemblage &, bool vary_composition = true) const;
  /// Save phase amounts/proportions in the original candidate endmember bases.
  PhaseStates snapshot(const Assemblage &, const std::vector<int> &) const;
  /// Map reduced active-site endmembers back to the candidate's original basis.
  Eigen::MatrixXd composition_basis(const Material &, int) const;
  /// Elemental formula rows for the supplied phase compositions, in ID order.
  Eigen::MatrixXd phase_formulae(const std::vector<int> &ids,
                                 const PhaseStates &states) const;
  /// Canonical IDs of phases above the relative amount cutoff in a saved state.
  std::vector<int> active_phases(const State &) const;
  /// Assemblage identity retaining candidate multiplicity, ignoring copy order.
  std::vector<int> canonical_ids(const std::vector<int> &) const;
  /// Reject near-identical copies of the same solution as distinct phases.
  bool distinct(const Assemblage &, const std::vector<int> &) const;
  /// Each solution composition must be a local Gibbs minimum on its occupied
  /// site face. Affinity tolerance alone cannot reject near-critical saddles.
  bool locally_stable(const Assemblage &) const;
};

/// Copy a mineral/solution model, including any reduced active-site basis.
std::shared_ptr<Material> clone(const std::shared_ptr<Material> &);

/// Two constraint groups used by continuation: phase amount/contour target,
/// followed by the plane through which the next boundary point is corrected.
ConstraintList constraints(std::unique_ptr<EqualityConstraint>,
                           std::unique_ptr<EqualityConstraint>);
} // namespace burnman::phase_diagrams::detail
