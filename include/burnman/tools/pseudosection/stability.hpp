// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#pragma once
#include "burnman/core/solution.hpp"
#include "burnman/core/solution_model.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"
#include "burnman/tools/pseudosection/common.hpp"

namespace burnman::pseudosections {
/// Same native stability search at a single P,T, useful for checking diagrams.
State stable_equilibrium(
    const types::FormulaMap &composition,
    const std::vector<std::shared_ptr<Material>> &candidates, double pressure,
    double temperature, const Settings &settings = Settings{});

} // namespace burnman::pseudosections

namespace burnman::pseudosections::detail {
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
// Per-calculation mutable workspace with cloned candidate materials.
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
  types::FormulaMap composition_at(double) const;
  void set_bulk(double);
  double composition_coordinate(const Assemblage &) const;
  void set_coordinate(Assemblage &, double) const;
  Eigen::Vector2d coordinates(const Assemblage &) const;
  Eigen::Vector2d physical_coordinates(const Eigen::Vector2d &) const;
  std::unique_ptr<EqualityConstraint>
  coordinate_constraint(Coordinate, double, Eigen::Index, double base_x = 0.,
                        bool normalized = false) const;
  ConstraintList state_constraints(const Eigen::Vector2d &) const;
  Eigen::MatrixXd coordinate_jacobian(const Assemblage &, Eigen::Index) const;
  Eigen::VectorXd parameter_scales(const Assemblage &, Eigen::Index,
                                   const Eigen::Vector2d &) const;
  Eigen::Vector2d project_direction(const Eigen::VectorXd &,
                                    const Assemblage &) const;
  bool direct_coordinates() const;
  template <typename Point>
  void record_coordinates(Point &p, const Assemblage &a) const {
    p.pressure = a.get_pressure();
    p.temperature = a.get_temperature();
    p.entropy = a.get_n_moles() * a.get_molar_entropy();
    p.volume = a.get_n_moles() * a.get_molar_volume();
    p.composition_coordinate = composition_coordinate(a);
  }
  // The free X parameter measures a change from the assemblage's current X.
  EquilibrationParameters parameters(const Assemblage &) const;
  // *_at accepts the selected diagram coordinates in their native units.
  WorkState stable_at(const Eigen::Vector2d &);
  // Equilibrate the supplied phases and check for lower-energy alternatives.
  WorkState fixed_at(const std::vector<int> &, const PhaseStates &,
                     const Eigen::Vector2d &);

  std::shared_ptr<Assemblage> make_at(const std::vector<int> &,
                                      const PhaseStates &,
                                      const Eigen::Vector2d &);
  bool eos_bulk_feasible_at(const Eigen::Vector2d &);
  void set_pt(double, double);
  bool eos_bulk_feasible(double, double);
  double energy(int, const Eigen::VectorXd &) const;
  Eigen::MatrixXd feasible_vertices(const Phase &) const;
  Minimum minimize_phase(const Phase &, const Eigen::VectorXd &,
                         const Eigen::VectorXd &);
  Minimum minimize(int, const Eigen::VectorXd &,
                   const Eigen::VectorXd &start = Eigen::VectorXd());
  std::vector<Minimum> minima(int, const Eigen::VectorXd &,
                              const Eigen::VectorXd &start = Eigen::VectorXd());
  // Discover stable phases at physical P,T (Pa, K).
  WorkState stable(double, double);
  // Refine phase amounts/compositions at P,T, or at requested diagram
  // coordinates.
  WorkState
  equilibrate_phase_set(const std::vector<int> &, const PhaseStates &, double,
                        double, std::optional<Eigen::Vector2d> = std::nullopt);
  std::shared_ptr<Assemblage> make_assemblage(const std::vector<int> &,
                                              const PhaseStates &, double,
                                              double,
                                              double face_tolerance = 1.e-7);
  std::shared_ptr<Assemblage> copy_assemblage(const Assemblage &) const;
  // Changes the trial assemblage, including on failure. Checks mass balance,
  // reactions and constraints; lower-energy alternatives need a separate check.
  EquilibriumSolve solve(Assemblage &, ConstraintList &,
                         bool vary_composition = true);
  Eigen::VectorXd potentials(const Assemblage &) const;
  // Return the most negative candidate affinity, or zero if none is negative.
  // Optionally return each candidate's best composition and affinity.
  double stability(const Assemblage &, std::vector<Minimum> * = nullptr);
  double mass_error(const Assemblage &, bool vary_composition = true) const;
  PhaseStates snapshot(const Assemblage &, const std::vector<int> &) const;
  Eigen::MatrixXd composition_basis(const Material &, int) const;
  Eigen::MatrixXd phase_formulae(const std::vector<int> &ids,
                                 const PhaseStates &states) const;
  std::vector<int> active_phases(const State &) const;
  std::vector<int> canonical_ids(const std::vector<int> &) const;
  bool distinct(const Assemblage &, const std::vector<int> &) const;
};

std::shared_ptr<Material> clone(const std::shared_ptr<Material> &);

// Row 0 fixes the phase amount/contour target; row 1 fixes the moving plane.
ConstraintList constraints(std::unique_ptr<EqualityConstraint>,
                           std::unique_ptr<EqualityConstraint>);
} // namespace burnman::pseudosections::detail
