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
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/equilibration/equilibrate.hpp"
#include "burnman/tools/pseudosection.hpp"
#include <map>
namespace burnman::pseudosections::detail {
using namespace equilibration;
std::shared_ptr<Material> clone(const std::shared_ptr<Material> &);
struct Phase {
  std::shared_ptr<Material> material;
  std::shared_ptr<Solution> solution;
  Eigen::MatrixXd a, vertices, occupancies;
  Eigen::VectorXd g;
  bool available = true;
  std::string domain_error;
};
struct Minimum {
  Eigen::VectorXd p;
  double affinity = 0.;
};
struct WorkState {
  State state;
  std::shared_ptr<Assemblage> assemblage;
  std::vector<int> ids;
};
// Keep the solved X parameter exactly, rather than recovering it from mass
// residuals. Otherwise finite balance tolerance can leave frame endpoints open.
struct SectionAssemblage : Assemblage {
  double coordinate = 0.;
  bool has_coordinate = false;
};
struct FaceSolution : Solution {
  Eigen::MatrixXd original_basis;
  FaceSolution(const Solution &s, const Eigen::MatrixXd &b)
      : Solution(s), original_basis(b) {}
};
struct FaceMineral : Mineral {
  Eigen::MatrixXd original_basis;
  FaceMineral(const Mineral &m, const Eigen::MatrixXd &b)
      : Mineral(m), original_basis(b) {}
};
struct Engine {
  types::FormulaMap bulk;
  CompositionSection section;
  types::FormulaMap bulk_start;
  Eigen::VectorXd full_start, bulk_direction;
  std::vector<FreeVectorMap> free_vectors;
  double fixed_pressure = 0., fixed_temperature = 0.;
  Settings settings;
  std::vector<Phase> phases;
  std::vector<std::string> elements;
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
  std::array<Eigen::Index, 2> coordinate_indices(Eigen::Index) const;
  Eigen::Vector2d project_direction(const Eigen::VectorXd &) const;
  EquilibrationParameters parameters(const Assemblage &) const;
  WorkState stable_at(const Eigen::Vector2d &);
  WorkState fixed_at(const std::vector<int> &, const std::vector<PhaseState> &,
                     const Eigen::Vector2d &);
  std::shared_ptr<Assemblage> make_at(const std::vector<int> &,
                                      const std::vector<PhaseState> &,
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
  WorkState stable(double, double);
  WorkState fixed_pt(const std::vector<int> &, const std::vector<PhaseState> &,
                     double, double);
  std::shared_ptr<Assemblage> make_assemblage(const std::vector<int> &,
                                              const std::vector<PhaseState> &,
                                              double, double,
                                              double face_tolerance = 1.e-7);
  std::shared_ptr<Assemblage> copy_assemblage(const Assemblage &) const;
  optim::roots::DampedNewtonResult solve(Assemblage &, ConstraintList &,
                                         bool vary_composition = true);
  Eigen::VectorXd potentials(const Assemblage &) const;
  double stability(const Assemblage &, std::vector<Minimum> * = nullptr);
  double mass_error(const Assemblage &, bool vary_composition = true) const;
  std::vector<PhaseState> snapshot(const Assemblage &,
                                   const std::vector<int> &) const;
  Eigen::MatrixXd composition_basis(const Material &, int) const;
};
ConstraintList constraints(std::unique_ptr<EqualityConstraint>,
                           std::unique_ptr<EqualityConstraint>);
} // namespace burnman::pseudosections::detail
