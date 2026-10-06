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
#include <algorithm>
#include <map>
namespace burnman::pseudosections::detail {
using namespace equilibration;
inline double cross(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
  return a.x() * b.y() - a.y() * b.x();
}
inline double segment_distance(const Eigen::Vector2d &p,
                               const Eigen::Vector2d &a,
                               const Eigen::Vector2d &b) {
  Eigen::Vector2d d = b - a;
  double t = d.squaredNorm() > 0.
                 ? std::clamp((p - a).dot(d) / d.squaredNorm(), 0., 1.)
                 : 0.;
  return (p - a - t * d).norm();
}
// One ring predicate for indexed planar faces and contour matrices:
// -1 outside, 0 on the boundary, +1 inside.
template <typename Vertex>
int ring_location(std::size_t n, Vertex vertex, const Eigen::Vector2d &p,
                  double tolerance) {
  bool inside = false;
  for (std::size_t i = 1; i < n; ++i) {
    Eigen::Vector2d a = vertex(i - 1), b = vertex(i);
    if (segment_distance(p, a, b) <= tolerance)
      return 0;
    if ((a.y() > p.y()) != (b.y() > p.y()) &&
        p.x() < a.x() + (p.y() - a.y()) * (b.x() - a.x()) / (b.y() - a.y()))
      inside = !inside;
  }
  return inside ? 1 : -1;
}
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
  Eigen::Vector2d physical_seed{1.e9, 1000.};
  std::array<double, 2> pressure_bounds{0., 150.e9},
      temperature_bounds{1., 6000.};
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
                     double, double,
                     std::optional<Eigen::Vector2d> = std::nullopt);
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
// Successful samples inconsistent with their containing region: sample,
// polygon indices. Ignore the uncertainty band around approximated chords.
std::vector<std::pair<std::size_t, std::size_t>>
field_conflicts(const Result &, const FieldPolygons &);
// Continuation removes constraint row 1 (the moving section plane). This
// shared tangent includes latent S/V amount directions as well as P/T/X.
Eigen::VectorXd continuation_tangent(const Engine &,
                                     const optim::roots::DampedNewtonResult &,
                                     const Assemblage &,
                                     const Eigen::Vector2d &,
                                     int *coordinate_rank = nullptr);
std::unique_ptr<EqualityConstraint>
continuation_plane(const Engine &, const Assemblage &, Eigen::Index,
                   const Eigen::Vector2d &target,
                   const Eigen::Vector2d &weights);
// A continuation plane in any two thermodynamic coordinates. Its derivatives
// come from the same equality constraints used by equilibrate().
class SectionConstraint : public EqualityConstraint {
  std::unique_ptr<EqualityConstraint> first_, second_;
  Eigen::Vector2d weights_;

public:
  SectionConstraint(std::unique_ptr<EqualityConstraint> first,
                    std::unique_ptr<EqualityConstraint> second,
                    const Eigen::Vector2d &weights)
      : first_(std::move(first)), second_(std::move(second)),
        weights_(weights) {}
  std::unique_ptr<EqualityConstraint> clone() const override {
    return std::make_unique<SectionConstraint>(first_->clone(),
                                               second_->clone(), weights_);
  }
  double evaluate(const Eigen::VectorXd &x,
                  const Assemblage &a) const override {
    return weights_[0] * first_->evaluate(x, a) +
           weights_[1] * second_->evaluate(x, a);
  }
  Eigen::VectorXd derivative(const Eigen::VectorXd &x, const Assemblage &a,
                             Eigen::Index n) const override {
    return weights_[0] * first_->derivative(x, a, n) +
           weights_[1] * second_->derivative(x, a, n);
  }
};
} // namespace burnman::pseudosections::detail
