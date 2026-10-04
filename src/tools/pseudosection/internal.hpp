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
  Settings settings;
  std::vector<Phase> phases;
  std::vector<std::string> elements;
  std::vector<int> components;
  Eigen::VectorXd b, full_bulk, potential_seed;
  int equilibrium_solves = 0, minimization_calls = 0;
  Engine(const types::FormulaMap &,
         const std::vector<std::shared_ptr<Material>> &, const Settings &);
  void set_pt(double, double);
  bool eos_bulk_feasible(double, double);
  double energy(int, const Eigen::VectorXd &) const;
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
  optim::roots::DampedNewtonResult solve(Assemblage &, ConstraintList &);
  Eigen::VectorXd potentials(const Assemblage &) const;
  double stability(const Assemblage &, std::vector<Minimum> * = nullptr);
  double mass_error(const Assemblage &) const;
  std::vector<PhaseState> snapshot(const Assemblage &,
                                   const std::vector<int> &) const;
  Eigen::MatrixXd composition_basis(const Material &, int) const;
};
ConstraintList constraints(std::unique_ptr<EqualityConstraint>,
                           std::unique_ptr<EqualityConstraint>);
} // namespace burnman::pseudosections::detail
