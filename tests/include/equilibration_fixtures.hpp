#ifndef TESTS_EQUILIBRATION_FIXTURES_HPP_INCLUDED
#define TESTS_EQUILIBRATION_FIXTURES_HPP_INCLUDED

#include "solution_fixtures.hpp"
#include <cmath>

// Matching elastic endmembers isolate an analytically soluble partitioning
// problem: each phase favours a different species by 10 kJ/mol.
inline Solution make_partitioning_oxide(const std::string &name,
                                        double mg_shift, double fe_shift) {
  FerropericlaseFixture fixture;
  Mineral mg = fixture.periclase;
  Mineral fe = mg;
  mg.params.F_0 = *mg.params.F_0 + mg_shift;
  fe.params.F_0 = *fe.params.F_0 + fe_shift;
  fe.params.name = "FeO";
  fe.params.formula = types::FormulaMap{{"Fe", 1.0}, {"O", 1.0}};
  fe.params.molar_mass = 0.0718444;
  Solution solution;
  solution.set_name(name);
  solution.set_solution_model(std::make_shared<solution_models::IdealSolution>(
      types::PairedEndmemberList{{mg, "[Mg]O"}, {fe, "[Fe]O"}}));
  solution.set_composition(Eigen::Array2d(0.8, 0.2));
  return solution;
}

struct PartitioningAssemblageFixture {
  types::FormulaMap bulk{{"Mg", 0.5}, {"Fe", 0.5}, {"O", 1.0}};
  Assemblage assemblage;

  PartitioningAssemblageFixture() {
    auto a = make_partitioning_oxide("Mg-rich", 0.0, 10000.0);
    auto b = make_partitioning_oxide("Fe-rich", 10000.0, 0.0);
    b.set_composition(Eigen::Array2d(0.2, 0.8));
    assemblage.add_phases(a, b);
    assemblage.set_fractions({0.4, 0.6});
    assemblage.set_method(types::EOSType::Auto);
    assemblage.set_n_moles(1.0);
    assemblage.set_state(1.0e9, 2000.0);
  }

  static double equilibrium_mg_fraction(double temperature) {
    return 1.0 / (1.0 + std::exp(-10000.0 / (constants::physics::gas_constant *
                                             temperature)));
  }
};

#endif
