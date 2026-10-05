#include "bindings.hpp"
#include "burnman/core/solution.hpp"

namespace burnman::python {
namespace {
using Interactions = std::vector<std::vector<double>>;

void check_endmembers(const types::PairedEndmemberList &endmembers) {
  if (endmembers.empty())
    throw std::invalid_argument(
        "A solution model requires at least one endmember.");
  for (const auto &[mineral, formula] : endmembers) {
    if (!mineral.params.formula || mineral.params.formula->empty()) {
      throw std::invalid_argument(
          "Solution endmembers require chemical formula dictionaries.");
    }
    if (formula.empty() || formula.find('[') == std::string::npos) {
      throw std::invalid_argument("Endmember site formulae must include "
                                  "bracketed sites, e.g. '[Mg]O'.");
    }
  }
}

void check_interactions(const Interactions &interactions, std::size_t size,
                        bool optional = false) {
  if (optional && interactions.empty())
    return;
  if (interactions.size() != size - 1) {
    throw std::invalid_argument(
        "Interaction parameters require n-1 rows of lengths n-1, n-2, ..., 1.");
  }
  for (std::size_t i = 0; i < interactions.size(); ++i) {
    if (interactions[i].size() != size - i - 1) {
      throw std::invalid_argument(
          "Interaction parameter row has the wrong length.");
    }
    for (double value : interactions[i]) {
      if (!std::isfinite(value))
        throw std::invalid_argument("Interaction parameters must be finite.");
    }
  }
}
} // namespace

void bind_solutions(py::module_ &m) {
  using namespace solution_models;
  auto model_class = py::class_<SolutionModel, std::shared_ptr<SolutionModel>>(
      m, "SolutionModel");
#define MODEL_PROPERTY(name)                                                   \
  model_class.def_property_readonly(                                           \
      #name, [](const SolutionModel &self) { return self.get_##name(); })
  MODEL_PROPERTY(n_endmembers);
  MODEL_PROPERTY(n_sites);
  MODEL_PROPERTY(site_names);
  MODEL_PROPERTY(endmember_occupancies);
  MODEL_PROPERTY(endmember_n_occupancies);
  MODEL_PROPERTY(site_multiplicities);
  MODEL_PROPERTY(formulas);
  MODEL_PROPERTY(sites);
#undef MODEL_PROPERTY
#define MODEL_COMPUTE(name)                                                    \
  model_class.def(                                                             \
      #name,                                                                   \
      [](const SolutionModel &self, double pressure, double temperature,       \
         const Eigen::ArrayXd &fractions) {                                    \
        check_pt(pressure, temperature);                                       \
        check_site_fractions(fractions, self.get_endmember_occupancies());     \
        return self.name(pressure, temperature, fractions);                    \
      },                                                                       \
      py::arg("pressure"), py::arg("temperature"), py::arg("molar_fractions"))
  MODEL_COMPUTE(compute_excess_gibbs_free_energy);
  MODEL_COMPUTE(compute_excess_volume);
  MODEL_COMPUTE(compute_excess_entropy);
  MODEL_COMPUTE(compute_excess_enthalpy);
  MODEL_COMPUTE(compute_excess_partial_gibbs_free_energies);
  MODEL_COMPUTE(compute_excess_partial_volumes);
  MODEL_COMPUTE(compute_excess_partial_entropies);
  MODEL_COMPUTE(compute_activities);
  MODEL_COMPUTE(compute_activity_coefficients);
  MODEL_COMPUTE(compute_gibbs_hessian);
  MODEL_COMPUTE(compute_entropy_hessian);
  MODEL_COMPUTE(compute_volume_hessian);
#undef MODEL_COMPUTE
  py::class_<IdealSolution, SolutionModel, std::shared_ptr<IdealSolution>>(
      m, "IdealSolution")
      .def(py::init([](const types::PairedEndmemberList &endmembers) {
             check_endmembers(endmembers);
             return std::make_shared<IdealSolution>(endmembers);
           }),
           py::arg("endmembers"));
  py::class_<AsymmetricRegularSolution, IdealSolution,
             std::shared_ptr<AsymmetricRegularSolution>>(
      m, "AsymmetricRegularSolution")
      .def(py::init([](const types::PairedEndmemberList &endmembers,
                       const std::vector<double> &alphas,
                       const Interactions &energy, const Interactions &volume,
                       const Interactions &entropy) {
             check_endmembers(endmembers);
             if (alphas.size() != endmembers.size())
               throw std::invalid_argument(
                   "alphas must match the number of endmembers.");
             for (double alpha : alphas) {
               if (!std::isfinite(alpha) || alpha <= 0.0)
                 throw std::invalid_argument(
                     "alphas must be positive and finite.");
             }
             check_interactions(energy, endmembers.size());
             check_interactions(volume, endmembers.size(), true);
             check_interactions(entropy, endmembers.size(), true);
             return std::make_shared<AsymmetricRegularSolution>(
                 endmembers, alphas, energy, volume, entropy);
           }),
           py::arg("endmembers"), py::arg("alphas"),
           py::arg("energy_interaction"),
           py::arg("volume_interaction") = Interactions{},
           py::arg("entropy_interaction") = Interactions{});
  py::class_<SymmetricRegularSolution, AsymmetricRegularSolution,
             std::shared_ptr<SymmetricRegularSolution>>(
      m, "SymmetricRegularSolution")
      .def(py::init([](const types::PairedEndmemberList &endmembers,
                       const Interactions &energy, const Interactions &volume,
                       const Interactions &entropy) {
             check_endmembers(endmembers);
             check_interactions(energy, endmembers.size());
             check_interactions(volume, endmembers.size(), true);
             check_interactions(entropy, endmembers.size(), true);
             return std::make_shared<SymmetricRegularSolution>(
                 endmembers, energy, volume, entropy);
           }),
           py::arg("endmembers"), py::arg("energy_interaction"),
           py::arg("volume_interaction") = Interactions{},
           py::arg("entropy_interaction") = Interactions{});

  auto solution =
      py::class_<Solution, CompositeMaterial, std::shared_ptr<Solution>>(
          m, "Solution");
  solution
      .def(py::init([](const std::shared_ptr<SolutionModel> &model,
                       const Eigen::ArrayXd &fractions,
                       const std::string &name) {
             if (!model)
               throw std::invalid_argument("solution_model cannot be None.");
             check_site_fractions(fractions,
                                  model->get_endmember_occupancies());
             // Models hold mutable endmember state. Give each solution its own
             // copy.
             std::shared_ptr<SolutionModel> owned;
             if (const auto *symmetric =
                     dynamic_cast<const SymmetricRegularSolution *>(
                         model.get())) {
               owned = std::make_shared<SymmetricRegularSolution>(*symmetric);
             } else if (const auto *asymmetric =
                            dynamic_cast<const AsymmetricRegularSolution *>(
                                model.get())) {
               owned = std::make_shared<AsymmetricRegularSolution>(*asymmetric);
             } else if (const auto *ideal =
                            dynamic_cast<const IdealSolution *>(model.get())) {
               owned = std::make_shared<IdealSolution>(*ideal);
             } else {
               throw std::invalid_argument("Unsupported solution model.");
             }
             auto self = std::make_shared<Solution>();
             self->set_solution_model(owned);
             // Incoming endmembers already have validated EOS objects. Combined
             // endmembers must retain their custom EOS when the model is
             // copied.
             self->set_composition(fractions);
             if (!name.empty())
               self->set_name(name);
             return self;
           }),
           py::arg("solution_model"), py::arg("molar_fractions"),
           py::arg("name") = "")
      .def(
          "set_composition",
          [](Solution &self, const Eigen::ArrayXd &fractions) {
            check_site_fractions(fractions, self.get_endmember_occupancies());
            self.set_composition(fractions);
          },
          py::arg("molar_fractions"));
#define SOLUTION_PROPERTY(name)                                                \
  bind_property<Solution>(solution, #name, "get_" #name,                       \
                          &Solution::get_##name, true)
  SOLUTION_PROPERTY(excess_gibbs);
  SOLUTION_PROPERTY(excess_volume);
  SOLUTION_PROPERTY(excess_entropy);
  SOLUTION_PROPERTY(excess_enthalpy);
  SOLUTION_PROPERTY(activities);
  SOLUTION_PROPERTY(activity_coefficients);
  SOLUTION_PROPERTY(excess_partial_gibbs);
  SOLUTION_PROPERTY(excess_partial_volumes);
  SOLUTION_PROPERTY(excess_partial_entropies);
  SOLUTION_PROPERTY(partial_volumes);
  SOLUTION_PROPERTY(partial_entropies);
  SOLUTION_PROPERTY(gibbs_hessian);
  SOLUTION_PROPERTY(entropy_hessian);
  SOLUTION_PROPERTY(volume_hessian);
#undef SOLUTION_PROPERTY
  bind_property<Solution>(solution, "molar_fractions", "get_molar_fractions",
                          &Solution::get_molar_fractions);
  bind_property<Solution>(solution, "basis", "get_basis", &Solution::get_basis);
  bind_property<Solution>(solution, "site_names", "get_site_names",
                          &Solution::get_site_names);
  bind_property<Solution>(solution, "endmember_occupancies",
                          "get_endmember_occupancies",
                          &Solution::get_endmember_occupancies);
  bind_property<Solution>(solution, "endmember_n_occupancies",
                          "get_endmember_n_occupancies",
                          &Solution::get_endmember_n_occupancies);
}
} // namespace burnman::python
