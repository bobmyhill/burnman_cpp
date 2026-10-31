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

#include "bindings.hpp"
#include "burnman/core/anisotropic_mineral.hpp"
#include "burnman/core/relaxed_solution.hpp"
#include "burnman/core/solution.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
namespace {
using Interactions = std::vector<std::vector<double>>;

void check_endmembers(const types::PairedEndmemberList &endmembers,
                      bool require_sites) {
  if (endmembers.empty())
    throw std::invalid_argument(
        "A solution model requires at least one endmember.");
  for (const auto &[mineral, formula] : endmembers) {
    if (!mineral.params.formula || mineral.params.formula->empty()) {
      throw std::invalid_argument(
          "Solution endmembers require chemical formula dictionaries.");
    }
    if (require_sites &&
        (formula.empty() || formula.find('[') == std::string::npos)) {
      throw std::invalid_argument("Endmember site formulae must include "
                                  "bracketed sites, e.g. '[Mg]O'.");
    }
  }
}

using Endmembers =
    std::vector<std::pair<std::shared_ptr<Mineral>, std::string>>;

types::PairedEndmemberList scalar_endmembers(const Endmembers &endmembers,
                                             bool require_sites = true) {
  types::PairedEndmemberList result;
  for (const auto &[mineral, formula] : endmembers) {
    if (!mineral)
      throw std::invalid_argument("Solution endmembers cannot be None.");
    result.emplace_back(*mineral, formula);
  }
  check_endmembers(result, require_sites);
  return result;
}

void check_model_fractions(const Eigen::ArrayXd &fractions,
                           const solution_models::SolutionModel &model) {
  if (dynamic_cast<const solution_models::MechanicalSolution *>(&model)) {
    if (fractions.size() != model.get_n_endmembers() ||
        !fractions.isFinite().all())
      throw std::invalid_argument(
          "Mechanical coefficients must be finite and match the endmembers.");
  } else {
    check_site_fractions(fractions, model.get_endmember_occupancies());
  }
}

template <typename Model>
std::shared_ptr<Model> retain_tensor_endmembers(std::shared_ptr<Model> model,
                                                const Endmembers &endmembers) {
  for (const auto &[mineral, formula] : endmembers)
    if (std::dynamic_pointer_cast<AnisotropicMineral>(mineral)) {
      for (const auto &entry : endmembers)
        model->polymorphic_endmembers.push_back(entry.first);
      break;
    }
  return model;
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

py::array_t<double> interaction_tensor(const types::NDArray<double> &values) {
  const auto n = static_cast<py::ssize_t>(values.shape()[0]);
  py::array_t<double> result({n, n, n});
  std::copy(values.data().begin(), values.data().end(), result.mutable_data());
  return result;
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
  MODEL_PROPERTY(n_occupancies);
  MODEL_PROPERTY(site_names);
  MODEL_PROPERTY(endmember_occupancies);
  MODEL_PROPERTY(endmember_n_occupancies);
  MODEL_PROPERTY(site_multiplicities);
  MODEL_PROPERTY(formulas);
  MODEL_PROPERTY(sites);
  MODEL_PROPERTY(empty_formula);
  MODEL_PROPERTY(general_formula);
#undef MODEL_PROPERTY
  model_class.def("clone", &SolutionModel::clone)
      .def_property_readonly(
          "endmembers",
          [](const SolutionModel &self) {
            py::list result;
            const auto &formulas = self.get_formulas();
            for (std::size_t i = 0; i < self.endmembers.size(); ++i) {
              auto mineral = self.polymorphic_endmembers.empty()
                                 ? std::make_shared<Mineral>(self.endmembers[i])
                                 : self.polymorphic_endmembers[i];
              result.append(py::make_tuple(mineral, formulas[i]));
            }
            return result;
          })
      .def_property_readonly("site_formulae", &SolutionModel::get_formulas)
      .def_property_readonly("endmember_noccupancies",
                             [](const SolutionModel &self) {
                               return self.get_endmember_n_occupancies();
                             });
#define MODEL_COMPUTE(name)                                                    \
  model_class.def(                                                             \
      #name,                                                                   \
      [](const SolutionModel &self, double pressure, double temperature,       \
         const Eigen::ArrayXd &fractions) {                                    \
        check_pt(pressure, temperature);                                       \
        check_model_fractions(fractions, self);                                \
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
  MODEL_COMPUTE(compute_Cp_excess);
  MODEL_COMPUTE(compute_alphaV_excess);
  MODEL_COMPUTE(compute_VoverKT_excess);
#undef MODEL_COMPUTE
  py::class_<MechanicalSolution, SolutionModel,
             std::shared_ptr<MechanicalSolution>>(
      m, "MechanicalSolution",
      "Linear endmember mixing without configurational entropy or "
      "interactions. Site formulae are labels; signed and unnormalised "
      "coefficients are allowed, as in Python BurnMan.")
      .def(py::init([](const Endmembers &endmembers) {
             return retain_tensor_endmembers(
                 std::make_shared<MechanicalSolution>(
                     scalar_endmembers(endmembers, false)),
                 endmembers);
           }),
           py::arg("endmembers"));
  py::class_<IdealSolution, SolutionModel, std::shared_ptr<IdealSolution>>(
      m, "IdealSolution")
      .def(py::init([](const Endmembers &endmembers) {
             return retain_tensor_endmembers(
                 std::make_shared<IdealSolution>(scalar_endmembers(endmembers)),
                 endmembers);
           }),
           py::arg("endmembers"));
  py::class_<AsymmetricRegularSolution, IdealSolution,
             std::shared_ptr<AsymmetricRegularSolution>>(
      m, "AsymmetricRegularSolution")
      .def(py::init([](const Endmembers &endmembers,
                       const std::vector<double> &alphas,
                       const Interactions &energy, const Interactions &volume,
                       const Interactions &entropy) {
             const auto scalar = scalar_endmembers(endmembers);
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
             return retain_tensor_endmembers(
                 std::make_shared<AsymmetricRegularSolution>(
                     scalar, alphas, energy, volume, entropy),
                 endmembers);
           }),
           py::arg("endmembers"), py::arg("alphas"),
           py::arg("energy_interaction"),
           py::arg("volume_interaction") = Interactions{},
           py::arg("entropy_interaction") = Interactions{})
      .def_property_readonly("alphas", &AsymmetricRegularSolution::get_alphas)
      .def_property_readonly(
          "We", &AsymmetricRegularSolution::get_energy_interactions)
      .def_property_readonly(
          "Ws", &AsymmetricRegularSolution::get_entropy_interactions)
      .def_property_readonly(
          "Wv", &AsymmetricRegularSolution::get_volume_interactions);
  py::class_<SymmetricRegularSolution, AsymmetricRegularSolution,
             std::shared_ptr<SymmetricRegularSolution>>(
      m, "SymmetricRegularSolution")
      .def(
          py::init([](const Endmembers &endmembers, const Interactions &energy,
                      const Interactions &volume, const Interactions &entropy) {
            const auto scalar = scalar_endmembers(endmembers);
            check_interactions(energy, endmembers.size());
            check_interactions(volume, endmembers.size(), true);
            check_interactions(entropy, endmembers.size(), true);
            return retain_tensor_endmembers(
                std::make_shared<SymmetricRegularSolution>(scalar, energy,
                                                           volume, entropy),
                endmembers);
          }),
          py::arg("endmembers"), py::arg("energy_interaction"),
          py::arg("volume_interaction") = Interactions{},
          py::arg("entropy_interaction") = Interactions{});

  using Binary = SubregularSolution::Interactions;
  using Ternary = SubregularSolution::TernaryTerms;
  py::class_<SubregularSolution, IdealSolution,
             std::shared_ptr<SubregularSolution>>(
      m, "SubregularSolution",
      "Ideal site entropy plus subregular binary [W_ij, W_ji] pairs and "
      "ternary [i, j, k, value] energy, volume and entropy terms.")
      .def(py::init([](const Endmembers &endmembers, const Binary &energy,
                       const std::optional<Binary> &volume,
                       const std::optional<Binary> &entropy,
                       const std::optional<Ternary> &ternary_e,
                       const std::optional<Ternary> &ternary_v,
                       const std::optional<Ternary> &ternary_s) {
             return retain_tensor_endmembers(
                 std::make_shared<SubregularSolution>(
                     scalar_endmembers(endmembers), energy,
                     volume.value_or(Binary{}), entropy.value_or(Binary{}),
                     ternary_e.value_or(Ternary{}),
                     ternary_v.value_or(Ternary{}),
                     ternary_s.value_or(Ternary{})),
                 endmembers);
           }),
           py::arg("endmembers"), py::arg("energy_interaction"),
           py::arg("volume_interaction") = py::none(),
           py::arg("entropy_interaction") = py::none(),
           py::arg("energy_ternary_terms") = py::none(),
           py::arg("volume_ternary_terms") = py::none(),
           py::arg("entropy_ternary_terms") = py::none())
      .def_property_readonly("Wijke",
                             [](const SubregularSolution &self) {
                               return interaction_tensor(
                                   self.get_energy_interactions());
                             })
      .def_property_readonly("Wijks",
                             [](const SubregularSolution &self) {
                               return interaction_tensor(
                                   self.get_entropy_interactions());
                             })
      .def_property_readonly("Wijkv", [](const SubregularSolution &self) {
        return interaction_tensor(self.get_volume_interactions());
      });

  py::class_<PolynomialSolution, IdealSolution,
             std::shared_ptr<PolynomialSolution>>(m, "PolynomialSolution")
      .def(py::init([](const Endmembers &endmembers,
                       const std::optional<Interactions> &esv,
                       const std::vector<std::shared_ptr<Mineral>> &minerals,
                       const std::optional<Interactions> &interactions,
                       const std::optional<Eigen::MatrixXd> &transformation) {
             return retain_tensor_endmembers(
                 std::make_shared<PolynomialSolution>(
                     scalar_endmembers(endmembers),
                     esv.value_or(Interactions{}), minerals,
                     interactions.value_or(Interactions{}), transformation),
                 endmembers);
           }),
           py::arg("endmembers"), py::arg("ESV_interactions") = py::none(),
           py::arg("interaction_endmembers") =
               std::vector<std::shared_ptr<Mineral>>{},
           py::arg("endmember_coefficients_and_interactions") = py::none(),
           py::arg("transformation_matrix") = py::none())
      .def_property_readonly("interaction_endmembers",
                             &PolynomialSolution::get_interaction_endmembers)
      .def_property_readonly("dqdp", &PolynomialSolution::get_dqdp,
                             py::return_value_policy::copy)
      .def_property_readonly("ESV_interactions",
                             &PolynomialSolution::get_ESV_interactions)
      .def_property_readonly(
          "endmember_coefficients_and_interactions",
          &PolynomialSolution::get_endmember_coefficients_and_interactions);

  auto solution =
      py::class_<Solution, CompositeMaterial, py::smart_holder>(m, "Solution");
  bind_optional_attribute(solution, "compositional_covariances",
                          &Solution::compositional_covariances);
  bind_optional_attribute(solution, "free_compositional_vectors",
                          &Solution::free_compositional_vectors);
  bind_optional_attribute(solution, "baseline_composition",
                          &Solution::baseline_composition);
  solution
      .def(
          "evaluate_with_volumes",
          [](const Solution &self, const std::vector<std::string> &names,
             const py::array_t<double, py::array::c_style |
                                           py::array::forcecast> &volumes,
             const py::array_t<double,
                               py::array::c_style | py::array::forcecast>
                 &temperatures) {
            if (volumes.ndim() != temperatures.ndim() ||
                !std::equal(volumes.shape(), volumes.shape() + volumes.ndim(),
                            temperatures.shape()))
              throw std::invalid_argument(
                  "volumes and temperatures must have the same shape.");
            const auto values = self.evaluate_with_volumes(
                names,
                Eigen::Map<const Eigen::ArrayXd>(volumes.data(),
                                                 volumes.size()),
                Eigen::Map<const Eigen::ArrayXd>(temperatures.data(),
                                                 temperatures.size()));
            std::vector<py::ssize_t> shape{
                static_cast<py::ssize_t>(names.size())};
            shape.insert(shape.end(), volumes.shape(),
                         volumes.shape() + volumes.ndim());
            py::array_t<double> result(shape);
            for (Eigen::Index i = 0; i < values.rows(); ++i)
              for (Eigen::Index j = 0; j < values.cols(); ++j)
                result.mutable_data()[i * values.cols() + j] = values(i, j);
            return result;
          },
          py::arg("vars_list"), py::arg("volumes"), py::arg("temperatures"))
      .def("set_state_with_volume", &Solution::set_state_with_volume,
           py::arg("volume"), py::arg("temperature"),
           py::arg("pressure_guesses") = Eigen::Vector2d(0., 1.e10))
      .def(py::init([](const std::shared_ptr<SolutionModel> &model,
                       const Eigen::ArrayXd &fractions,
                       const std::string &name) {
             if (!model)
               throw std::invalid_argument("solution_model cannot be None.");
             check_model_fractions(fractions, *model);
             // Models hold mutable endmember state. Give each solution its own
             // copy.
             auto owned = model->clone();
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
            check_model_fractions(fractions, *self.get_solution_model());
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
  bind_property<Solution>(solution, "solution_model", "get_solution_model",
                          &Solution::get_solution_model);
  bind_property<Solution>(solution, "site_occupancies", "get_site_occupancies",
                          &Solution::get_site_occupancies);
  solution.def(
      "site_formula", &Solution::site_formula, py::arg("precision") = 2,
      py::arg("print_absent_species") = true,
      py::arg("site_occupancies") = py::none(),
      "Return the molar formula with fractional site occupancies. "
      "precision gives decimal places; print_absent_species includes "
      "zero occupancies. Optional site_occupancies replace the current "
      "occupancies in site_names order.");
  bind_property<Solution>(solution, "endmember_occupancies",
                          "get_endmember_occupancies",
                          &Solution::get_endmember_occupancies);
  bind_property<Solution>(solution, "endmember_n_occupancies",
                          "get_endmember_n_occupancies",
                          &Solution::get_endmember_n_occupancies);
  py::class_<RelaxedSolution, Solution, py::smart_holder>(m, "RelaxedSolution")
      .def(py::init<Solution, const Eigen::MatrixXd &,
                    const Eigen::MatrixXd &>(),
           py::arg("solution"), py::arg("relaxation_vectors"),
           py::arg("unrelaxed_vectors"))
      .def(
          "set_composition",
          [](RelaxedSolution &self, const Eigen::ArrayXd &fractions,
             const std::optional<Eigen::VectorXd> &q, bool relaxed) {
            self.set_composition(
                fractions,
                q.value_or(Eigen::VectorXd::Zero(self.get_dndq().cols())),
                relaxed);
          },
          py::arg("molar_fractions"), py::arg("q_initial") = py::none(),
          py::arg("relaxed") = true)
      .def(
          "set_state",
          [](RelaxedSolution &self, double p, double t, bool relaxed) {
            check_pt(p, t);
            self.set_state(p, t, relaxed);
          },
          py::arg("pressure"), py::arg("temperature"),
          py::arg("relaxed") = true)
      .def_property_readonly("dndq", &RelaxedSolution::get_dndq)
      .def_property_readonly("dndx", &RelaxedSolution::get_dndx)
      .def_property_readonly("unrelaxed_vectors",
                             &RelaxedSolution::get_unrelaxed_vectors);
}
} // namespace burnman::python
