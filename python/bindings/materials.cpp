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
#include "burnman/core/assemblage.hpp"
#include "burnman/core/mineral.hpp"
#include "burnman/core/relaxed_composite.hpp"
#include "burnman/minerals/SE_2015.hpp"
#include "burnman/minerals/other.hpp"
#include "burnman/minerals/water.hpp"
#include <pybind11/numpy.h>

namespace burnman::python {
using namespace pybind11::literals;
void bind_materials(py::module_ &m) {
  m.def("_liquid_iron", &minerals::other::liquid_iron);
  m.def("_ps1994_water", &minerals::Pitzer_Sterner_1994::H2O_Pitzer_Sterner);
  m.def("_se2015_bcc_iron", &minerals::SE_2015::bcc_iron);
  m.def("_se2015_fcc_iron", &minerals::SE_2015::fcc_iron);
  m.def("_se2015_hcp_iron", &minerals::SE_2015::hcp_iron);
  m.def("_se2015_liquid_iron", &minerals::SE_2015::liquid_iron);
  auto material = py::class_<Material, py::smart_holder>(m, "Material");
  material
      .def(
          "set_state",
          [](Material &self, double pressure, double temperature) {
            check_pt(pressure, temperature);
            self.set_state(pressure, temperature);
          },
          py::arg("pressure"), py::arg("temperature"))
      .def("set_method",
           py::overload_cast<types::EOSType>(&Material::set_method),
           py::arg("method"))
      .def("reset_cache", &Material::reset_cache)
      .def("reset", &Material::reset_cache)
      .def("copy", &Material::copy,
           "Return an independent Material class object with the same state "
           "and composition.")
      .def("clear_computed_properties", &Material::clear_computed_properties)
      .def("has_state", &Material::has_state)
      .def("set_name", &Material::set_name, py::arg("name"))
      .def("get_name", &Material::get_name)
      .def_property("name", &Material::get_name, &Material::set_name);
  material.def(
      "evaluate",
      [](const Material &self, const std::vector<std::string> &properties,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &pressures,
         const py::array_t<double, py::array::c_style | py::array::forcecast>
             &temperatures) {
        if (pressures.ndim() != temperatures.ndim() ||
            !std::equal(pressures.shape(), pressures.shape() + pressures.ndim(),
                        temperatures.shape()))
          throw std::invalid_argument(
              "pressures and temperatures must have the same shape.");
        const auto values =
            self.evaluate(properties,
                          Eigen::Map<const Eigen::ArrayXd>(pressures.data(),
                                                           pressures.size()),
                          Eigen::Map<const Eigen::ArrayXd>(
                              temperatures.data(), temperatures.size()));
        std::vector<py::ssize_t> shape{
            static_cast<py::ssize_t>(properties.size())};
        shape.insert(shape.end(), pressures.shape(),
                     pressures.shape() + pressures.ndim());
        py::array_t<double> result(shape);
        for (Eigen::Index i = 0; i < values.rows(); ++i)
          for (Eigen::Index j = 0; j < values.cols(); ++j)
            result.mutable_data()[i * values.cols() + j] = values(i, j);
        return result;
      },
      py::arg("vars_list"), py::arg("pressures"), py::arg("temperatures"),
      "Evaluate scalar properties at matching P,T arrays. Return shape "
      "(len(vars_list), *pressures.shape), preserving the material and its "
      "phases. Conventional property aliases are accepted.");
  auto formula = [](const Material &self) {
    if (const auto *mineral = dynamic_cast<const Mineral *>(&self)) {
      if (!mineral->params.formula)
        throw std::invalid_argument("Mineral parameters have no formula.");
    }
    return self.get_formula();
  };
  material.def_property_readonly("formula", formula)
      .def("get_formula", formula);
#define STATE_PROPERTY(name)                                                   \
  bind_property<Material>(material, #name, "get_" #name,                       \
                          &Material::get_##name, true)
  STATE_PROPERTY(pressure);
  STATE_PROPERTY(temperature);
  STATE_PROPERTY(molar_internal_energy);
  STATE_PROPERTY(molar_gibbs);
  STATE_PROPERTY(molar_helmholtz);
  STATE_PROPERTY(molar_volume);
  STATE_PROPERTY(density);
  STATE_PROPERTY(molar_entropy);
  STATE_PROPERTY(molar_enthalpy);
  STATE_PROPERTY(isothermal_bulk_modulus_reuss);
  STATE_PROPERTY(isentropic_bulk_modulus_reuss);
  STATE_PROPERTY(isothermal_compressibility_reuss);
  STATE_PROPERTY(isentropic_compressibility_reuss);
  STATE_PROPERTY(shear_modulus);
  STATE_PROPERTY(p_wave_velocity);
  STATE_PROPERTY(bulk_sound_velocity);
  STATE_PROPERTY(shear_wave_velocity);
  STATE_PROPERTY(grueneisen_parameter);
  STATE_PROPERTY(thermal_expansivity);
  STATE_PROPERTY(molar_heat_capacity_v);
  STATE_PROPERTY(molar_heat_capacity_p);
  STATE_PROPERTY(isentropic_thermal_gradient);
#undef STATE_PROPERTY
  bind_property<Material>(material, "molar_mass", "get_molar_mass",
                          &Material::get_molar_mass);

  py::class_<Mineral, Material, py::smart_holder>(m, "Mineral",
                                                  py::multiple_inheritance())
      .def(py::init([](const types::MineralParams &params) {
             if (!params.molar_mass || !std::isfinite(*params.molar_mass) ||
                 *params.molar_mass <= 0.0) {
               throw std::invalid_argument(
                   "Minerals require a positive finite molar_mass (kg/mol).");
             }
             auto mineral = std::make_shared<Mineral>();
             mineral->params = params;
             mineral->set_method(types::EOSType::Auto);
             return mineral;
           }),
           py::arg("params"))
      .def_property_readonly(
          "params", [](const Mineral &self) { return self.params; },
          "A copy of the validated parameters; construct a new mineral to "
          "change them.")
      .def("set_state_with_volume", &Mineral::set_state_with_volume,
           py::arg("volume"), py::arg("temperature"),
           py::arg("pressure_guesses") = Eigen::Vector2d(0., 1.e10),
           "Set V,T using EOS pressure or inversion of the modified molar "
           "volume.")
      .def_property(
          "uncertainties",
          [](const Mineral &self) {
            if (!self.uncertainties)
              throw py::attribute_error("No parameter uncertainties supplied.");
            return *self.uncertainties;
          },
          [](Mineral &self,
             const std::optional<std::unordered_map<std::string, double>>
                 &values) { self.uncertainties = values; },
          "Published parameter standard errors in the dataset's names and "
          "units. "
          "Returned dictionaries are independent copies; None clears the data.")
      .def(
          "set_property_modifiers",
          [](Mineral &self, const py::iterable &modifiers) {
            self.set_property_modifier_params(parse_modifiers(modifiers));
          },
          py::arg("modifiers"),
          "Set BurnMan (name, parameter-dictionary) property modifiers.")
      .def("get_property_modifiers",
           [](const Mineral &self) {
             check_state(self);
             const auto xs = self.get_property_modifiers();
             return py::dict("G"_a = xs.G, "dGdT"_a = xs.dGdT,
                             "dGdP"_a = xs.dGdP, "d2GdT2"_a = xs.d2GdT2,
                             "d2GdP2"_a = xs.d2GdP2, "d2GdPdT"_a = xs.d2GdPdT);
           })
      .def_property_readonly("molar_volume_unmodified",
                             [](const Mineral &self) {
                               check_state(self);
                               return self.get_molar_volume_unmodified();
                             });

  auto composite = py::class_<CompositeMaterial, Material, py::smart_holder>(
      m, "CompositeMaterial", py::multiple_inheritance());
#define COMPOSITE_PROPERTY(name)                                               \
  bind_property<CompositeMaterial>(composite, #name, "get_" #name,             \
                                   &CompositeMaterial::get_##name)
  COMPOSITE_PROPERTY(n_endmembers);
  COMPOSITE_PROPERTY(n_elements);
  COMPOSITE_PROPERTY(n_reactions);
  COMPOSITE_PROPERTY(elements);
  COMPOSITE_PROPERTY(endmember_names);
  COMPOSITE_PROPERTY(endmember_formulae);
  COMPOSITE_PROPERTY(independent_element_indices);
  COMPOSITE_PROPERTY(dependent_element_indices);
  COMPOSITE_PROPERTY(stoichiometric_matrix);
  COMPOSITE_PROPERTY(reduced_stoichiometric_matrix);
  COMPOSITE_PROPERTY(compositional_basis);
  COMPOSITE_PROPERTY(compositional_null_basis);
  COMPOSITE_PROPERTY(reaction_basis);
#undef COMPOSITE_PROPERTY
  composite.def_property_readonly(
      "reaction_basis_as_strings",
      [](const CompositeMaterial &self) {
        return py::module_::import("burnman_cpp.tools.chemistry")
            .attr("reaction_matrix_as_strings")(self.get_reaction_basis(),
                                                self.get_endmember_names());
      },
      "Format reaction_basis using endmember_names, with reactants on the "
      "left and rational coefficients where possible.");
  bind_property<CompositeMaterial>(composite, "partial_gibbs",
                                   "get_partial_gibbs",
                                   &CompositeMaterial::get_partial_gibbs, true);

  auto assemblage = py::class_<Assemblage, CompositeMaterial, py::smart_holder>(
      m, "Assemblage");
  bind_optional_attribute(assemblage, "state_priors",
                          &Assemblage::state_priors);
  bind_optional_attribute(assemblage, "state_covariances",
                          &Assemblage::state_covariances);
  bind_optional_attribute(assemblage, "state_inverse_covariances",
                          &Assemblage::state_inverse_covariances);
  bind_optional_attribute(assemblage, "reaction_matrix_for_optimization",
                          &Assemblage::reaction_matrix_for_optimization);
  assemblage
      .def(py::init([](const std::vector<std::shared_ptr<Material>> &phases,
                       const Eigen::ArrayXd &fractions,
                       types::FractionType fraction_type) {
             if (phases.empty())
               throw std::invalid_argument(
                   "An assemblage requires at least one phase.");
             for (const auto &phase : phases) {
               if (!phase)
                 throw std::invalid_argument(
                     "Assemblage phases cannot be None.");
               if (const auto *mineral =
                       dynamic_cast<const Mineral *>(phase.get())) {
                 if (!mineral->params.formula)
                   throw std::invalid_argument(
                       "Assemblage minerals require formula dictionaries.");
               }
               phase->get_formula();
             }
             check_fractions(fractions,
                             static_cast<Eigen::Index>(phases.size()), false);
             auto self = std::make_shared<Assemblage>();
             self->add_phases(phases);
             self->set_fractions(fractions, fraction_type);
             self->set_averaging_scheme(types::AveragingType::VRH);
             self->set_n_moles(1.0);
             return self;
           }),
           py::arg("phases"), py::arg("fractions"),
           py::arg("fraction_type") = types::FractionType::Molar)
      .def(
          "set_fractions",
          [](Assemblage &self, const Eigen::ArrayXd &fractions,
             types::FractionType type) {
            check_fractions(fractions, self.get_n_phases(), false);
            self.set_fractions(fractions, type);
          },
          py::arg("fractions"),
          py::arg("fraction_type") = types::FractionType::Molar)
      .def(
          "set_averaging_scheme",
          [](Assemblage &self, types::AveragingType scheme) {
            self.set_averaging_scheme(scheme);
            self.reset_cache();
          },
          py::arg("scheme"))
      .def(
          "get_phase",
          [](const Assemblage &self, py::ssize_t index) {
            if (index < 0)
              index += self.get_n_phases();
            if (index < 0 || index >= self.get_n_phases())
              throw py::index_error("Phase index out of range.");
            return self.get_phase(static_cast<std::size_t>(index));
          },
          py::arg("index"))
      .def_property_readonly(
          "phases",
          [](const Assemblage &self) {
            std::vector<std::shared_ptr<Material>> phases;
            for (Eigen::Index i = 0; i < self.get_n_phases(); ++i)
              phases.push_back(self.get_phase(static_cast<std::size_t>(i)));
            return phases;
          })
      .def_property("n_moles", &Assemblage::get_n_moles,
                    [](Assemblage &self, double value) {
                      if (!std::isfinite(value) || value <= 0.0)
                        throw std::invalid_argument(
                            "n_moles must be positive and finite.");
                      self.set_n_moles(value);
                    });
  bind_property<Assemblage>(assemblage, "n_phases", "get_n_phases",
                            &Assemblage::get_n_phases);
  bind_property<Assemblage>(assemblage, "molar_fractions",
                            "get_molar_fractions",
                            &Assemblage::get_molar_fractions);
  bind_property<Assemblage>(assemblage, "volume_fractions",
                            "get_volume_fractions",
                            &Assemblage::get_volume_fractions, true);
  bind_property<Assemblage>(assemblage, "endmembers_per_phase",
                            "get_endmembers_per_phase",
                            &Assemblage::get_endmembers_per_phase);
  bind_property<Assemblage>(assemblage, "reaction_affinities",
                            "get_reaction_affinities",
                            &Assemblage::get_reaction_affinities, true);
  assemblage
      .def("set_components", &Assemblage::set_components, py::arg("components"),
           "Define component formulae as linear sums of the assemblage "
           "endmembers.")
      .def(
          "chemical_potential",
          [](Assemblage &self,
             const std::optional<std::vector<types::FormulaMap>> &components) {
            return self.chemical_potential(
                components.value_or(std::vector<types::FormulaMap>{}));
          },
          py::arg("components") = py::none(),
          "Return component chemical potentials in J/mol. The assemblage must "
          "be equilibrated. Components outside the span of its endmember "
          "formulae have undefined chemical potentials and raise an error.");
  assemblage
      .def_property(
          "equilibrium_tolerance", &Assemblage::get_equilibrium_tolerance,
          [](Assemblage &self, double value) {
            if (!std::isfinite(value) || value < 0.)
              throw std::invalid_argument(
                  "Equilibrium tolerance must be finite and nonnegative.");
            self.set_equilibrium_tolerance(value);
          })
      .def_property_readonly("equilibrated", [](const Assemblage &self) {
        check_state(self);
        const auto affinities = self.get_reaction_affinities();
        return affinities.allFinite() &&
               (affinities.array().abs() <= self.get_equilibrium_tolerance())
                   .all();
      });
  bind_property<Assemblage>(assemblage, "endmember_partial_gibbs",
                            "get_endmember_partial_gibbs",
                            &Assemblage::get_partial_gibbs, true);
  bind_property<Assemblage>(assemblage, "endmember_partial_volumes",
                            "get_endmember_partial_volumes",
                            &Assemblage::get_endmember_partial_volumes, true);
  bind_property<Assemblage>(assemblage, "endmember_partial_entropies",
                            "get_endmember_partial_entropies",
                            &Assemblage::get_endmember_partial_entropies, true);

  auto relaxed = py::class_<RelaxedComposite, Assemblage, py::smart_holder>(
      m, "RelaxedComposite");
  relaxed
      .def(py::init<std::shared_ptr<Assemblage>, const Eigen::MatrixXd &>(),
           py::arg("composite"), py::arg("relaxation_vectors"))
      .def(py::init([](const RelaxedComposite &other) {
             return std::dynamic_pointer_cast<RelaxedComposite>(other.copy());
           }),
           py::arg("other"))
      .def(
          "set_state",
          [](RelaxedComposite &self, double p, double t,
             const std::optional<types::FormulaMap> &bulk) {
            check_pt(p, t);
            if (bulk)
              self.set_state(p, t, *bulk);
            else
              self.set_state(p, t);
          },
          py::arg("pressure"), py::arg("temperature"),
          py::arg("bulk_composition") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def("copy_state_from_unrelaxed",
           &RelaxedComposite::copy_state_from_unrelaxed)
      .def("_clear_state", &RelaxedComposite::clear_state)
      .def_property_readonly("unrelaxed", &RelaxedComposite::get_unrelaxed)
      .def_property_readonly("n_relaxation_vectors",
                             &RelaxedComposite::get_n_relaxation_vectors)
      .def_property_readonly("dndq", &RelaxedComposite::get_dndq,
                             py::return_value_policy::copy)
      .def_property_readonly("dqdz_relaxed",
                             &RelaxedComposite::get_dqdz_relaxed,
                             py::return_value_policy::copy)
      .def_property_readonly("_d2Gdqdq_fixed_PT",
                             &RelaxedComposite::get_structural_hessian,
                             py::return_value_policy::copy)
      .def_property_readonly("_d2Gdqdq_fixed_PT_pinv",
                             &RelaxedComposite::get_susceptibility,
                             py::return_value_policy::copy)
      .def_property_readonly("_d2Gdqdz", &RelaxedComposite::get_coupling,
                             py::return_value_policy::copy)
      .def_property_readonly("_d2Gdzdz_Q",
                             &RelaxedComposite::get_frozen_hessian,
                             py::return_value_policy::copy)
      .def_property_readonly("_d2Gdzdz", &RelaxedComposite::get_relaxed_hessian,
                             py::return_value_policy::copy);
}
} // namespace burnman::python
