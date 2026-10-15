// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#include "burnman/tools/phase_diagram/properties.hpp"
#include "burnman/core/relaxed_solution.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/polytope.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>

namespace burnman::phase_diagrams {
namespace {
using Getter = double (Material::*)() const;
const std::map<std::string, Getter> getters = {
#define PROPERTY(name) {#name, &Material::get_##name}
    PROPERTY(pressure),
    PROPERTY(temperature),
    PROPERTY(molar_mass),
    PROPERTY(molar_internal_energy),
    PROPERTY(molar_gibbs),
    PROPERTY(molar_helmholtz),
    PROPERTY(molar_volume),
    PROPERTY(density),
    PROPERTY(molar_entropy),
    PROPERTY(molar_enthalpy),
    PROPERTY(isothermal_bulk_modulus_reuss),
    PROPERTY(isentropic_bulk_modulus_reuss),
    PROPERTY(isothermal_compressibility_reuss),
    PROPERTY(isentropic_compressibility_reuss),
    PROPERTY(shear_modulus),
    PROPERTY(p_wave_velocity),
    PROPERTY(bulk_sound_velocity),
    PROPERTY(shear_wave_velocity),
    PROPERTY(grueneisen_parameter),
    PROPERTY(thermal_expansivity),
    PROPERTY(molar_heat_capacity_v),
    PROPERTY(molar_heat_capacity_p),
    PROPERTY(isentropic_thermal_gradient)
#undef PROPERTY
};
void validate_properties(const std::vector<std::string> &names) {
  if (names.empty())
    throw std::invalid_argument("Supply at least one thermodynamic property.");
  for (const auto &name : names)
    if (!getters.count(name))
      throw std::invalid_argument("Unknown scalar thermodynamic property: " +
                                  name);
}

struct ReactionCurvature {
  Eigen::MatrixXd basis, scaled;
  Eigen::VectorXd scale, eigenvalues;
  Eigen::MatrixXd eigenvectors;
  explicit ReactionCurvature(const Assemblage &a)
      : basis(a.get_reaction_basis()) {
    if (!basis.rows())
      return;
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(basis.cols(), basis.cols());
    const auto counts = a.get_endmembers_per_phase();
    const auto fractions = a.get_molar_fractions();
    Eigen::Index offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const Eigen::Index n = counts[i];
      if (n > 1) {
        const double f = fractions[static_cast<Eigen::Index>(i)];
        if (f <= 0.)
          throw std::runtime_error("Zero-amount solution in reaction Hessian.");
        // Solution's Hessian differentiates fractions; the extensive Hessian
        // differentiates species amounts. Use a one-current-mole reference.
        h.block(offset, offset, n, n) =
            a.get_phase<Solution>(i)->get_gibbs_hessian() / f;
      }
      offset += n;
    }
    scaled = basis * h * basis.transpose();
    if (!scaled.allFinite())
      throw std::runtime_error("Non-finite reaction Hessian.");
    scale =
        scaled.diagonal().cwiseAbs().cwiseMax(1.).cwiseSqrt().cwiseInverse();
    scaled = scale.asDiagonal() * scaled * scale.asDiagonal();
    scaled = .5 * (scaled + scaled.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(scaled);
    if (eig.info() != Eigen::Success)
      throw std::runtime_error("Reaction Hessian eigensolve failed.");
    eigenvalues = eig.eigenvalues();
    eigenvectors = eig.eigenvectors();
    if (eigenvalues.minCoeff() <
        -1.e-9 * std::max(1., eigenvalues.cwiseAbs().maxCoeff()))
      throw std::runtime_error("Equilibrium has negative Gibbs curvature.");
  }
  Eigen::Matrix2d correction(const Assemblage &a) const {
    if (!basis.rows())
      return Eigen::Matrix2d::Zero();
    // Do not replace a divergent univariant/critical response by a finite
    // pseudoinverse. The paper's inverse requires positive reaction curvature.
    if (eigenvalues.minCoeff() <= 1.e-12 * std::max(1., eigenvalues.maxCoeff()))
      throw std::runtime_error(
          "Singular reaction Hessian: relaxed response is undefined.");
    Eigen::MatrixXd partial(basis.cols(), 2);
    const auto counts = a.get_endmembers_per_phase();
    Eigen::Index offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const Eigen::Index n = counts[i];
      if (auto s = a.get_phase<Solution>(i)) {
        partial.block(offset, 0, n, 1) = s->get_partial_volumes();
        partial.block(offset, 1, n, 1) = -s->get_partial_entropies();
      } else {
        partial(offset, 0) = a.get_phase(i)->get_molar_volume();
        partial(offset, 1) = -a.get_phase(i)->get_molar_entropy();
      }
      offset += n;
    }
    const Eigen::MatrixXd coupling =
        eigenvectors.transpose() * scale.asDiagonal() * basis * partial;
    return coupling.transpose() * eigenvalues.cwiseInverse().asDiagonal() *
           coupling;
  }
};

class Properties {
  const Assemblage &a;
  bool relaxed;
  std::optional<ReactionCurvature> curvature;
  std::optional<Eigen::Vector3d> response;
  const Eigen::Vector3d &thermal() {
    if (!response) {
      if (!curvature)
        curvature.emplace(a);
      const auto c = curvature->correction(a);
      const double v = a.get_molar_volume();
      response = Eigen::Vector3d(
          a.get_thermal_expansivity() - c(0, 1) / v,
          a.get_molar_heat_capacity_p() + a.get_temperature() * c(1, 1),
          a.get_isothermal_compressibility_reuss() + c(0, 0) / v);
    }
    return *response;
  }

public:
  Properties(const Assemblage &input, bool relax) : a(input), relaxed(relax) {}
  void check_stability() { curvature.emplace(a); }
  double value(const std::string &name) {
    double value;
    if (!relaxed && name == "isentropic_thermal_gradient")
      value = a.get_thermal_expansivity() * a.get_molar_volume() *
              a.get_temperature() / a.get_molar_heat_capacity_p();
    else if (!relaxed || name == "pressure" || name == "temperature" ||
             name == "molar_mass" || name == "molar_internal_energy" ||
             name == "molar_gibbs" || name == "molar_helmholtz" ||
             name == "molar_volume" || name == "density" ||
             name == "molar_entropy" || name == "molar_enthalpy" ||
             name == "shear_modulus" || name == "shear_wave_velocity")
      value = (a.*getters.at(name))();
    else {
      const auto &r = thermal();
      const double alpha = r[0], cp = r[1], kt = 1. / r[2];
      const double v = a.get_molar_volume(), t = a.get_temperature();
      const double cv = cp - t * v * alpha * alpha * kt;
      const double ks = kt * cp / cv;
      if (name == "thermal_expansivity")
        value = alpha;
      else if (name == "molar_heat_capacity_p")
        value = cp;
      else if (name == "molar_heat_capacity_v")
        value = cv;
      else if (name == "isothermal_compressibility_reuss")
        value = r[2];
      else if (name == "isothermal_bulk_modulus_reuss")
        value = kt;
      else if (name == "isentropic_bulk_modulus_reuss")
        value = ks;
      else if (name == "isentropic_compressibility_reuss")
        value = 1. / ks;
      else if (name == "grueneisen_parameter")
        value = alpha * kt * v / cv;
      else if (name == "isentropic_thermal_gradient")
        value = alpha * v * t / cp;
      else if (name == "bulk_sound_velocity")
        value = std::sqrt(ks / a.get_density());
      else
        value =
            std::sqrt((ks + 4. / 3. * a.get_shear_modulus()) / a.get_density());
    }
    if (!std::isfinite(value))
      throw std::runtime_error("Non-finite property: " + name);
    return value;
  }
};

bool contains(const FieldPolygon &field, const Eigen::Vector2d &q,
              const Eigen::Vector2d &origin, const Eigen::Vector2d &range) {
  const Eigen::Vector2d p = (q - origin).cwiseQuotient(range);
  auto location = [&](const Eigen::MatrixXd &ring) {
    return detail::ring_location(
        static_cast<std::size_t>(ring.rows()),
        [&](std::size_t i) -> Eigen::Vector2d {
          return (ring.row(static_cast<Eigen::Index>(i)).transpose() - origin)
              .cwiseQuotient(range);
        },
        p, 1.e-9);
  };
  return location(field.vertices) >= 0 &&
         std::none_of(field.holes.begin(), field.holes.end(),
                      [&](const auto &hole) { return location(hole) >= 0; });
}

void validate_candidates(
    const Result &source,
    const std::vector<std::shared_ptr<Material>> &candidates) {
  const int stride = source.settings.max_phase_instances;
  if (source.composition_start.empty() || stride < 1)
    throw std::invalid_argument("Grid requires saved composition_start and a "
                                "positive phase-ID stride.");
  if (source.phase_names.size() !=
      candidates.size() * static_cast<std::size_t>(stride))
    throw std::invalid_argument(
        "Grid candidate names/order must match the saved phase_diagram.");
  for (std::size_t i = 0; i < candidates.size(); ++i)
    if (!candidates[i] ||
        candidates[i]->get_name() !=
            source.phase_names[i * static_cast<std::size_t>(stride)])
      throw std::invalid_argument(
          "Grid candidate names/order must match the saved phase_diagram.");
}

GridSolution capture_solution(const Assemblage &a, const detail::Engine &engine,
                              const std::vector<int> &ids) {
  GridSolution solution;
  solution.n_moles = a.get_n_moles();
  solution.molar_fractions = a.get_molar_fractions();
  engine.record_coordinates(solution.state, a);
  solution.state.phases = engine.snapshot(a, ids);
  solution.state.mass_balance_error = engine.mass_error(a);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto phase = a.get_phase(i);
    solution.phase_bases.push_back(engine.composition_basis(
        *phase, solution.state.phases[i].candidate_index));
    if (auto s = std::dynamic_pointer_cast<Solution>(phase))
      solution.phase_compositions.push_back(s->get_molar_fractions().matrix());
    else
      solution.phase_compositions.push_back(Eigen::VectorXd::Ones(1));
  }
  return solution;
}

std::shared_ptr<Assemblage> restore_solution(const GridSolution &solution,
                                             detail::Engine &engine) {
  const auto &state = solution.state;
  if (!std::isfinite(state.pressure) || !std::isfinite(state.temperature) ||
      state.pressure < 0. || state.temperature < 0. || state.phases.empty() ||
      state.phases.size() != solution.phase_bases.size() ||
      state.phases.size() != solution.phase_compositions.size() ||
      solution.molar_fractions.size() !=
          static_cast<Eigen::Index>(state.phases.size()) ||
      !solution.molar_fractions.allFinite() ||
      solution.molar_fractions.minCoeff() < -1.e-12 ||
      std::abs(solution.molar_fractions.sum() - 1.) > 1.e-8 ||
      !std::isfinite(solution.n_moles) || solution.n_moles <= 0.)
    throw std::invalid_argument("Invalid saved grid solution.");
  auto a = std::make_shared<Assemblage>();
  Eigen::ArrayXd amounts(static_cast<Eigen::Index>(state.phases.size()));
  for (std::size_t i = 0; i < state.phases.size(); ++i) {
    const auto &p = state.phases[i];
    const auto &basis = solution.phase_bases[i];
    const auto &composition = solution.phase_compositions[i];
    if (p.id < 0 ||
        p.candidate_index != p.id / engine.settings.max_phase_instances ||
        static_cast<std::size_t>(p.candidate_index) >= engine.phases.size() ||
        !std::isfinite(p.amount) || p.amount < -1.e-12 * solution.n_moles)
      throw std::invalid_argument(
          "Invalid saved grid phase identity or amount.");
    auto phase = detail::clone(
        engine.phases[static_cast<std::size_t>(p.candidate_index)].material);
    const auto n = engine.phases[static_cast<std::size_t>(p.candidate_index)]
                       .vertices.cols();
    if (basis.cols() != n || basis.rows() < 1 || basis.rows() > n ||
        composition.size() != basis.rows() || p.composition.size() != n ||
        !basis.allFinite() || !composition.allFinite() ||
        !p.composition.allFinite() ||
        std::abs(composition.sum() - 1.) > 1.e-8 ||
        (basis.transpose() * composition - p.composition).norm() > 1.e-8)
      throw std::invalid_argument(
          "Invalid saved grid phase basis or composition.");
    if (auto s = std::dynamic_pointer_cast<Solution>(phase)) {
      if (basis.rows() == n && basis.isIdentity(1.e-14))
        s->set_composition(composition.array());
      else {
        if (basis.fullPivLu().rank() != basis.rows())
          throw std::invalid_argument(
              "Saved grid phase basis is rank deficient.");
        phase = polytope::transform_solution_to_new_basis(
            *s, basis, composition.array(), p.name);
      }
    } else if (basis.rows() != 1 || !basis.isIdentity(1.e-14))
      throw std::invalid_argument("Invalid saved pure-phase basis.");
    phase->set_name(p.name);
    a->add_phases({phase});
    amounts[static_cast<Eigen::Index>(i)] = p.amount;
  }
  if (!std::isfinite(amounts.sum()) || amounts.sum() <= 0.)
    throw std::invalid_argument(
        "Saved grid solution has no positive phase amount.");
  if ((amounts - solution.molar_fractions * solution.n_moles).matrix().norm() >
      1.e-10 * solution.n_moles)
    throw std::invalid_argument("Saved grid phase amounts disagree with their "
                                "fractions and total moles.");
  a->set_fractions(solution.molar_fractions);
  a->set_n_moles(solution.n_moles);
  a->set_averaging_scheme(types::AveragingType::VRH);
  a->set_state(state.pressure, state.temperature);
  return a;
}
} // namespace

std::vector<std::string> thermodynamic_property_names() {
  std::vector<std::string> names;
  for (const auto &entry : getters)
    names.push_back(entry.first);
  return names;
}
std::map<std::string, double>
assemblage_properties(const Assemblage &a,
                      const std::vector<std::string> &names, bool relaxed) {
  validate_properties(names);
  for (Eigen::Index i = 0; i < a.get_n_phases(); ++i)
    if (a.get_phase<RelaxedSolution>(static_cast<std::size_t>(i)))
      throw std::invalid_argument(
          "assemblage_properties requires ordinary Solutions with explicit "
          "equilibrated populations; phase_diagram_grid unwraps "
          "RelaxedSolution "
          "candidates automatically.");
  Properties properties(a, relaxed);
  std::map<std::string, double> values;
  for (const auto &name : names)
    values[name] = properties.value(name);
  return values;
}

PropertyGrid
phase_diagram_grid(const Result &source,
                   const std::vector<std::shared_ptr<Material>> &candidates,
                   const std::vector<std::string> &names,
                   const Eigen::VectorXd &x, const Eigen::VectorXd &y,
                   bool relaxed, bool store_solutions) {
  validate_properties(names);
  const auto bounds = source.coordinate_ranges();
  const Eigen::Vector2d origin(bounds[0][0], bounds[1][0]);
  const Eigen::Vector2d range(bounds[0][1] - bounds[0][0],
                              bounds[1][1] - bounds[1][0]);
  if (!origin.allFinite() || !range.allFinite() || range.minCoeff() <= 0.)
    throw std::invalid_argument(
        "Grid diagram ranges must be finite and increasing.");
  for (int axis = 0; axis < 2; ++axis) {
    const auto &coordinates = axis == 0 ? x : y;
    if (!coordinates.size() || !coordinates.allFinite() ||
        coordinates.minCoeff() < bounds[static_cast<std::size_t>(axis)][0] ||
        coordinates.maxCoeff() > bounds[static_cast<std::size_t>(axis)][1])
      throw std::invalid_argument(
          "Grid coordinates must be finite, nonempty and inside the diagram.");
    for (Eigen::Index i = 1; i < coordinates.size(); ++i)
      if (coordinates[i] <= coordinates[i - 1])
        throw std::invalid_argument(
            "Grid coordinates must be strictly increasing.");
  }
  const int stride = source.settings.max_phase_instances;
  validate_candidates(source, candidates);

  detail::Engine engine(source.composition_start, candidates, source.settings,
                        source.section);
  engine.pressure_bounds = source.pressure_range;
  engine.temperature_bounds = source.temperature_range;
  PropertyGrid result;
  result.diagram_type = source.section.type;
  result.relaxed = relaxed;
  result.x = x;
  result.y = y;
  result.context.pressure_range = source.pressure_range;
  result.context.temperature_range = source.temperature_range;
  result.context.section = source.section;
  result.context.composition_start = source.composition_start;
  result.context.settings = source.settings;
  result.context.phase_names = source.phase_names;
  if (store_solutions)
    result.solutions.resize(static_cast<std::size_t>(x.size() * y.size()));
  result.field_numbers = Eigen::MatrixXi::Zero(y.size(), x.size());
  result.success = result.field_numbers.cast<bool>().array();
  result.equilibrium_success = result.success;
  for (const auto &name : names)
    result.values[name] = Eigen::MatrixXd::Constant(
        y.size(), x.size(), std::numeric_limits<double>::quiet_NaN());
  auto issue = [&](const std::string &message, bool equilibrium = true) {
    result.resolved = false;
    result.diagnostics.push_back(message);
    if (equilibrium)
      result.equilibrium_diagnostics.push_back(message);
  };
  const auto geometry = field_polygons(source);
  for (const auto &message : geometry.diagnostics)
    issue(message);
  for (const auto &field : geometry.polygons) {
    if (field.outside_model_domain)
      continue;
    const auto prefix = "Field " + std::to_string(field.field_number) + ": ";
    if (field.phases.empty() || field.has_open_boundary ||
        field.is_provisional) {
      issue(prefix + "unidentified, open or provisional field; grid skipped");
      continue;
    }
    const detail::PhaseStates *saved = nullptr;
    Eigen::Vector2d physical;
    Eigen::Vector2d seed_target = field.label_position;
    double best = std::numeric_limits<double>::infinity();
    auto consider = [&](const auto &state) {
      for (int id : field.phases)
        if (std::none_of(state.phases.begin(), state.phases.end(),
                         [&](const auto &p) { return p.id == id; }))
          return;
      const double distance =
          (diagram_coordinates(state, source.section.type) - seed_target)
              .cwiseQuotient(range)
              .norm();
      if (distance < best) {
        saved = &state.phases;
        physical << state.pressure, state.temperature;
        best = distance;
      }
    };
    if (field.sample_index >= 0)
      consider(source.samples.at(static_cast<std::size_t>(field.sample_index)));
    if (!saved)
      for (const auto &state : source.samples)
        if (state.success && !state.outside_model_domain)
          consider(state);
    if (!saved)
      for (const auto &line : source.boundaries)
        for (const auto &state : line.points)
          consider(state);
    if (!saved) {
      issue(prefix + "no saved equilibrium contains the field assemblage");
      continue;
    }
    for (const auto &p : *saved)
      if (std::find(field.phases.begin(), field.phases.end(), p.id) !=
          field.phases.end()) {
        if (p.id < 0 || p.candidate_index != p.id / stride ||
            static_cast<std::size_t>(p.candidate_index) >=
                engine.phases.size() ||
            p.composition.size() !=
                engine.phases[static_cast<std::size_t>(p.candidate_index)]
                    .vertices.cols() ||
            !p.composition.allFinite() || !std::isfinite(p.amount))
          throw std::invalid_argument("Grid requires valid saved phase "
                                      "compositions and candidate_index.");
      }
    detail::PhaseStates warm = *saved;
    engine.physical_seed = physical;
    for (Eigen::Index j = 0; j < y.size(); ++j)
      for (Eigen::Index k = 0; k < x.size(); ++k) {
        const Eigen::Index i = j % 2 == 0 ? k : x.size() - 1 - k;
        const Eigen::Vector2d q(x[i], y[j]);
        if (result.field_numbers(j, i) || !contains(field, q, origin, range))
          continue;
        result.field_numbers(j, i) = field.field_number;
        const auto point = prefix + "grid [" + std::to_string(j) + ", " +
                           std::to_string(i) + "]: ";
        try {
          if (!engine.free_vectors.empty())
            engine.set_bulk(q[1]);
          auto solve_at = [&](const detail::PhaseStates &initial) {
            auto a = engine.make_at(field.phases, initial, q);
            a->set_averaging_scheme(types::AveragingType::VRH);
            auto constraints = engine.state_constraints(q);
            auto solve = engine.solve(*a, constraints, false);
            if (!solve.success)
              throw std::runtime_error("Equilibrium solve failed: " +
                                       solve.message);
            Properties(*a, relaxed).check_stability();
            return a;
          };
          std::shared_ptr<Assemblage> a;
          try {
            a = solve_at(warm);
          } catch (const std::exception &) {
            // Retry from nearby accepted data when a grid step leaves the
            // Newton basin. Retain the same phases without a stability search.
            seed_target = q;
            best = std::numeric_limits<double>::infinity();
            saved = nullptr;
            for (const auto &state : source.samples)
              if (state.success && !state.outside_model_domain)
                consider(state);
            for (const auto &line : source.boundaries)
              for (const auto &state : line.points)
                consider(state);
            if (!saved)
              throw;
            engine.physical_seed = physical;
            a = solve_at(*saved);
          }
          Properties properties(*a, relaxed);
          properties.check_stability();
          warm = engine.snapshot(*a, field.phases);
          engine.physical_seed << a->get_pressure(), a->get_temperature();
          if (store_solutions)
            result.solutions[static_cast<std::size_t>(j * x.size() + i)] =
                capture_solution(*a, engine, field.phases);
          result.equilibrium_success(j, i) = true;
          result.success(j, i) = 1;
          for (const auto &name : names) {
            try {
              result.values.at(name)(j, i) = properties.value(name);
            } catch (const std::exception &e) {
              result.success(j, i) = 0;
              issue(point + name + ": " + e.what(), false);
            }
          }
        } catch (const std::exception &e) {
          issue(point + e.what());
        }
      }
  }
  result.equilibrium_solves = engine.equilibrium_solves;
  return result;
}

PropertyGrid
evaluate_property_grid(const PropertyGrid &saved,
                       const std::vector<std::shared_ptr<Material>> &candidates,
                       const std::vector<std::string> &names, bool relaxed) {
  validate_properties(names);
  if (saved.solutions.empty())
    throw std::invalid_argument(
        "Grid has no saved solutions; calculate with store_solutions=True.");
  if (saved.solutions.size() !=
      static_cast<std::size_t>(saved.x.size() * saved.y.size()))
    throw std::invalid_argument("Saved grid solutions do not match its shape.");
  validate_candidates(saved.context, candidates);
  detail::Engine engine(saved.context.composition_start, candidates,
                        saved.context.settings, saved.context.section);
  PropertyGrid result = saved;
  result.relaxed = relaxed;
  result.equilibrium_solves = 0;
  result.diagnostics = saved.equilibrium_diagnostics;
  result.resolved = result.diagnostics.empty();
  result.success.setConstant(false);
  result.values.clear();
  for (const auto &name : names)
    result.values[name] =
        Eigen::MatrixXd::Constant(saved.y.size(), saved.x.size(),
                                  std::numeric_limits<double>::quiet_NaN());
  auto issue = [&](const std::string &message) {
    result.diagnostics.push_back(message);
    result.resolved = false;
  };
  for (Eigen::Index j = 0; j < saved.y.size(); ++j)
    for (Eigen::Index i = 0; i < saved.x.size(); ++i) {
      const auto &solution =
          saved.solutions[static_cast<std::size_t>(j * saved.x.size() + i)];
      if (!solution)
        continue;
      const auto prefix = "Field " + std::to_string(saved.field_numbers(j, i)) +
                          ": grid [" + std::to_string(j) + ", " +
                          std::to_string(i) + "]: ";
      try {
        auto a = restore_solution(*solution, engine);
        if (!engine.free_vectors.empty())
          engine.set_bulk(solution->state.composition_coordinate);
        if (engine.mass_error(*a, false) >
            saved.context.settings.mass_balance_tolerance * 2.)
          throw std::invalid_argument(
              "Saved grid solution does not match its bulk composition.");
        Properties properties(*a, relaxed);
        properties.check_stability();
        result.success(j, i) = true;
        for (const auto &name : names) {
          try {
            result.values.at(name)(j, i) = properties.value(name);
          } catch (const std::exception &e) {
            result.success(j, i) = false;
            issue(prefix + name + ": " + e.what());
          }
        }
      } catch (const std::exception &e) {
        issue(prefix + e.what());
      }
    }
  return result;
}
} // namespace burnman::phase_diagrams
