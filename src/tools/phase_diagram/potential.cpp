// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/index_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <nlopt.hpp>

namespace burnman::phase_diagrams::detail {
namespace {
// Work in the existing equilibrium parameterisation: P, T, phase amounts and
// independent endmember fractions. Fixed P/T parameters are eliminated. The
// entropy/volume Jacobians are shared with constrained equilibrium solves.
struct PotentialMinimum {
  Assemblage &a;
  EquilibrationParameters prm;
  Eigen::VectorXd fixed, scales, equation_scales, inequality_scales;
  std::vector<Eigen::Index> variables;
  ConstraintGroup constraints;
  bool fix_entropy = false, fix_volume = false;
  double energy_scale, offset = 0.;
  Eigen::VectorXd cached, gradient, equations, fallback;
  Eigen::MatrixXd jacobian;
  double value = 0., fallback_error = std::numeric_limits<double>::infinity();
  double fallback_value = std::numeric_limits<double>::infinity();
  std::size_t evaluations = 0;

  PotentialMinimum(Assemblage &input, const types::FormulaMap &bulk,
                   const std::vector<std::pair<Coordinate, double>> &targets)
      : a(input), prm(get_equilibration_parameters(a, bulk, {})),
        fixed(get_parameter_vector(a)),
        scales(Eigen::VectorXd::Ones(fixed.size())),
        energy_scale(1.e5 * a.get_n_moles()) {
    scales[0] = 1.e10;
    scales[1] = 1000.;
    for (auto k : prm.phase_amount_indices)
      scales[k] = a.get_n_moles();
    bool fix_pressure = false, fix_temperature = false;
    std::vector<double> normalisers;
    for (auto [axis, target] : targets) {
      if (axis == Coordinate::P) {
        fix_pressure = true;
        fixed[0] = target;
      } else if (axis == Coordinate::T) {
        fix_temperature = true;
        fixed[1] = target;
      } else if (axis == Coordinate::S) {
        fix_entropy = true;
        constraints.push_back(std::make_unique<EntropyConstraint>(target));
        normalisers.push_back(std::max(std::abs(target), a.get_n_moles()));
      } else if (axis == Coordinate::V) {
        fix_volume = true;
        constraints.push_back(std::make_unique<VolumeConstraint>(target));
        normalisers.push_back(
            std::max(std::abs(target), 1.e-12 * a.get_n_moles()));
      }
    }
    for (Eigen::Index k = 0; k < fixed.size(); ++k)
      if ((k != 0 || !fix_pressure) && (k != 1 || !fix_temperature))
        variables.push_back(k);
    const Eigen::Index count = static_cast<Eigen::Index>(constraints.size());
    equation_scales.resize(count + prm.reduced_composition_vector.size());
    for (Eigen::Index k = 0; k < count; ++k)
      equation_scales[k] = normalisers[static_cast<std::size_t>(k)];
    equation_scales.tail(prm.reduced_composition_vector.size()) =
        prm.reduced_composition_vector.cwiseAbs().cwiseMax(
            1.e-8 * prm.bulk_composition_vector.norm());
    inequality_scales = (prm.constraint_matrix * scales.asDiagonal())
                            .rowwise()
                            .norm()
                            .cwiseMax(1.e-30);
    offset = energy();
  }

  double energy() const {
    // G + TS = H; G - PV = F; G + TS - PV = U. Every term is extensive.
    double e = a.get_molar_gibbs();
    if (fix_entropy)
      e += a.get_temperature() * a.get_molar_entropy();
    if (fix_volume)
      e -= a.get_pressure() * a.get_molar_volume();
    return a.get_n_moles() * e;
  }

  Eigen::VectorXd parameters(const std::vector<double> &z) const {
    auto x = fixed;
    for (std::size_t k = 0; k < variables.size(); ++k)
      x[variables[k]] = z[k] * scales[variables[k]];
    return x;
  }

  void evaluate(const std::vector<double> &z) {
    Eigen::Map<const Eigen::VectorXd> mapped(
        z.data(), static_cast<Eigen::Index>(z.size()));
    if (cached.size() == mapped.size() &&
        (cached.array() == mapped.array()).all())
      return;
    ++evaluations;
    const auto x = parameters(z);
    set_composition_and_state_from_parameters(a, x);
    const Eigen::Index n = x.size(), ne = a.get_n_endmembers();
    const Eigen::Index nc = static_cast<Eigen::Index>(constraints.size());
    const Eigen::Index nb = prm.reduced_composition_vector.size();
    Eigen::MatrixXd dn = Eigen::MatrixXd::Zero(ne, n);
    Eigen::VectorXd dg = Eigen::VectorXd::Zero(n);
    const double amount = a.get_n_moles(), t = a.get_temperature(),
                 p = a.get_pressure();
    const double s = amount * a.get_molar_entropy(),
                 v = amount * a.get_molar_volume();
    dg[0] = v;
    dg[1] = -s;
    Eigen::Index species = 0, parameter = 2;
    const auto counts = a.get_endmembers_per_phase();
    for (std::size_t k = 0; k < counts.size(); ++k) {
      const Eigen::Index count = counts[k];
      dg[parameter] = a.get_phase(k)->get_molar_gibbs();
      if (count == 1)
        dn(species, parameter) = 1.;
      else {
        const auto solution = a.get_phase<Solution>(k);
        const double phase_amount = x[parameter];
        dn.block(species, parameter, count, 1) =
            solution->get_molar_fractions();
        dn.block(species + 1, parameter + 1, count - 1, count - 1)
            .diagonal()
            .setConstant(phase_amount);
        dn.block(species, parameter + 1, 1, count - 1)
            .setConstant(-phase_amount);
        const auto partial = solution->get_partial_gibbs();
        dg.segment(parameter + 1, count - 1) =
            phase_amount * (partial.tail(count - 1) - partial[0]).matrix();
      }
      species += count;
      parameter += count;
    }
    if (fix_entropy) {
      dg += t * EntropyConstraint(0.).derivative(x, a, n);
      dg[1] += s;
    }
    if (fix_volume) {
      dg -= p * VolumeConstraint(0.).derivative(x, a, n);
      dg[0] -= v;
    }
    equations.resize(nc + nb);
    Eigen::MatrixXd full_jacobian = Eigen::MatrixXd::Zero(nc + nb, n);
    for (Eigen::Index k = 0; k < nc; ++k) {
      const auto &constraint = constraints[static_cast<std::size_t>(k)];
      equations[k] = constraint->evaluate(x, a);
      full_jacobian.row(k) = constraint->derivative(x, a, n);
    }
    const auto &stoich = a.get_reduced_stoichiometric_matrix();
    equations.tail(nb) = stoich.transpose() * get_endmember_amounts(a) -
                         prm.reduced_composition_vector;
    full_jacobian.bottomRows(nb) = stoich.transpose() * dn;
    equations.array() /= equation_scales.array();
    full_jacobian = equation_scales.cwiseInverse().asDiagonal() * full_jacobian;
    value = (energy() - offset) / energy_scale;
    gradient.resize(mapped.size());
    jacobian.resize(nc + nb, mapped.size());
    for (std::size_t k = 0; k < variables.size(); ++k) {
      gradient[static_cast<Eigen::Index>(k)] =
          dg[variables[k]] * scales[variables[k]] / energy_scale;
      jacobian.col(static_cast<Eigen::Index>(k)) =
          full_jacobian.col(variables[k]) * scales[variables[k]];
    }
    if (!std::isfinite(value) || !gradient.allFinite() ||
        !equations.allFinite() || !jacobian.allFinite())
      throw std::runtime_error(
          "Non-finite constrained thermodynamic potential.");
    // Keep a valid evaluated state if SLSQP later leaves the EOS domain. Among
    // feasible states prefer lower energy; otherwise retain the closest state.
    const double error =
        std::max(equations.cwiseAbs().maxCoeff(),
                 ((prm.constraint_matrix * x + prm.constraint_vector).array() /
                  inequality_scales.array())
                     .maxCoeff());
    if ((error <= 1.e-8 &&
         (fallback_error > 1.e-8 || value < fallback_value)) ||
        (fallback_error > 1.e-8 && error < fallback_error)) {
      fallback = mapped;
      fallback_error = error;
      fallback_value = value;
    }
    cached = mapped;
  }
};

struct PotentialStart {
  std::vector<int> ids;
  Eigen::VectorXd x, tolerance;
  std::vector<Eigen::MatrixXd> bases;

  PotentialStart(const Engine &engine, const PotentialMinimum &objective,
                 const std::vector<int> &phases)
      : ids(phases), x(objective.fixed),
        tolerance(Eigen::VectorXd::Constant(
            x.size(),
            std::min(1.e-10, engine.settings.composition_tolerance * .01))) {
    tolerance.head<2>() = equilibrium_pt_tolerance();
    for (auto index : objective.prm.phase_amount_indices)
      tolerance[index] = std::max(1.e-30, 1.e-10 * objective.a.get_n_moles());
    for (std::size_t k = 0; k < ids.size(); ++k)
      bases.push_back(engine.composition_basis(
          *objective.a.get_phase(k),
          ids[k] / engine.settings.max_phase_instances));
  }

  bool same(const PotentialStart &other) const {
    if (ids != other.ids || x.size() != other.x.size() || !x.allFinite() ||
        !other.x.allFinite() ||
        ((x - other.x).cwiseAbs().array() >
         tolerance.cwiseMin(other.tolerance).array())
            .any())
      return false;
    for (std::size_t k = 0; k < bases.size(); ++k)
      if (bases[k].rows() != other.bases[k].rows() ||
          bases[k].cols() != other.bases[k].cols() ||
          (bases[k] - other.bases[k]).cwiseAbs().maxCoeff() > 1.e-12)
        return false;
    return true;
  }
};

bool same_retained_state(const Engine &engine, const WorkState &a,
                         const WorkState &b) {
  if (!a.assemblage || !b.assemblage || a.ids != b.ids ||
      !std::isfinite(a.state.pressure) || !std::isfinite(b.state.pressure) ||
      !std::isfinite(a.state.temperature) ||
      !std::isfinite(b.state.temperature) ||
      std::abs(a.state.pressure - b.state.pressure) >
          equilibrium_pt_tolerance()[0] ||
      std::abs(a.state.temperature - b.state.temperature) >
          equilibrium_pt_tolerance()[1] ||
      a.state.phases.size() != b.state.phases.size())
    return false;
  const double amount_tolerance =
      1.e-10 *
      std::max(total_amount(a.state.phases), total_amount(b.state.phases));
  const double composition_tolerance =
      std::min(1.e-10, engine.settings.composition_tolerance * .01);
  for (std::size_t k = 0; k < a.state.phases.size(); ++k) {
    const auto &left = a.state.phases[k], &right = b.state.phases[k];
    if (left.id != right.id ||
        left.composition.size() != right.composition.size() ||
        !left.composition.allFinite() || !right.composition.allFinite() ||
        !std::isfinite(left.amount) || !std::isfinite(right.amount) ||
        std::abs(left.amount - right.amount) > amount_tolerance ||
        (left.composition - right.composition).cwiseAbs().maxCoeff() >
            composition_tolerance)
      return false;
    const auto x = engine.composition_basis(*a.assemblage->get_phase(k),
                                            left.candidate_index),
               y = engine.composition_basis(*b.assemblage->get_phase(k),
                                            right.candidate_index);
    if (x.rows() != y.rows() || x.cols() != y.cols() || !x.allFinite() ||
        !y.allFinite() || (x - y).cwiseAbs().maxCoeff() > 1.e-12)
      return false;
  }
  return true;
}

void minimize_active_potential(PotentialMinimum &d, const Engine &engine) {
  const auto dimension = static_cast<unsigned int>(d.variables.size());
  nlopt::opt opt(nlopt::LD_SLSQP, dimension);
  opt.set_min_objective(
      [](const std::vector<double> &z, std::vector<double> &g, void *context) {
        auto &data = *static_cast<PotentialMinimum *>(context);
        data.evaluate(z);
        if (!g.empty())
          std::copy(data.gradient.data(),
                    data.gradient.data() + data.gradient.size(), g.begin());
        return data.value;
      },
      &d);
  opt.add_equality_mconstraint(
      [](unsigned int m, double *values, unsigned int n, const double *z,
         double *gradient, void *context) {
        auto &data = *static_cast<PotentialMinimum *>(context);
        data.evaluate(std::vector<double>(z, z + n));
        std::copy(data.equations.data(), data.equations.data() + m, values);
        if (gradient)
          Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>(gradient, m, n) =
              data.jacobian;
      },
      &d,
      std::vector<double>(static_cast<std::size_t>(d.equation_scales.size()),
                          1.e-10));
  struct Linear {
    Eigen::VectorXd a;
    double b;
  };
  std::vector<Linear> inequalities;
  for (Eigen::Index row = 0; row < d.prm.constraint_matrix.rows(); ++row) {
    // P/T and phase-amount lower bounds are supplied as box bounds below.
    auto full = d.prm.constraint_matrix.row(row);
    if ((full.array() != 0.).count() == 1 &&
        (full.head(2).norm() > 0. ||
         full(d.prm.phase_amount_indices).matrix().norm() > 0.))
      continue;
    Linear c{Eigen::VectorXd(dimension),
             full.dot(d.fixed) + d.prm.constraint_vector[row]};
    for (std::size_t k = 0; k < d.variables.size(); ++k) {
      const auto index = d.variables[k];
      c.a[static_cast<Eigen::Index>(k)] = full[index] * d.scales[index];
      c.b -= full[index] * d.fixed[index];
    }
    if (c.a.norm() > 0.)
      inequalities.push_back(std::move(c));
  }
  for (auto &c : inequalities)
    opt.add_inequality_constraint(
        [](const std::vector<double> &z, std::vector<double> &g,
           void *context) {
          auto &c = *static_cast<Linear *>(context);
          if (!g.empty())
            std::copy(c.a.data(), c.a.data() + c.a.size(), g.begin());
          return c.a.dot(Eigen::Map<const Eigen::VectorXd>(
                     z.data(), static_cast<Eigen::Index>(z.size()))) +
                 c.b;
        },
        &c, 1.e-10);
  std::vector<double> z, low, high;
  for (auto k : d.variables) {
    z.push_back(d.fixed[k] / d.scales[k]);
    const bool amount =
        (d.prm.phase_amount_indices == burnman::utils::checked_int(k)).any();
    low.push_back(k == 0 || amount ? 0. : -HUGE_VAL);
    high.push_back(HUGE_VAL);
    if (k < 2) {
      const auto bounds =
          k == 0 ? engine.pressure_bounds : engine.temperature_bounds;
      low.back() = std::min(bounds[0], d.fixed[k]) / d.scales[k];
      if (k == 1)
        low.back() = std::max(low.back(), 1.e-9);
      high.back() = std::max(bounds[1], d.fixed[k]) / d.scales[k];
    }
  }
  opt.set_lower_bounds(low);
  opt.set_upper_bounds(high);
  opt.set_xtol_abs(1.e-11);
  opt.set_ftol_abs(1.e-13);
  opt.set_maxeval(
      std::min(2000, 20 * engine.settings.max_refinement_iterations));
  d.evaluate(z);
  double value = 0.;
  try {
    opt.optimize(z, value);
    d.evaluate(z);
  } catch (const std::exception &error) {
    // A valid state retained above is still eligible for Newton polishing.
    if (engine.settings.verbose)
      std::cerr << "Constrained potential minimisation: " << error.what()
                << "; polishing the best valid state.\n";
  }
  if (d.fallback.size()) {
    std::vector<double> best(d.fallback.data(),
                             d.fallback.data() + d.fallback.size());
    set_composition_and_state_from_parameters(d.a, d.parameters(best));
  }
}
} // namespace

WorkState Engine::feasible_start(double p, double t) {
  WorkState out;
  out.state.pressure = p;
  out.state.temperature = t;
  try {
    set_pt(p, t);
    std::vector<std::pair<int, Eigen::VectorXd>> compounds;
    for (std::size_t base = 0; base < phases.size(); ++base) {
      const auto &phase = phases[base];
      if (!phase.available) {
        if (std::find(settings.required_eos_phases.begin(),
                      settings.required_eos_phases.end(),
                      phase.material->get_name()) !=
            settings.required_eos_phases.end())
          throw std::runtime_error("Required model outside EOS domain: " +
                                   phase.domain_error);
        continue;
      }
      const auto vertices = feasible_vertices(phase);
      for (Eigen::Index k = 0; k < vertices.rows(); ++k)
        compounds.emplace_back(static_cast<int>(base),
                               vertices.row(k).transpose());
      if (vertices.rows() > 1)
        compounds.emplace_back(static_cast<int>(base),
                               vertices.colwise().mean().transpose());
    }
    if (compounds.empty())
      return out;
    Eigen::MatrixXd formulae(compounds.size(), components.size());
    for (std::size_t k = 0; k < compounds.size(); ++k) {
      const auto &[base, composition] = compounds[k];
      const Eigen::VectorXd formula =
          phases[static_cast<std::size_t>(base)].a.transpose() * composition;
      for (std::size_t j = 0; j < components.size(); ++j)
        formulae(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)) =
            formula[components[j]];
    }
    const auto balance = polytope::gibbs_linear_program(
        formulae, Eigen::VectorXd::Zero(compounds.size()), b);
    potential_seed = balance.chemical_potentials;
    PhaseStates states;
    for (std::size_t base = 0; base < phases.size(); ++base) {
      PhaseState phase;
      phase.id = static_cast<int>(base) * settings.max_phase_instances;
      phase.candidate_index = static_cast<int>(base);
      phase.composition = Eigen::VectorXd::Zero(phases[base].vertices.cols());
      for (std::size_t k = 0; k < compounds.size(); ++k)
        if (compounds[k].first == static_cast<int>(base) &&
            balance.amounts[static_cast<Eigen::Index>(k)] > 0.) {
          const double amount = balance.amounts[static_cast<Eigen::Index>(k)];
          phase.amount += amount;
          phase.composition += amount * compounds[k].second;
        }
      if (phase.amount > 0.) {
        phase.composition /= phase.amount;
        states.push_back(std::move(phase));
      }
    }
    out.ids = phase_ids(states);
    out.assemblage = make_assemblage(out.ids, states, p, t);
    record_coordinates(out.state, *out.assemblage);
    out.state.phases = snapshot(*out.assemblage, out.ids);
    out.state.mass_balance_error = mass_error(*out.assemblage);
    out.state.message = "Mass-balanced starting assemblage";
  } catch (const std::exception &error) {
    out.state.message = error.what();
    out.state.outside_model_domain =
        std::any_of(phases.begin(), phases.end(),
                    [](const auto &phase) { return !phase.available; });
  }
  return out;
}

WorkState Engine::minimize_potential(const WorkState &seed,
                                     const Eigen::Vector2d &q) {
  auto axes = diagram_axes(section.type);
  if (axes[1] == Coordinate::X)
    axes[1] = *section.fixed_coordinate;
  std::vector<std::pair<Coordinate, double>> targets = {
      {axes[0], q[0]},
      {axes[1],
       has_composition_axis(section.type) ? section.fixed_value : q[1]}};
  WorkState current = seed;
  std::vector<PotentialStart> attempted;
  for (int iteration = 0;
       iteration < std::min(32, settings.max_refinement_iterations);
       ++iteration) {
    if (!current.assemblage || current.state.phases.empty())
      return current;
    try {
      auto states = current.state.phases;
      std::vector<Minimum> missing;
      stability(*current.assemblage, &missing);
      for (std::size_t base = 0; base < missing.size(); ++base) {
        if (missing[base].affinity >= -settings.affinity_tolerance)
          continue;
        std::vector<bool> used(
            static_cast<std::size_t>(settings.max_phase_instances), false);
        bool represented = false;
        for (const auto &state : states)
          if (state.candidate_index == static_cast<int>(base)) {
            used[static_cast<std::size_t>(state.id %
                                          settings.max_phase_instances)] = true;
            represented =
                represented || (state.composition - missing[base].p).norm() <
                                   settings.composition_tolerance;
          }
        auto copy = std::find(used.begin(), used.end(), false);
        if (represented || copy == used.end())
          continue;
        PhaseState entering;
        entering.candidate_index = static_cast<int>(base);
        entering.id = static_cast<int>(base) * settings.max_phase_instances +
                      static_cast<int>(copy - used.begin());
        entering.composition = missing[base].p;
        entering.amount = 0.;
        states.push_back(std::move(entering));
      }
      const double p = current.assemblage->get_pressure(),
                   t = current.assemblage->get_temperature();
      auto ids = phase_ids(states);
      auto a = make_assemblage(ids, states, p, t);
      if (!free_vectors.empty())
        set_coordinate(*a, q[1]);
      PotentialMinimum objective(*a, bulk, targets);
      const char *potential =
          objective.fix_entropy ? (objective.fix_volume ? "E" : "H") : "F";
      PotentialStart input(*this, objective, ids);
      // A phase can enter here and disappear again during Newton polishing.
      // Compare full optimisation inputs, including active solution faces,
      // rather than treating phase entry itself as evidence of progress.
      if (std::any_of(attempted.begin(), attempted.end(),
                      [&](const auto &old) { return input.same(old); })) {
        current.state.success = false;
        current.state.message =
            "Constrained potential search stalled at a repeated state.";
        if (settings.verbose)
          std::cerr << potential
                    << " minimisation stopped: repeated optimisation state; "
                       "trying another starting assemblage.\n";
        return current;
      }
      attempted.push_back(std::move(input));
      ++minimization_calls;
      const auto start = std::chrono::steady_clock::now();
      if (settings.verbose)
        std::cerr << potential << " minimisation " << iteration + 1 << ": "
                  << ids.size() << " phases; P=" << p << " Pa, T=" << t
                  << " K.\n";
      minimize_active_potential(objective, *this);
      if (settings.verbose)
        std::cerr << potential << " minimisation " << iteration + 1 << ": "
                  << objective.evaluations << " evaluations in "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << " s.\n";
      states = snapshot(*a, ids);
      auto active = present_phase_ids(
          states, std::min(settings.amount_tolerance, 1.e-10));
      auto next = equilibrate_phase_set(active, states, a->get_pressure(),
                                        a->get_temperature(), q);
      if (next.state.success)
        return next;
      // Polishing may reject the newly proposed phase and return precisely
      // the previous physical state. Its unused candidate composition is not
      // progress, even when that changes the next expanded optimisation input.
      if (same_retained_state(*this, current, next)) {
        next.state.success = false;
        next.state.message = "Constrained potential search stalled at an "
                             "unchanged retained state.";
        if (settings.verbose)
          std::cerr << potential
                    << " minimisation stopped: retained assemblage and state "
                       "unchanged; trying another starting assemblage.\n";
        return next;
      }
      current = std::move(next);
    } catch (const std::exception &error) {
      current.state.message = error.what();
      current.state.success = false;
      return current;
    }
  }
  return current;
}
} // namespace burnman::phase_diagrams::detail
