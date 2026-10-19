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

#include "burnman/tools/phase_diagram/tracing.hpp"
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/phase_diagram/boundary_repair.hpp"
#include "burnman/tools/phase_diagram/consistency.hpp"
#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/phase_diagram/critical.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/polygons.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/index_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
namespace burnman::phase_diagrams::detail {
Eigen::Vector2d Tracer::normalise(double first, double second) const {
  return (Eigen::Vector2d(first, second) - origin).cwiseQuotient(range);
}

Eigen::Vector2d Tracer::normalise(const Eigen::Vector2d &q) const {
  return (q - origin).cwiseQuotient(range);
}

Eigen::Vector2d Tracer::normalised_coordinates(const Assemblage &a) const {
  return normalise(engine.coordinates(a));
}

bool Tracer::inside(const Eigen::Vector2d &u, double margin) const {
  return (u.array() >= -margin).all() && (u.array() <= 1 + margin).all();
}

std::unique_ptr<EqualityConstraint>
Tracer::section(const Assemblage &a, int n, const Eigen::Vector2d &u,
                const Eigen::Vector2d &normal) const {
  const auto target = (origin + u.cwiseProduct(range)).eval();
  return continuation_plane(engine, a, n, target, normal.cwiseQuotient(range));
}

ConstraintList Tracer::boundary_constraints(const Assemblage &a,
                                            const std::vector<int> &ids,
                                            int zero, const Eigen::Vector2d &u,
                                            const Eigen::Vector2d &normal) {
  auto prm = engine.parameters(a);
  auto it = std::find(ids.begin(), ids.end(), zero);
  if (it == ids.end())
    throw std::runtime_error("Missing zero phase in boundary.");
  return constraints(
      std::make_unique<PhaseFractionConstraint>(it - ids.begin(), 0., prm),
      section(a, burnman::utils::checked_int(prm.n_parameters), u, normal));
}

EquilibriumSolve Tracer::solve_boundary(Assemblage &a,
                                        const std::vector<int> &ids, int zero,
                                        const Eigen::Vector2d &position,
                                        const Eigen::Vector2d &normal) {
  auto c = boundary_constraints(a, ids, zero, position, normal);
  return engine.solve(a, c);
}

bool Tracer::valid_amount_edge(const Assemblage &a, const std::vector<int> &ids,
                               int zero) {
  if (!engine.free_vectors.empty() ||
      std::any_of(ids.begin(), ids.end(), [&](int id) {
        return bool(engine
                        .phases[static_cast<std::size_t>(
                            id / engine.settings.max_phase_instances)]
                        .solution);
      }))
    return true;
  // For pure phases at fixed bulk, the amount polytope is independent of
  // P,T. Reject a phase-out condition if that phase can never be present.
  // When several amounts vanish together, use one canonical constraint.
  // This preserves reduced-variance reactions while avoiding duplicate
  // traces and equal-Gibbs lines between irrelevant zero-amount phases.
  auto [it, inserted] = amount_vertices.try_emplace(ids);
  if (inserted)
    it->second =
        polytope::composite_polytope_at_constrained_composition(a, engine.bulk)
            .get_vertices();
  const auto &vertices = it->second;
  if (!vertices.rows())
    return false;
  double tolerance = 1.e-10 * vertices.cwiseAbs().maxCoeff();
  auto z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
  if (vertices.col(z).maxCoeff() <= tolerance)
    return false;
  std::vector<Eigen::Index> face;
  for (Eigen::Index i = 0; i < vertices.rows(); ++i)
    if (std::abs(vertices(i, z)) <= tolerance)
      face.push_back(i);
  if (face.empty())
    return false;
  for (Eigen::Index j = 0; j < z; ++j)
    if (std::all_of(face.begin(), face.end(), [&](Eigen::Index i) {
          return std::abs(vertices(i, j)) <= tolerance;
        }))
      return false;
  return true;
}

Eigen::VectorXd Tracer::tangent(const EquilibriumSolve &s, const Assemblage &a,
                                int *coordinate_rank) {
  return continuation_tangent(engine, s, a, range, coordinate_rank);
}

BoundaryPoint Tracer::point(const Assemblage &a, const std::vector<int> &ids,
                            const EquilibriumSolve &solve, double affinity) {
  BoundaryPoint p;
  engine.record_coordinates(p, a);
  p.phases = engine.snapshot(a, ids);
  p.mass_balance_error = engine.mass_error(a);
  p.minimum_affinity = affinity;
  p.residual = solve.F.size() > 2 + static_cast<int>(engine.free_vectors.size())
                   ? solve.F
                         .tail(solve.F.size() - 2 -
                               static_cast<int>(engine.free_vectors.size()))
                         .norm()
                   : 0.;
  return p;
}

bool Tracer::valid(const Assemblage &a, const EquilibriumSolve &s) const {
  return s.success && s.x.allFinite() && inside(normalised_coordinates(a)) &&
         engine.mass_error(a) <= engine.settings.mass_balance_tolerance &&
         engine.locally_stable(a);
}

bool Tracer::needs_face_refit(double affinity, bool composition) const {
  // A trace site can lower the energy by much less than the phase-entry
  // tolerance. Releasing its face changes a composition, not the assemblage.
  const double tolerance =
      engine.settings.affinity_tolerance * (composition ? .05 : 1.);
  return affinity < -(engine.direct_coordinates() && !composition
                          ? tolerance
                          : std::min(tolerance, 1.e-5));
}

double Tracer::phase_entry_tolerance(std::size_t base,
                                     const PhaseStates &states,
                                     const Minimum &minimum) const {
  double tolerance = engine.settings.affinity_tolerance;
  bool present = false, separate = true;
  const auto &phase = engine.phases[base];
  for (const auto &state : states)
    if (state.candidate_index == static_cast<int>(base)) {
      present = true;
      separate = separate && engine.distinct_compositions(
                                 static_cast<int>(base), state.composition,
                                 minimum.p, phase.material->get_pressure(),
                                 phase.material->get_temperature());
    }
  // A resolved composition difference need not be a second Gibbs basin.
  // A lower-energy homogeneous mixture rules out a weak solvus entry and
  // avoids repeatedly trying a copy that the corrector must reject.
  if (present && separate)
    tolerance = std::min(tolerance, 1.e-5);
  return tolerance;
}

bool Tracer::same_branch(const Assemblage &a, const Assemblage &b,
                         const std::vector<int> &ids) const {
  auto old = engine.snapshot(a, ids), next = engine.snapshot(b, ids);
  if (!same_solution_branches(engine, old, next, b.get_pressure(),
                              b.get_temperature()))
    return false;
  // Ordinary continuation must not exchange or coalesce existing copies.
  // Junction/critical corrections can meet them under separate constraints.
  for (std::size_t i = 0; i < ids.size(); ++i)
    for (std::size_t j = i + 1; j < ids.size(); ++j)
      if (old[i].candidate_index == old[j].candidate_index &&
          (old[i].composition - old[j].composition)
                  .dot(next[i].composition - next[j].composition) <= 0.)
        return false;
  return true;
}

bool Tracer::near_third_copy_event(const std::vector<int> &ids, int zero,
                                   int other, const PhaseStates &states) const {
  const int base = zero / engine.settings.max_phase_instances;
  if (engine.direct_coordinates() ||
      base != other / engine.settings.max_phase_instances ||
      !engine.phases[static_cast<std::size_t>(base)].solution ||
      std::count_if(ids.begin(), ids.end(), [&](int id) {
        return id / engine.settings.max_phase_instances == base;
      }) < 3)
    return false;
  for (std::size_t i = 0; i < states.size(); ++i)
    for (std::size_t j = i + 1; j < states.size(); ++j)
      if (states[i].candidate_index == base &&
          states[j].candidate_index == base &&
          (states[i].composition - states[j].composition).norm() < .01)
        return true;
  return false;
}

bool Tracer::closing_solvus(const Assemblage &a, const std::vector<int> &ids,
                            int zero, const Eigen::VectorXd &direction,
                            const EquilibrationParameters &prm) const {
  auto states = engine.snapshot(a, ids);
  auto z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
  auto composition_direction = [&](Eigen::Index k) -> Eigen::VectorXd {
    auto phase = a.get_phase(static_cast<std::size_t>(k));
    const auto basis = engine.composition_basis(
        *phase, states[static_cast<std::size_t>(k)].candidate_index);
    Eigen::VectorXd dp = Eigen::VectorXd::Zero(basis.rows());
    dp.tail(dp.size() - 1) =
        direction.segment(prm.phase_amount_indices[k] + 1, dp.size() - 1);
    dp[0] = -dp.tail(dp.size() - 1).sum();
    return basis.transpose() * dp;
  };
  for (std::size_t k = 0; k < states.size(); ++k)
    if (k != static_cast<std::size_t>(z) &&
        states[k].candidate_index ==
            states[static_cast<std::size_t>(z)].candidate_index) {
      auto delta = (states[static_cast<std::size_t>(z)].composition -
                    states[k].composition)
                       .eval();
      if (delta.norm() < .01 &&
          delta.dot(composition_direction(z) -
                    composition_direction(static_cast<Eigen::Index>(k))) < 0.)
        return true;
    }
  return false;
}

void Tracer::predict(Assemblage &a, const Eigen::VectorXd &x,
                     const Eigen::VectorXd &delta,
                     const EquilibrationParameters &prm) {
  Eigen::VectorXd guess = x + delta;
  for (int j : prm.phase_amount_indices)
    guess[j] = std::max(0., guess[j]);
  auto change = (guess - x).eval();
  auto slack = (prm.constraint_matrix * x + prm.constraint_vector).eval();
  auto slope = (prm.constraint_matrix * change).eval();
  double fraction = 1.;
  for (int i = 2; i < slope.size(); ++i)
    if (slope[i] > 1.e-14)
      fraction = std::min(fraction, std::max(0., -slack[i]) / slope[i]);
  guess = x + (fraction < 1. ? fraction * .99 : 1.) * change;
  guess.head<2>() = (x + delta).head<2>();
  const double old_coordinate = engine.composition_coordinate(a);
  if (!engine.free_vectors.empty())
    guess[guess.size() - 1] = (x + delta)[guess.size() - 1];
  for (int j : prm.phase_amount_indices)
    guess[j] = std::max(0., guess[j]);
  set_composition_and_state_from_parameters(a, guess);
  if (!engine.free_vectors.empty())
    engine.set_coordinate(a, old_coordinate + (guess - x)[guess.size() - 1]);
}

bool Tracer::composition_order(const PhaseState &a, const PhaseState &b) {
  if (a.candidate_index != b.candidate_index)
    return a.candidate_index < b.candidate_index;
  for (int k = 0; k < a.composition.size(); ++k) {
    auto left = std::llround(a.composition[k] * 1.e7),
         right = std::llround(b.composition[k] * 1.e7);
    if (left != right)
      return left < right;
  }
  return a.id < b.id;
}

Tracer::Seed Tracer::canonical_seed(const Seed &original) const {
  Seed out = original;
  PhaseStates states;
  for (auto &state : out.phases)
    if (std::find(out.ids.begin(), out.ids.end(), state.id) != out.ids.end())
      states.push_back(state);
  std::sort(states.begin(), states.end(), composition_order);
  std::map<int, int> count, renumber;
  out.ids.clear();
  for (auto &state : states) {
    int id = state.candidate_index * engine.settings.max_phase_instances +
             count[state.candidate_index]++;
    if (state.id == original.zero)
      out.zero = id;
    renumber[state.id] = id;
    state.id = id;
    state.name = result.phase_names[static_cast<std::size_t>(id)];
    out.ids.push_back(id);
  }
  for (auto &point : out.prefix) {
    for (auto &phase : point.phases) {
      phase.id = renumber.at(phase.id);
      phase.name = result.phase_names[static_cast<std::size_t>(phase.id)];
    }
    std::sort(point.phases.begin(), point.phases.end(),
              [](auto &a, auto &b) { return a.id < b.id; });
  }
  out.phases = std::move(states);
  return out;
}

void Tracer::add_sample(const State &state, int existing) {
  int index = existing >= 0
                  ? existing
                  : burnman::utils::checked_int(result.samples.size());
  State canonical = state;
  if (state.success) {
    const double total = total_amount(state.phases);
    erase_if(canonical.phases, [&](auto &ph) {
      return ph.amount <= engine.settings.amount_tolerance * total;
    });
    // A saved, nearly identical solution pair is one phase, not two minima.
    // Weighted averaging preserves elemental mass exactly for linear
    // formulae.
    for (std::size_t i = 0; i < canonical.phases.size(); ++i)
      for (std::size_t j = i + 1; j < canonical.phases.size();) {
        auto &a = canonical.phases[i];
        auto &b = canonical.phases[j];
        if (a.candidate_index == b.candidate_index &&
            (a.composition - b.composition).norm() <=
                engine.settings.composition_tolerance * 10.) {
          a.composition =
              (a.amount * a.composition + b.amount * b.composition) /
              (a.amount + b.amount);
          a.amount += b.amount;
          canonical.phases.erase(canonical.phases.begin() +
                                 static_cast<std::ptrdiff_t>(j));
        } else
          ++j;
      }
    std::sort(canonical.phases.begin(), canonical.phases.end(),
              composition_order);
    std::map<int, int> count;
    for (auto &ph : canonical.phases) {
      ph.id = ph.candidate_index * engine.settings.max_phase_instances +
              count[ph.candidate_index]++;
      ph.name = result.phase_names[static_cast<std::size_t>(ph.id)];
    }
  }
  if (existing >= 0) {
    result.samples[static_cast<std::size_t>(existing)] = canonical;
    // Reverification can replace an assemblage. Remove its old membership
    // so an obsolete field cannot survive solely through that sample index.
    for (auto &field : result.fields)
      field.sample_indices.erase(std::remove(field.sample_indices.begin(),
                                             field.sample_indices.end(),
                                             existing),
                                 field.sample_indices.end());
    erase_if(result.fields,
             [](const Field &field) { return field.sample_indices.empty(); });
    field_index.clear();
    for (std::size_t i = 0; i < result.fields.size(); ++i) {
      auto &field = result.fields[i];
      field.id = burnman::utils::checked_int(i);
      field_index[field.phases] = field.id;
    }
  } else
    result.samples.push_back(canonical);
  if (!state.success) {
    std::ostringstream msg;
    msg << "Unresolved seed P=" << state.pressure
        << " Pa, T=" << state.temperature << " K: " << state.message;
    result.diagnostics.push_back(msg.str());
    return;
  }
  auto ids = engine.active_phases(canonical);
  auto it = field_index.find(ids);
  int id;
  if (it == field_index.end()) {
    id = burnman::utils::checked_int(result.fields.size());
    field_index[ids] = id;
    result.fields.push_back({id, ids, {}});
  } else
    id = it->second;
  result.fields[static_cast<std::size_t>(id)].sample_indices.push_back(index);
}

bool Tracer::covered(const Seed &seed) const {
  for (auto &line : result.boundaries)
    if (line.assemblage == seed.ids && line.zero_phase == seed.zero) {
      // An incident branch must retain the same solution compositions;
      // renumbering copies alone does not establish that it covers a seed.
      if (seed.node >= 0) {
        if (!line.points.empty())
          for (bool start : {true, false})
            if ((start ? line.start_node : line.end_node) == seed.node &&
                same_boundary_compositions(
                    seed.phases, seed.zero,
                    (start ? line.points.front() : line.points.back()).phases,
                    line.zero_phase, engine.settings.composition_tolerance,
                    engine.settings.amount_tolerance))
              return true;
        continue;
      }
      for (std::size_t k = 1; k < line.points.size(); ++k) {
        auto a = normalise(line.points[k - 1]), b = normalise(line.points[k]);
        auto d = (b - a).eval();
        double f =
            d.squaredNorm() > 0
                ? std::clamp((seed.position - a).dot(d) / d.squaredNorm(), 0.,
                             1.)
                : 0.;
        if ((a + f * d - seed.position).norm() <
            engine.settings.node_tolerance * 2)
          return true;
      }
    }
  return false;
}

std::vector<int> Tracer::event_signature(const std::vector<int> &ids,
                                         bool polymorphs) const {
  std::vector<int> out;
  for (int id : ids) {
    int base = id / engine.settings.max_phase_instances;
    if (polymorphs && !engine.phases[static_cast<std::size_t>(base)].solution)
      for (int j = 0; j < base; ++j)
        if (!engine.phases[static_cast<std::size_t>(j)].solution &&
            engine.phases[static_cast<std::size_t>(j)]
                    .material->get_formula() ==
                engine.phases[static_cast<std::size_t>(base)]
                    .material->get_formula()) {
          base = j;
          break;
        }
    out.push_back(base);
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool Tracer::same_event(const Node &previous, const std::vector<int> &ids,
                        const std::vector<int> &zero) const {
  if (!engine.direct_coordinates() && previous.kind == "junction") {
    // Replacing equal-formula polymorphs, or different solution copies,
    // changes latent entropy/volume. Their zero-amount choices can be
    // distinct nearby vertices even at identical P,T.
    auto old_zero = previous.zero_phases, next_zero = zero;
    std::sort(old_zero.begin(), old_zero.end());
    std::sort(next_zero.begin(), next_zero.end());
    if (old_zero != next_zero)
      return false;
  }
  return event_signature(previous.assemblage) == event_signature(ids) &&
         event_signature(previous.zero_phases, engine.direct_coordinates()) ==
             event_signature(zero, engine.direct_coordinates());
}

int Tracer::node(Assemblage &a, const std::vector<int> &ids,
                 const std::vector<int> &zero, const std::string &kind,
                 const Eigen::MatrixXd *jacobian) {
  auto u = normalised_coordinates(a);
  for (auto &old : result.nodes)
    if ((normalise(old) - u).norm() < engine.settings.node_tolerance) {
      double distance = (normalise(old) - u).norm();
      bool compatible_kind =
          old.kind == kind ||
          (kind == "critical_point" && old.kind == "reduced_variance");
      if (distance > 1.e-8 && (!compatible_kind || !same_event(old, ids, zero)))
        continue;
      if (kind == "critical_point" &&
          solution_separation(engine.snapshot(a, ids)) < 1.e-8 &&
          critical_axis.size()) {
        engine.record_coordinates(old, a);
        old.critical_mode = critical_axis;
        old.kind = "critical_point";
      }
      if (kind == "junction")
        expand(a, ids, zero, old.id);
      if (kind == "critical_point")
        expand_critical(a, ids, zero, old.id);
      return old.id;
    }
  Node n;
  n.id = burnman::utils::checked_int(result.nodes.size());
  engine.record_coordinates(n, a);
  n.kind = kind;
  n.assemblage = ids;
  n.zero_phases = zero;
  if (kind == "critical_point" &&
      solution_separation(engine.snapshot(a, ids)) < 1.e-8)
    n.critical_mode = critical_axis;
  n.gibbs_variance = field_variance(a, ids.size());
  n.pt_nullity = 0;
  // An intrinsic critical solve contains one coalesced solution copy. Its
  // Jacobian cannot be projected against the restored two-copy endpoint.
  if (jacobian && jacobian->cols() == engine.parameters(a).n_parameters) {
    auto scales = engine.parameter_scales(a, jacobian->cols(), range);
    Eigen::MatrixXd j = (*jacobian) * scales.asDiagonal();
    for (int i = 0; i < j.rows(); ++i)
      if (j.row(i).norm() > 0)
        j.row(i) /= j.row(i).norm();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeFullV);
    int rank = burnman::utils::checked_int(
        (svd.singularValues().array() > 1.e-9).count());
    if (rank < j.cols()) {
      Eigen::MatrixXd projection =
          range.cwiseInverse().asDiagonal() *
          engine.coordinate_jacobian(a, jacobian->cols()) *
          scales.asDiagonal() * svd.matrixV().rightCols(j.cols() - rank);
      Eigen::JacobiSVD<Eigen::MatrixXd> ptsvd(projection);
      n.pt_nullity = burnman::utils::checked_int(
          (ptsvd.singularValues().array() > 1.e-7).count());
    }
  }
  result.nodes.push_back(n);
  if (kind == "junction")
    expand(a, ids, zero, n.id);
  if (kind == "critical_point")
    expand_critical(a, ids, zero, n.id);
  return n.id;
}

void Tracer::separate_junctions() {
  // Older traces merged nearby, thermodynamically different events. Their
  // accepted endpoint states still locate each event; restore that topology
  // without moving any thermodynamic point or drawing a guessed connection.
  for (auto &line : result.boundaries) {
    if (line.points.empty())
      continue;
    auto [back, forward] = boundary_statuses(line);
    for (bool start : {true, false}) {
      if ((start ? back : forward) != "junction")
        continue;
      auto &id = start ? line.start_node : line.end_node;
      if (id < 0 ||
          result.nodes.at(static_cast<std::size_t>(id)).kind != "junction")
        continue;
      auto &p = start ? line.points.front() : line.points.back();
      auto u = normalise(p);
      std::vector<int> ids, zeros;
      double total = 0.;
      for (auto &ph : p.phases)
        total += std::abs(ph.amount);
      for (auto &ph : p.phases) {
        ids.push_back(ph.id);
        if (std::abs(ph.amount) <= 1.e-10 * std::max(1., total))
          zeros.push_back(ph.id);
      }
      if (zeros.size() < 2)
        continue;
      int found = -1;
      for (auto &n : result.nodes)
        if (n.kind == "junction") {
          double distance = (normalise(n) - u).norm();
          if (distance < 1.e-8 || (distance < engine.settings.node_tolerance &&
                                   same_event(n, ids, zeros))) {
            found = n.id;
            break;
          }
        }
      if (found < 0) {
        Node n;
        n.id = burnman::utils::checked_int(result.nodes.size());
        n.pressure = p.pressure;
        n.temperature = p.temperature;
        n.entropy = p.entropy;
        n.volume = p.volume;
        n.composition_coordinate = p.composition_coordinate;
        n.kind = "junction";
        n.assemblage = ids;
        n.zero_phases = zeros;
        auto a = engine.make_at(ids, p.phases,
                                diagram_coordinates(p, result.section.type),
                                Eigen::Vector2d(p.pressure, p.temperature));
        n.gibbs_variance = field_variance(*a, ids.size());
        result.nodes.push_back(n);
        found = n.id;
      }
      id = found;
    }
  }
}

ConstraintList Tracer::separation_constraints(const Assemblage &a,
                                              const std::vector<int> &ids,
                                              int zero, int first, int second,
                                              const Eigen::VectorXd &axis,
                                              double target) {
  auto prm = engine.parameters(a);
  Eigen::VectorXd coefficient = Eigen::VectorXd::Zero(prm.n_parameters);
  for (auto [phase, sign] :
       {std::pair<int, double>{first, 1.}, {second, -1.}}) {
    Eigen::VectorXd projected =
        engine.composition_basis(*a.get_phase(static_cast<std::size_t>(phase)),
                                 ids[static_cast<std::size_t>(phase)] /
                                     engine.settings.max_phase_instances) *
        axis;
    coefficient.segment(prm.phase_amount_indices[phase] + 1,
                        projected.size() - 1) =
        sign *
        (projected.tail(projected.size() - 1).array() - projected[0]).matrix();
  }
  int z = burnman::utils::checked_int(std::find(ids.begin(), ids.end(), zero) -
                                      ids.begin());
  return constraints(std::make_unique<PhaseFractionConstraint>(z, 0., prm),
                     std::make_unique<LinearXConstraint>(coefficient, target));
}

bool Tracer::approach_critical(std::shared_ptr<Assemblage> &a,
                               const std::vector<int> &ids, int zero,
                               EquilibriumSolve &solve,
                               std::vector<BoundaryPoint> &points) {
  critical_axis.resize(0);
  auto initial = engine.snapshot(*a, ids);
  int z = burnman::utils::checked_int(std::find(ids.begin(), ids.end(), zero) -
                                      ids.begin());
  for (std::size_t k = 0; k < ids.size(); ++k)
    if (k != static_cast<std::size_t>(z) &&
        initial[k].candidate_index ==
            initial[static_cast<std::size_t>(z)].candidate_index) {
      Eigen::VectorXd axis = initial[static_cast<std::size_t>(z)].composition -
                             initial[k].composition;
      double delta = axis.norm();
      if (delta <= 1.e-8) {
        auto &previous = points.empty() ? initial : points.back().phases;
        for (auto &ph : previous)
          if (ph.id == zero)
            for (auto &other : previous)
              if (other.id == ids[k])
                axis = ph.composition - other.composition;
        if (axis.norm() <= 1.e-8)
          for (auto &n : result.nodes)
            if (n.critical_mode.size() == axis.size() &&
                (normalise(n) - normalised_coordinates(*a)).norm() <
                    engine.settings.node_tolerance)
              axis = n.critical_mode * .001;
        delta = axis.norm();
      }
      if (delta <= 1.e-8 || delta >= .04)
        continue;
      axis /= delta;
      // Solve the intrinsic critical conditions on a single surviving copy.
      // The third derivative is analytic, avoiding cancellation of chemical
      // potentials that differ by the cube of the composition separation.
      try {
        auto subset = ids;
        subset.erase(subset.begin() + z);
        auto states = initial;
        states[k].composition =
            (initial[static_cast<std::size_t>(z)].composition +
             initial[k].composition) *
            .5;
        states[k].amount += initial[static_cast<std::size_t>(z)].amount;
        auto critical = engine.make_assemblage(
            subset, states, a->get_pressure(), a->get_temperature());
        int phase = burnman::utils::checked_int(
            std::find(subset.begin(), subset.end(), ids[k]) - subset.begin());
        if (!critical->get_phase<Solution>(static_cast<std::size_t>(phase)))
          continue;
        auto prm = engine.parameters(*critical);
        auto mode_basis = engine.composition_basis(
            *critical->get_phase(static_cast<std::size_t>(phase)),
            ids[k] / engine.settings.max_phase_instances);
        Eigen::VectorXd projected_axis =
            mode_basis.transpose().completeOrthogonalDecomposition().solve(
                axis);
        auto curvature = std::make_unique<CriticalConstraint>(
            *critical, phase, prm, projected_axis, false);
        auto third = std::make_unique<CriticalConstraint>(*critical, phase, prm,
                                                          projected_axis, true);
        auto c = constraints(std::move(curvature), std::move(third));
        auto s = engine.solve(*critical, c);
        if (valid(*critical, s)) {
          double affinity = engine.stability(*critical);
          if (affinity >= -engine.settings.affinity_tolerance) {
            critical_axis = mode_basis.transpose() *
                            static_cast<CriticalConstraint *>(c[0][0].get())
                                ->mode(*critical, phase);
            auto full = engine.snapshot(*critical, subset);
            PhaseState absent = initial[static_cast<std::size_t>(z)];
            absent.amount = 0.;
            absent.composition =
                full[static_cast<std::size_t>(phase)].composition;
            full.push_back(absent);
            a = engine.make_assemblage(ids, full, critical->get_pressure(),
                                       critical->get_temperature());
            solve = s;
            points.push_back(point(*a, ids, solve, affinity));
            return true;
          }
        }
      } catch (const std::exception &error) {
        if (engine.settings.verbose)
          std::cerr << "Intrinsic critical solve: " << error.what() << '\n';
      }
      auto reflected = initial;
      reflected[static_cast<std::size_t>(z)].amount = initial[k].amount;
      reflected[static_cast<std::size_t>(z)].composition =
          initial[k].composition;
      reflected[k].amount = 0.;
      reflected[k].composition =
          2. * initial[k].composition -
          initial[static_cast<std::size_t>(z)].composition;
      const double p = a->get_pressure(), t = a->get_temperature();
      try {
        auto left_base = engine.make_assemblage(ids, initial, p, t),
             right_base = engine.make_assemblage(ids, reflected, p, t);
        auto left_constraints =
            separation_constraints(*left_base, ids, zero, z,
                                   burnman::utils::checked_int(k), axis, delta);
        auto right_constraints =
            separation_constraints(*right_base, ids, ids[k], z,
                                   burnman::utils::checked_int(k), axis, delta);
        auto left_solve = engine.solve(*left_base, left_constraints),
             right_solve = engine.solve(*right_base, right_constraints);
        if (!valid(*left_base, left_solve) || !valid(*right_base, right_solve))
          continue;
        // Both arms must converge, remain stable and agree in P,T within the
        // explicitly chosen node tolerance. Never connect distant open
        // endpoints.
        for (double factor = 1.; factor >= 1. / 4096. && delta * factor > 1.e-8;
             factor *= .5)
          try {
            auto left = engine.copy_assemblage(*left_base);
            auto right = engine.copy_assemblage(*right_base);
            for (auto entry :
                 {std::pair{left, left_solve}, std::pair{right, right_solve}}) {
              Eigen::VectorXd rhs =
                  Eigen::VectorXd::Zero(entry.second.x.size());
              rhs[1] = delta * (factor - 1.);
              Eigen::VectorXd dx = entry.second.J.partialPivLu().solve(rhs);
              if (dx.allFinite())
                predict(*entry.first, entry.second.x, dx,
                        engine.parameters(*entry.first));
            }
            auto lc = separation_constraints(*left, ids, zero, z,
                                             burnman::utils::checked_int(k),
                                             axis, delta * factor);
            auto rc = separation_constraints(*right, ids, ids[k], z,
                                             burnman::utils::checked_int(k),
                                             axis, delta * factor);
            auto ls = engine.solve(*left, lc), rs = engine.solve(*right, rc);
            if (engine.settings.verbose)
              std::cerr << "Critical approach " << zero
                        << ", separation=" << delta * factor << ": arms "
                        << ls.success << '/' << rs.success << ", distance="
                        << (normalised_coordinates(*left) -
                            normalised_coordinates(*right))
                               .norm()
                        << ", left constraints=" << ls.F.head<2>().transpose()
                        << ", left residual="
                        << ls.F.tail(ls.F.size() - 2).cwiseAbs().maxCoeff()
                        << ", left mass=" << engine.mass_error(*left) << '\n';
            if (!valid(*left, ls) || !valid(*right, rs) ||
                solution_separation(engine.snapshot(*left, ids)) <
                    delta * factor * .5 ||
                solution_separation(engine.snapshot(*right, ids)) <
                    delta * factor * .5 ||
                (normalised_coordinates(*left) - normalised_coordinates(*right))
                        .norm() > engine.settings.node_tolerance)
              continue;
            double la = engine.stability(*left), ra = engine.stability(*right);
            if (std::min(la, ra) < -engine.settings.affinity_tolerance)
              continue;
            a = left;
            solve = ls;
            points.push_back(point(*a, ids, solve, la));
            return true;
          } catch (const std::exception &) {
          }
      } catch (const std::exception &) {
      }
    }
  return false;
}

void Tracer::expand_critical(const Assemblage &a, const std::vector<int> &ids,
                             const std::vector<int> &zeros, int index) {
  // Construct the adjoining solvus arm with a composition-section corrector,
  // then continue away from coalescence until a unique P,T tangent returns.
  auto states = engine.snapshot(a, ids);
  for (int zero : zeros)
    for (std::size_t k = 0; k < ids.size(); ++k)
      if (ids[k] != zero && ids[k] / engine.settings.max_phase_instances ==
                                zero / engine.settings.max_phase_instances) {
        std::ostringstream key;
        key << "critical:" << index << ':' << zero << ':' << ids[k];
        if (!expanded.insert(key.str()).second)
          continue;
        auto reflected = states;
        auto z = std::find(ids.begin(), ids.end(), zero) - ids.begin();
        reflected[static_cast<std::size_t>(z)].amount = states[k].amount;
        reflected[static_cast<std::size_t>(z)].composition =
            states[k].composition;
        reflected[k].amount = 0.;
        reflected[k].composition =
            2. * states[k].composition -
            states[static_cast<std::size_t>(z)].composition;
        auto &phase =
            engine.phases[static_cast<std::size_t>(states[k].candidate_index)];
        if (!phase.solution ||
            (reflected[k].composition.transpose() * phase.occupancies)
                    .minCoeff() < -1.e-10)
          continue;
        int other = ids[k];
        Eigen::VectorXd axis = states[static_cast<std::size_t>(z)].composition -
                               states[k].composition;
        double delta = axis.norm();
        bool exact =
            delta <= 1.e-8 && result.nodes[static_cast<std::size_t>(index)]
                                      .critical_mode.size() == axis.size();
        if (exact) {
          axis = result.nodes[static_cast<std::size_t>(index)].critical_mode;
          delta = .008;
          reflected[static_cast<std::size_t>(z)].composition =
              states[k].composition + axis * delta * .5;
          reflected[k].composition = states[k].composition - axis * delta * .5;
        } else {
          if (delta <= 1.e-8)
            continue;
          axis /= delta;
        }
        try {
          auto near = engine.make_assemblage(ids, reflected, a.get_pressure(),
                                             a.get_temperature());
          auto nc = separation_constraints(
              *near, ids, other, burnman::utils::checked_int(z),
              burnman::utils::checked_int(k), axis, delta);
          auto ns = engine.solve(*near, nc);
          if (engine.settings.verbose)
            std::cerr << "Critical adjoining arm " << other << ": "
                      << ns.success << ", distance="
                      << (normalised_coordinates(*near) -
                          normalised_coordinates(a))
                             .norm()
                      << '\n';
          if (!valid(*near, ns) ||
              solution_separation(engine.snapshot(*near, ids)) < delta * .5 ||
              (!exact &&
               (normalised_coordinates(*near) - normalised_coordinates(a))
                       .norm() > engine.settings.node_tolerance))
            continue;
          double affinity = engine.stability(*near);
          if (affinity < -engine.settings.affinity_tolerance)
            continue;
          std::vector<BoundaryPoint> prefix;
          if (exact) {
            BoundaryPoint critical;
            engine.record_coordinates(critical, a);
            critical.composition_coordinate = engine.composition_coordinate(a);
            critical.phases = states;
            critical.phases[static_cast<std::size_t>(z)].amount =
                states[k].amount;
            critical.phases[k].amount = 0.;
            critical.mass_balance_error = engine.mass_error(a);
            critical.minimum_affinity = affinity;
            prefix.push_back(std::move(critical));
          }
          prefix.push_back(point(*near, ids, ns, affinity));
          // Retain only verified points. A failed trial starts again from the
          // last accepted compositions, with an enlarged separation section
          // as fallback.
          auto accepted = near;
          auto accepted_solve = ns;
          for (double target = delta * 2.; target <= .04; target *= 2.) {
            auto trial = engine.make_assemblage(
                ids, engine.snapshot(*accepted, ids), accepted->get_pressure(),
                accepted->get_temperature());
            Eigen::VectorXd rhs =
                Eigen::VectorXd::Zero(accepted_solve.x.size());
            auto current = engine.snapshot(*accepted, ids);
            rhs[1] = target -
                     axis.dot(current[static_cast<std::size_t>(z)].composition -
                              current[k].composition);
            Eigen::VectorXd dx = accepted_solve.J.partialPivLu().solve(rhs);
            if (dx.allFinite())
              predict(*trial, accepted_solve.x, dx, engine.parameters(*trial));
            auto c = separation_constraints(
                *trial, ids, other, burnman::utils::checked_int(z),
                burnman::utils::checked_int(k), axis, target);
            auto s = engine.solve(*trial, c);
            if (engine.settings.verbose)
              std::cerr << "Critical arm separation " << target << ": "
                        << s.success << '\n';
            if (!valid(*trial, s) ||
                solution_separation(engine.snapshot(*trial, ids)) < target * .5)
              continue;
            double aff = engine.stability(*trial);
            if (aff < -engine.settings.affinity_tolerance)
              continue;
            prefix.push_back(point(*trial, ids, s, aff));
            accepted = trial;
            accepted_solve = s;
            auto d = tangent(s, *trial);
            if (target >= .008 && d.size() && prefix.size() > 1) {
              auto previous = normalise(prefix[prefix.size() - 2]);
              Seed seed{
                  ids,
                  other,
                  -2,
                  engine.snapshot(*trial, ids),
                  normalised_coordinates(*trial),
                  (normalised_coordinates(*trial) - previous).normalized()};
              seed.prefix = std::move(prefix);
              seed.bridge_node = index;
              seed.physical = Eigen::Vector2d(trial->get_pressure(),
                                              trial->get_temperature());
              queue.push_back(std::move(seed));
              break;
            }
          }
        } catch (const std::exception &error) {
          if (engine.settings.verbose)
            std::cerr << "Critical branch recovery: " << error.what() << '\n';
        }
      }
}

void Tracer::expand(const Assemblage &a, const std::vector<int> &ids,
                    const std::vector<int> &zeros, int index) {
  std::ostringstream key;
  key << index << ':';
  for (int id : ids)
    key << id << ',';
  key << ':';
  for (int z : zeros)
    key << z << ',';
  if (!expanded.insert(key.str()).second)
    return;
  // A generic double-zero point supplies four branches: each zero phase in
  // the full assemblage, and each zero with the other absent. A broader
  // removal search also discovers polymorph replacements at reduced variance.
  auto states = engine.snapshot(a, ids);
  Eigen::MatrixXd formulae =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(ids.size()),
                            static_cast<Eigen::Index>(engine.elements.size()));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    auto f = a.get_phase(i)->get_formula();
    for (std::size_t j = 0; j < engine.elements.size(); ++j)
      if (f.count(engine.elements[j]))
        formulae(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            f.at(engine.elements[j]);
  }
  Eigen::FullPivLU<Eigen::MatrixXd> amounts_rank(formulae);
  amounts_rank.setThreshold(1.e-9);
  bool reduced_amount_variance =
      amounts_rank.rank() < static_cast<int>(ids.size());
  int gibbs_variance = field_variance(a, ids.size());
  for (int z : zeros) {
    // A P-T invariant can project to a finite latent S/V edge. The tangent
    // projection below rejects any branch invisible in the chosen axes.
    if (gibbs_variance >= 0 &&
        (result.section.type != DiagramType::PT || gibbs_variance > 0)) {
      queue.push_back({ids, z, index, states, normalised_coordinates(a),
                       Eigen::Vector2d::Zero()});
      queue.back().physical =
          Eigen::Vector2d(a.get_pressure(), a.get_temperature());
    }
    for (int removed : ids)
      if (removed != z &&
          (reduced_amount_variance ||
           std::find(zeros.begin(), zeros.end(), removed) != zeros.end())) {
        auto subset = ids;
        subset.erase(std::find(subset.begin(), subset.end(), removed));
        auto trial = states;
        if (result.section.type == DiagramType::PT && gibbs_variance == 0) {
          // At a P-T invariant, compositions are fixed while reaction
          // extents change. A mass-balance LP identifies feasible limiting
          // mixtures before an expensive boundary Newton solve is queued.
          try {
            auto balanced = reaction_seed_amounts(engine, subset, states, z);
            if (!balanced)
              continue;
            trial = std::move(*balanced);
          } catch (const std::exception &) {
            // A numerical LP failure is not an infeasibility certificate.
          }
        }
        queue.push_back({subset, z, index, trial, normalised_coordinates(a),
                         Eigen::Vector2d::Zero()});
        queue.back().physical =
            Eigen::Vector2d(a.get_pressure(), a.get_temperature());
      }
  }
}

void Tracer::bracket(WorkState left, WorkState right, int depth) {
  if (!left.state.success || !right.state.success)
    return;
  // Match copy IDs only in reaction seeds. Keep the endpoint WorkStates
  // consistent with their assemblages for midpoint continuation.
  auto left_phases = left.state.phases, right_phases = right.state.phases;
  // Copy IDs are local to an assemblage. On entering a solvus the sole
  // existing composition may become copy #2: match compositions before
  // assigning the zero amount, rather than averaging different branches.
  auto indices = [&](const PhaseStates &phases) {
    std::map<int, std::vector<std::size_t>> groups;
    const double total = total_amount(phases);
    for (std::size_t i = 0; i < phases.size(); ++i)
      if (phases[i].amount > engine.settings.amount_tolerance * total)
        groups[phases[i].candidate_index].push_back(i);
    return groups;
  };
  auto first = indices(left_phases), second = indices(right_phases);
  for (const auto &[candidate, one] : first) {
    const auto &two = second[candidate];
    if (two.empty() || one.size() == two.size())
      continue;
    auto &smaller = one.size() < two.size() ? left_phases : right_phases;
    const auto &larger = one.size() < two.size() ? right_phases : left_phases;
    const auto &small = one.size() < two.size() ? one : two;
    const auto &large = one.size() < two.size() ? two : one;
    std::vector<std::size_t> order(large.size()), best;
    for (std::size_t i = 0; i < order.size(); ++i)
      order[i] = i;
    double minimum = std::numeric_limits<double>::infinity();
    do {
      double distance = 0.;
      for (std::size_t i = 0; i < small.size(); ++i)
        distance += (smaller[small[i]].composition -
                     larger[large[order[i]]].composition)
                        .squaredNorm();
      if (distance < minimum) {
        minimum = distance;
        best = order;
      }
    } while (std::next_permutation(order.begin(), order.end()));
    for (std::size_t i = 0; i < small.size(); ++i)
      smaller[small[i]].id = larger[large[best[i]]].id;
  }
  auto ids = [&](const PhaseStates &phases) {
    std::vector<int> values;
    for (const auto &[candidate, group] : indices(phases))
      for (auto index : group)
        values.push_back(phases[index].id);
    std::sort(values.begin(), values.end());
    return values;
  };
  auto a = ids(left_phases), b = ids(right_phases);
  if (a == b)
    return;
  std::vector<int> joined, changed;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(joined));
  std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                std::back_inserter(changed));
  // Two phases entering/leaving need separate zero-amount boundaries too.
  // A direct solve on their union can find only the outer boundary and miss
  // the intervening field. A one-in/one-out exchange can share one line.
  if ((changed.size() > 2 || (changed.size() == 2 && a.size() != b.size())) &&
      depth < 20 &&
      (normalise(left.state) - normalise(right.state)).norm() >
          engine.settings.min_step * 2.) {
    auto u = (.5 * (normalise(left.state) + normalise(right.state))).eval();
    auto actual = (origin + u.cwiseProduct(range)).eval();
    auto mid = engine.stable_at(actual, &left, &right);
    add_sample(mid.state);
    bracket(left, mid, depth + 1);
    bracket(mid, right, depth + 1);
    return;
  }
  PhaseStates states, other;
  for (const auto &[candidate, group] : first)
    for (auto index : group)
      states.push_back(left_phases[index]);
  for (const auto &[candidate, group] : second)
    for (auto index : group)
      other.push_back(right_phases[index]);
  for (auto &s : other) {
    auto it = find_phase(states, s.id);
    if (it == states.end())
      states.push_back(s);
    else {
      it->composition = .5 * (it->composition + s.composition);
      it->amount = .5 * (it->amount + s.amount);
    }
  }
  auto u = (.5 * (normalise(left.state) + normalise(right.state))).eval();
  auto direction =
      (normalise(right.state) - normalise(left.state)).normalized().eval();
  for (int z : changed) {
    queue.push_back({joined, z, -1, states, u,
                     Eigen::Vector2d(-direction[1], direction[0])});
    queue.back().physical = Eigen::Vector2d(
        .5 * (left.state.pressure + right.state.pressure),
        .5 * (left.state.temperature + right.state.temperature));
  }
}

int Tracer::endpoint(Assemblage &a, const std::vector<int> &ids, int z,
                     int other, const PhaseStates &previous_phases,
                     const Eigen::Vector2d &previous, double maximum_distance,
                     double &accepted_affinity, EquilibriumSolve *solution,
                     const Eigen::Vector2d &direction) {
  auto prm = engine.parameters(a);
  auto zi = std::find(ids.begin(), ids.end(), z) - ids.begin(),
       oi = std::find(ids.begin(), ids.end(), other) - ids.begin();
  auto c = constraints(std::make_unique<PhaseFractionConstraint>(zi, 0., prm),
                       std::make_unique<PhaseFractionConstraint>(oi, 0., prm));
  auto s = engine.solve(a, c);
  bool verified = valid(a, s), separate = engine.distinct(a, ids);
  std::vector<Minimum> minima;
  double affinity = verified ? engine.stability(a, &minima) : 0.;
  if (verified && separate && needs_face_refit(affinity) &&
      engine.settings.active_solution_faces) {
    // A new trace site can become active at a double-zero junction too.
    // Preserve both phase-fraction constraints while releasing that face.
    auto refitted = engine.copy_assemblage(a);
    if (refit_composition_faces(refitted, ids, z, previous,
                                Eigen::Vector2d::Zero(), s, affinity, &minima,
                                1.e-7, other)) {
      a = *refitted;
      engine.set_coordinate(a, engine.composition_coordinate(*refitted));
      verified = valid(a, s);
      separate = engine.distinct(a, ids);
    }
  }
  const auto offset = (normalised_coordinates(a) - previous).eval();
  double displacement = offset.norm();
  bool singular_copy_event = false;
  if (verified && separate &&
      near_third_copy_event(ids, z, other, engine.snapshot(a, ids))) {
    Eigen::MatrixXd j =
        s.J * engine.parameter_scales(a, s.J.cols(), range).asDiagonal();
    for (Eigen::Index row = 0; row < j.rows(); ++row)
      if (j.row(row).norm() > 0.)
        j.row(row) /= j.row(row).norm();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(j);
    singular_copy_event =
        (svd.singularValues().array() > 1.e-9).count() < j.cols();
  }
  if (!verified || !separate || singular_copy_event ||
      !same_solution_branches(engine, previous_phases, engine.snapshot(a, ids),
                              a.get_pressure(), a.get_temperature()) ||
      affinity < -engine.settings.affinity_tolerance ||
      displacement > maximum_distance ||
      offset.dot(direction) <
          -engine.settings.node_tolerance * direction.norm()) {
    if (engine.settings.verbose)
      std::cerr << "Rejected junction zeros=" << z << ',' << other
                << " verified=" << verified << " distinct=" << separate
                << " isolated=" << !singular_copy_event
                << " affinity=" << affinity << " displacement=" << displacement
                << " maximum=" << maximum_distance << " P=" << a.get_pressure()
                << " T=" << a.get_temperature() << ": " << s.message << '\n';
    return -1;
  }
  if (solution)
    *solution = s;
  accepted_affinity = affinity;
  return node(a, ids, {z, other}, "junction", &s.J);
}

std::shared_ptr<Assemblage>
Tracer::phase_entry(const Assemblage &previous, const Assemblage &next,
                    const std::vector<int> &ids, int zero, int added,
                    const Minimum &entering, const Eigen::Vector2d &unit,
                    double maximum_distance, double &accepted_affinity,
                    int &index, EquilibriumSolve &solution) {
  auto original = engine.snapshot(previous, ids);
  auto previous_position = normalised_coordinates(previous);
  const int base = added / engine.settings.max_phase_instances;
  struct Trial {
    std::shared_ptr<Assemblage> a;
    Minimum minimum;
    Eigen::Vector2d u;
  };
  auto evaluate = [&](const Eigen::Vector2d &u, const PhaseStates &warm,
                      Trial &trial) {
    if (!inside(u))
      return false;
    auto location = (origin + u.cwiseProduct(range)).eval();
    try {
      trial.a = engine.make_at(
          ids, warm, location,
          Eigen::Vector2d(previous.get_pressure(), previous.get_temperature()));

      auto s = solve_boundary(*trial.a, ids, zero, u, unit);
      if (!valid(*trial.a, s) || !engine.distinct(*trial.a, ids) ||
          !same_branch(previous, *trial.a, ids))
        return false;
      engine.set_pt(trial.a->get_pressure(), trial.a->get_temperature());
      trial.minimum =
          engine.minimize(base, engine.potentials(*trial.a), entering.p);
      trial.u = normalised_coordinates(*trial.a);
      return true;
    } catch (const std::exception &) {
      return false;
    }
  };
  Trial left, right;
  if (!evaluate(previous_position, original, left) ||
      !evaluate(normalised_coordinates(next), engine.snapshot(next, ids),
                right) ||
      right.minimum.affinity >= 0.)
    return {};
  // An accepted point can have a small negative affinity within tolerance.
  // Step back on the same branch to obtain a genuine sign-changing bracket.
  if (left.minimum.affinity < 0.) {
    double distance =
        std::max((right.u - left.u).norm(), engine.settings.min_step);
    for (int k = 0; k < 12 && left.minimum.affinity < 0.; ++k) {
      Trial earlier;
      if (distance > maximum_distance ||
          !evaluate(previous_position - distance * unit, original, earlier))
        break;
      left = std::move(earlier);
      distance *= 2.;
    }
  }
  if (left.minimum.affinity < 0.)
    return {};
  Trial best =
      std::abs(left.minimum.affinity) < std::abs(right.minimum.affinity)
          ? left
          : right;
  for (int k = 0; k < 60 && std::abs(best.minimum.affinity) > 1.e-7; ++k) {
    double fraction = left.minimum.affinity /
                      (left.minimum.affinity - right.minimum.affinity);
    fraction = std::clamp(fraction, .1, .9);
    Trial trial;
    auto u = (left.u + fraction * (right.u - left.u)).eval();
    auto warm = engine.snapshot(*(fraction < .5 ? left.a : right.a), ids);
    if (!evaluate(u, warm, trial))
      return {};
    if (std::abs(trial.minimum.affinity) < std::abs(best.minimum.affinity))
      best = trial;
    if (trial.minimum.affinity >= 0.)
      left = std::move(trial);
    else
      right = std::move(trial);
    if ((right.u - left.u).norm() < 1.e-11)
      break;
  }
  auto states = engine.snapshot(*best.a, ids);
  PhaseState fresh;
  fresh.id = added;
  fresh.candidate_index = base;
  fresh.composition = best.minimum.p;
  fresh.amount = 0.;
  states.push_back(fresh);
  auto joined = ids;
  joined.push_back(added);
  std::sort(joined.begin(), joined.end());
  try {
    auto junction = engine.make_assemblage(
        joined, states, best.a->get_pressure(), best.a->get_temperature());
    engine.set_coordinate(*junction, engine.composition_coordinate(*best.a));
    index = endpoint(*junction, joined, zero, added, states, previous_position,
                     maximum_distance, accepted_affinity, &solution);
    if (index >= 0)
      return junction;
  } catch (const std::exception &error) {
    if (engine.settings.verbose)
      std::cerr << "Bracketed phase entry " << added << ": " << error.what()
                << '\n';
  }
  return {};
}

bool Tracer::refit_composition_faces(
    std::shared_ptr<Assemblage> &input, const std::vector<int> &ids, int zero,
    const Eigen::Vector2d &where, const Eigen::Vector2d &normal,
    EquilibriumSolve &solved, double &accepted_affinity,
    std::vector<Minimum> *minima, double face_tolerance, int second_zero) {
  if (!engine.settings.active_solution_faces)
    return false;
  auto current = input;
  std::vector<Minimum> candidates = minima ? *minima : std::vector<Minimum>();
  for (int pass = 0; pass < 3; ++pass)
    try {
      auto states = engine.snapshot(*current, ids);
      for (std::size_t k = 0; k < ids.size(); ++k) {
        int base = states[k].candidate_index;
        auto basis = engine.composition_basis(*current->get_phase(k), base);
        if (basis.rows() == basis.cols() || candidates.empty() ||
            !needs_face_refit(
                candidates[static_cast<std::size_t>(base)].affinity, true))
          continue;
        auto minimum = candidates[static_cast<std::size_t>(base)];
        if ((minimum.p - states[k].composition).norm() >= .05)
          minimum = engine.minimize(base, engine.potentials(*current),
                                    states[k].composition);
        const Eigen::VectorXd weights =
            basis.transpose().completeOrthogonalDecomposition().solve(
                minimum.p);
        // Releasing a face requires a resolved population outside that
        // face. A nearby minimum within it only changes the warm seed and
        // can switch branches or repeatedly freeze numerical trace sites.
        if (needs_face_refit(minimum.affinity, true) &&
            (basis.transpose() * weights - minimum.p).norm() > 1.e-10 &&
            (minimum.p - states[k].composition).norm() < .05) {
          states[k].composition = minimum.p;
          face_tolerance = std::min(face_tolerance, 1.e-9);
        }
      }
      // Infer the new face while retaining positive trace populations.
      // Freezing and releasing the same population changes its chemical
      // potential and can mimic an additional solution copy.
      for (auto &state : states)
        state.composition_basis.reset();
      auto trial = engine.make_assemblage(ids, states, current->get_pressure(),
                                          current->get_temperature(),
                                          face_tolerance, true);
      engine.set_coordinate(*trial, engine.composition_coordinate(*current));
      bool changed = false;
      for (std::size_t k = 0; k < ids.size(); ++k) {
        auto old = engine.composition_basis(*current->get_phase(k),
                                            states[k].candidate_index),
             next = engine.composition_basis(*trial->get_phase(k),
                                             states[k].candidate_index);
        changed = changed || old.rows() != next.rows() ||
                  (old.rows() == next.rows() && (old - next).norm() > 1.e-8);
      }
      if (!changed)
        return false;

      ConstraintList c;
      if (second_zero >= 0) {
        auto prm = engine.parameters(*trial);
        c = constraints(
            std::make_unique<PhaseFractionConstraint>(
                std::find(ids.begin(), ids.end(), zero) - ids.begin(), 0., prm),
            std::make_unique<PhaseFractionConstraint>(
                std::find(ids.begin(), ids.end(), second_zero) - ids.begin(),
                0., prm));
      } else
        c = boundary_constraints(*trial, ids, zero, where, normal);
      auto solve = engine.solve(*trial, c);
      if (!valid(*trial, solve) || !engine.distinct(*trial, ids) ||
          !same_branch(*input, *trial, ids) ||
          (normalised_coordinates(*trial) - where).norm() >
              engine.settings.step * 3.)
        return false;
      double affinity = engine.stability(*trial, &candidates);
      if (affinity >= -engine.settings.affinity_tolerance) {
        input = trial;
        solved = solve;
        accepted_affinity = affinity;
        if (minima)
          *minima = candidates;
        return true;
      }
      current = trial;
    } catch (const std::exception &) {
      return false;
    }
  return false;
}

std::vector<BoundaryPoint>
Tracer::follow(std::shared_ptr<Assemblage> a, const Seed &seed,
               EquilibriumSolve solve, Eigen::VectorXd direction, int &end_node,
               std::string &termination) {
  // An unsuccessful endpoint must not inherit the seed's junction ID: that
  // would snap an unfinished curve back to its starting node when plotting.
  end_node = -1;
  std::vector<BoundaryPoint> points;
  double step = engine.settings.step;
  auto ids = seed.ids;
  double travelled = 0.;
  Eigen::Vector2d initial_unit =
      engine.project_direction(direction, *a).cwiseQuotient(range);
  for (int iteration = 0; iteration < engine.settings.max_trace_steps;
       ++iteration) {
    auto old_x = solve.x;
    auto old_u = normalised_coordinates(*a);

    const double maximum_step =
        line_resolution
            ? refinement_step(result, *line_resolution, engine.coordinates(*a),
                              engine.project_direction(direction, *a),
                              engine.settings)
            : engine.settings.step;
    double requested = std::min(step, maximum_step);
    double affinity = 0.;
    // Predict the first phase-out event from amount derivatives.
    auto prm = engine.parameters(*a);
    // Chemical-potential differences lose accuracy as solvus copies merge.
    // Locate the intrinsic critical point before a nearly identical-copy
    // root can send the ordinary corrector beyond it and double back.
    if (closing_solvus(*a, ids, seed.zero, direction, prm)) {
      auto critical = engine.copy_assemblage(*a);
      auto critical_solve = solve;
      std::vector<BoundaryPoint> end;
      if (approach_critical(critical, ids, seed.zero, critical_solve, end) &&
          critical_axis.size() &&
          (normalised_coordinates(*critical) - old_u).norm() <=
              engine.settings.step * 3.) {
        points.insert(points.end(), end.begin(), end.end());
        end_node = node(*critical, ids, {seed.zero}, "critical_point",
                        &critical_solve.J);
        termination = "solution critical point";
        return points;
      }
    }
    int leaving = -1;
    double event_step = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < ids.size(); ++k)
      if (ids[k] != seed.zero) {
        int j = prm.phase_amount_indices[static_cast<Eigen::Index>(k)];
        if (direction[j] < -1.e-12) {
          double distance = -old_x[j] / direction[j];
          if (distance >= -engine.settings.min_step && distance < event_step) {
            event_step = std::max(0., distance);
            leaving = ids[k];
          }
        }
      }
    if (leaving >= 0 && event_step < requested * 1.1) {
      auto event_a = engine.copy_assemblage(*a);
      try {
        auto delta = (std::max(0., event_step) * direction).eval();
        if (old_x[0] + delta[0] >= 0 && old_x[1] + delta[1] > 0)
          predict(*event_a, old_x, delta, prm);
        EquilibriumSolve event_s;
        int n = endpoint(
            *event_a, ids, seed.zero, leaving, engine.snapshot(*a, ids), old_u,
            std::max(requested * 3., engine.settings.node_tolerance * 2.),
            affinity, &event_s,
            engine.project_direction(direction, *a).cwiseQuotient(range));
        if (n >= 0) {
          end_node = n;
          points.push_back(point(*event_a, ids, event_s, affinity));
          termination = "junction";
          return points;
        }
      } catch (const std::exception &) {
      }
      if (event_step < engine.settings.min_step) {
        // A nearly inactive composition coordinate can make the projected
        // amount derivatives singular. Verify a neighbouring face before
        // interpreting that derivative as an instantaneous phase-out event.
        Eigen::Vector2d event_unit =
            engine.project_direction(direction, *a).cwiseQuotient(range);
        if (refit_composition_faces(a, ids, seed.zero, old_u, event_unit, solve,
                                    affinity, nullptr, 5.e-7)) {
          auto refitted = tangent(solve, *a);
          if (refitted.size()) {
            if (engine.project_direction(refitted, *a)
                    .cwiseQuotient(range)
                    .dot(event_unit) < 0.)
              refitted = -refitted;
            direction = refitted;
            continue;
          }
        }
        termination = "unresolved phase-out event";
        return points;
      }
      requested = std::min(requested, event_step * .7);
    }
    auto unit =
        engine.project_direction(direction, *a).cwiseQuotient(range).eval();
    int border_axis = -1;
    double border = 0., border_step = requested;
    for (int k = 0; k < 2; ++k)
      if (std::abs(unit[k]) > 1.e-12) {
        double edge = unit[k] > 0 ? 1. : 0.,
               distance = (edge - old_u[k]) / unit[k];
        if (distance >= -1.e-8 && distance <= border_step) {
          border_step = std::max(0., distance);
          border_axis = k;
          border = edge;
        }
      }
    std::shared_ptr<Assemblage> next;
    EquilibriumSolve next_s;
    bool accepted = false;
    bool stability_verified = false;
    double used = border_axis >= 0 ? border_step : requested;
    std::string failure;
    bool merged = false;
    std::vector<Minimum> ms;
    for (int retry = 0; retry < 16; ++retry) {
      if (used < engine.settings.min_step && border_axis < 0)
        break;
      next = engine.copy_assemblage(*a);
      try {
        predict(*next, old_x, used * direction, prm);
        if (!engine.free_vectors.empty() && border_axis == 1 &&
            (border == 0. || border == 1.))
          next = engine.make_at(ids, engine.snapshot(*next, ids),
                                engine.coordinates(*next));
        ConstraintList c;

        if (border_axis >= 0) {
          Eigen::Vector2d n = Eigen::Vector2d::Zero(), u = old_u;
          n[border_axis] = 1;
          u[border_axis] = border;
          c = boundary_constraints(*next, ids, seed.zero, u, n);
        } else
          c = boundary_constraints(*next, ids, seed.zero, old_u + used * unit,
                                   unit);
        next_s = engine.solve(*next, c);
        merged = !engine.distinct(*next, ids) || !same_branch(*a, *next, ids);
        accepted = valid(*next, next_s) && !merged &&
                   (normalised_coordinates(*next) - old_u).norm() <=
                       std::max(used * 3, engine.settings.node_tolerance);
        failure = next_s.message;
        if (accepted) {
          // A required competitor can leave its EOS domain while all
          // phases on this line still have valid volumes. Check before
          // accepting the continuation step so the curve approaches the
          // model limit instead of losing all its previously traced points.
          affinity = engine.stability(*next, &ms);
          stability_verified = true;
          break;
        }
      } catch (const std::exception &error) {
        accepted = false;
        failure = error.what();
      }
      used *= .5;
      border_axis = -1;
    }
    if (!accepted && next) {
      auto target =
          (old_u + std::max(used, engine.settings.min_step) * unit).eval();
      if (refit_composition_faces(next, ids, seed.zero, target, unit, next_s,
                                  affinity, &ms) &&
          same_branch(*a, *next, ids)) {
        accepted = true;
        stability_verified = true;
        border_axis = -1;
      }
    }
    if (!accepted) {
      if (merged && solution_separation(engine.snapshot(*a, ids)) < .01 &&
          approach_critical(a, ids, seed.zero, solve, points)) {
        end_node = node(*a, ids, {seed.zero}, "critical_point", &solve.J);
        termination = "solution critical point";
      } else
        termination = "corrector failed";
      if (engine.settings.verbose)
        std::cerr << "Corrector stopped at P=" << a->get_pressure()
                  << ", T=" << a->get_temperature() << ": " << failure << '\n';
      return points;
    }
    if (!stability_verified)
      affinity = engine.stability(*next, &ms);
    if (engine.settings.active_solution_faces) {
      Eigen::Vector2d target = old_u + used * unit, normal = unit;
      if (border_axis >= 0) {
        target[border_axis] = border;
        normal.setZero();
        normal[border_axis] = 1.;
      }
      refit_composition_faces(next, ids, seed.zero, target, normal, next_s,
                              affinity, &ms);
    }
    const auto current_states = engine.snapshot(*next, ids);
    // Locate a distinct solvus minimum before its branch loses stability,
    // including weak P-T solvi inside the ordinary discovery tolerance.
    bool entering = false;
    for (std::size_t base = 0; base < ms.size(); ++base)
      entering = entering ||
                 ms[base].affinity <
                     -phase_entry_tolerance(base, current_states, ms[base]);
    if (entering) {
      // A previously absent phase becomes stable: add it at zero amount and
      // solve the two zero constraints to locate the node, then spawn
      // branches.
      bool located = false;
      for (std::size_t base = 0; base < ms.size(); ++base)
        if (ms[base].affinity <
            -phase_entry_tolerance(base, current_states, ms[base])) {
          int added = burnman::utils::checked_int(
              base *
              static_cast<std::size_t>(engine.settings.max_phase_instances));
          while (std::find(ids.begin(), ids.end(), added) != ids.end() &&
                 added < (static_cast<int>(base) + 1) *
                             engine.settings.max_phase_instances)
            ++added;
          if (added >= (static_cast<int>(base) + 1) *
                           engine.settings.max_phase_instances)
            continue;
          auto joined = ids;
          joined.push_back(added);
          std::sort(joined.begin(), joined.end());
          auto states = engine.snapshot(*a, ids);
          PhaseState fresh;
          fresh.id = added;
          fresh.candidate_index = burnman::utils::checked_int(base);
          fresh.composition = ms[base].p;
          fresh.amount = 0.;
          states.push_back(fresh);
          // The accepted state may already be slightly beyond zero affinity:
          // stability permits a finite numerical tolerance. Halving the step
          // must not shrink the endpoint acceptance radius below node
          // tolerance.
          try {
            auto junction = engine.make_assemblage(
                joined, states, a->get_pressure(), a->get_temperature());
            engine.set_coordinate(*junction, engine.composition_coordinate(*a));
            EquilibriumSolve event_s;
            double event_affinity = 0.;
            int n = endpoint(
                *junction, joined, seed.zero, added, states, old_u,
                std::max((engine.direct_coordinates() ? used : requested) * 3.,
                         engine.settings.node_tolerance * 2.),
                event_affinity, &event_s);
            if (n >= 0) {
              auto p = point(*junction, joined, event_s, event_affinity);
              points.push_back(p);
              end_node = n;
              located = true;
              break;
            }
          } catch (const std::exception &error) {
            if (engine.settings.verbose)
              std::cerr << "Junction candidate " << added << " on zero "
                        << seed.zero << " failed: " << error.what() << '\n';
          }
          if (!located) {
            int n = -1;
            EquilibriumSolve event_s;
            double event_affinity = 0.;
            auto junction = phase_entry(
                *a, *next, ids, seed.zero, added, ms[base], unit,
                std::max((engine.direct_coordinates() ? used : requested) * 3.,
                         engine.settings.node_tolerance * 2.),
                event_affinity, n, event_s);
            if (junction) {
              points.push_back(
                  point(*junction, joined, event_s, event_affinity));
              end_node = n;
              located = true;
              break;
            }
          }
        }

      if (located) {
        termination = "junction";
        return points;
      }
      step = used * .5;
      if (step < engine.settings.min_step) {
        termination = "unresolved phase-in or solvus event";
        return points;
      }
      continue;
    }
    auto u = normalised_coordinates(*next);
    travelled += (u - old_u).norm();
    points.push_back(point(*next, ids, next_s, affinity));
    if (border_axis >= 0) {
      end_node = node(*next, ids, {seed.zero}, "domain_edge", &next_s.J);
      termination = "domain edge";
      return points;
    }
    // A small step near a solvus is not a closed loop. Require an actual
    // return segment, consistent compositions and an aligned outgoing
    // tangent.
    auto displacement = (u - old_u).eval();
    double projection =
        (displacement.squaredNorm() != 0.)
            ? std::clamp((seed.position - old_u).dot(displacement) /
                             displacement.squaredNorm(),
                         0., 1.)
            : 0.;
    bool return_segment =
        (old_u + projection * displacement - seed.position).norm() <
        engine.settings.node_tolerance;
    bool same_compositions = true;
    auto now = engine.snapshot(*next, ids);
    for (std::size_t i = 0; i < ids.size(); ++i)
      if ((now[i].composition - seed.phases[i].composition).norm() >
          engine.settings.composition_tolerance * 100.)
        same_compositions = false;
    if (seed.node == -1 && points.size() > 6 &&
        travelled > engine.settings.step * 3. && return_segment &&
        same_compositions && unit.dot(initial_unit) > .5) {
      termination = "closed loop";
      return points;
    }
    auto new_direction = tangent(next_s, *next);
    if (!new_direction.size()) {
      bool critical = solution_separation(engine.snapshot(*next, ids)) < .01;
      if (critical)
        approach_critical(next, ids, seed.zero, next_s, points);
      end_node =
          node(*next, ids, {seed.zero},
               critical ? "critical_point" : "reduced_variance", &next_s.J);
      termination = critical ? "solution critical point" : "reduced variance";
      return points;
    }
    if (engine.project_direction(new_direction, *next)
            .cwiseQuotient(range)
            .dot(unit) < 0)
      new_direction = -new_direction;
    a = next;
    solve = next_s;
    direction = new_direction;
    step = std::min(engine.settings.step, used * 1.4);
  }
  termination = "trace step limit";
  return points;
}

void Tracer::label(Boundary &line, bool response_only) {
  if (line.points.size() < 2 || lies_on_calculation_frame(line, result))
    return;
  double length = 0.;
  for (std::size_t k = 1; k < line.points.size(); ++k)
    length +=
        (normalise(line.points[k]) - normalise(line.points[k - 1])).norm();
  // Label away from nodes, where a normal perturbation crosses only this
  // edge. At a junction the same perturbation can cross several fields.
  // Choose the geometric midpoint and correct it onto the edge. On short
  // branches the middle stored point can still be the starting junction.
  double walked = 0.;
  std::size_t segment = 1;
  for (; segment + 1 < line.points.size(); ++segment) {
    double span =
        (normalise(line.points[segment]) - normalise(line.points[segment - 1]))
            .norm();
    if (walked + span >= length * .5)
      break;
    walked += span;
  }
  auto left = normalise(line.points[segment - 1]);
  auto right = normalise(line.points[segment]);
  auto along = (right - left).eval();
  double span = along.norm();
  Eigen::Vector2d centre =
      span > 0
          ? left + std::clamp((length * .5 - walked) / span, 0., 1.) * along
          : left;
  if (span < 1.e-10)
    along = Eigen::Vector2d(1., 0.);
  else
    along /= span;
  auto location = (origin + centre.cwiseProduct(range)).eval();
  BoundaryPoint anchor;
  std::array<std::vector<int>, 2> neighbours;
  for (int attempt = 0; attempt < 3; ++attempt)
    try {
      auto states =
          line.points[segment - 1 + static_cast<std::size_t>(attempt == 1)]
              .phases;
      if (attempt == 2)
        states = average_phase_states(std::move(states),
                                      line.points[segment].phases);
      const auto &warm =
          line.points[segment - 1 + static_cast<std::size_t>(attempt == 1)];
      auto a = engine.make_at(line.assemblage, states, location,
                              Eigen::Vector2d(warm.pressure, warm.temperature));

      auto s =
          solve_boundary(*a, line.assemblage, line.zero_phase, centre, along);
      if (!valid(*a, s) || !engine.distinct(*a, line.assemblage))
        continue;
      std::vector<Minimum> minima;
      double affinity = engine.stability(*a, &minima);
      if (needs_face_refit(affinity))
        refit_composition_faces(a, line.assemblage, line.zero_phase, centre,
                                along, s, affinity, &minima);
      if (affinity < -engine.settings.affinity_tolerance ||
          (normalised_coordinates(*a) - centre).norm() > span)
        continue;
      anchor = point(*a, line.assemblage, s, affinity);
      centre = normalise(anchor);
      const auto direction = tangent(s, *a);
      if (direction.size()) {
        auto unit = engine.project_direction(direction, *a)
                        .cwiseQuotient(range)
                        .normalized()
                        .eval();
        along = unit.dot(along) < 0. ? -unit : unit;
      }
      // At a regular phase-out arc, differentiate the actual equilibrium
      // rather than probing across an arbitrarily wide neighbouring field.
      std::array<State, 2> limiting_states;
      neighbours = boundary_sides(
          engine, *a, line.assemblage, line.zero_phase, s, range,
          Eigen::Vector2d(-along[1], along[0]), &limiting_states);
      if (!neighbours[0].empty() && !neighbours[1].empty()) {
        line.side_a = neighbours[0];
        line.side_b = neighbours[1];
        line.is_solution_replacement = line.side_a == line.side_b;
        for (auto &state : limiting_states)
          if (state.success) {
            state.minimum_affinity = affinity;
            add_sample(state);
          }
        if (engine.settings.verbose)
          std::cerr << "Boundary " << line.id
                    << ": side assemblages from equilibrium response.\n";
        return;
      }
      break;
    } catch (const std::exception &) {
    }
  if (response_only)
    return;
  // An endpoint can lie on several phase lines at once. Keep the midpoint
  // when no interior corrector verifies, using its interpolated warm start.
  if (anchor.phases.empty()) {
    anchor = line.points[segment - 1];
    for (auto &state : anchor.phases)
      for (const auto &other : line.points[segment].phases)
        if (state.id == other.id) {
          state.composition = .5 * (state.composition + other.composition);
          state.amount = .5 * (state.amount + other.amount);
        }
  }
  Eigen::Vector2d side(-along[1], along[0]);
  std::vector<std::vector<int>> side_assemblages = {line.assemblage};
  auto absent = line.assemblage;
  absent.erase(std::find(absent.begin(), absent.end(), line.zero_phase));
  if (!absent.empty())
    side_assemblages.push_back(absent);
  // At a phase replacement, two stoichiometrically indistinguishable
  // amounts are interchangeable at coexistence. Try the opposite removal.
  const auto formulae = engine.phase_formulae(line.assemblage, anchor.phases);
  const bool observed_full_assemblage =
      field_index.count(engine.canonical_ids(line.assemblage)) != 0;
  Eigen::FullPivLU<Eigen::MatrixXd> variance(formulae);
  variance.setThreshold(1.e-9);
  if (variance.rank() < formulae.rows())
    for (int removed : line.assemblage)
      if (removed != line.zero_phase) {
        auto subset = line.assemblage;
        subset.erase(std::find(subset.begin(), subset.end(), removed));
        if (!subset.empty())
          side_assemblages.push_back(subset);
      }
  std::array<State, 2> side_states;
  std::array<State, 2> previous_sides;
  std::array<WorkState, 2> warm_sides;
  line.is_solution_replacement = false;
  const int base = line.zero_phase / engine.settings.max_phase_instances;
  const bool replaces_solution =
      std::count_if(line.assemblage.begin(), line.assemblage.end(),
                    [&](int id) {
                      return id / engine.settings.max_phase_instances == base;
                    }) > 1;
  auto branches = [&](const State &state) {
    std::vector<int> ids;
    const double total = total_amount(state.phases);
    double separation = std::numeric_limits<double>::infinity();
    for (auto &first : anchor.phases)
      for (auto &second : anchor.phases)
        if (first.candidate_index == base && second.candidate_index == base &&
            first.id != second.id)
          separation = std::min(
              separation, (first.composition - second.composition).norm());
    if (!std::isfinite(separation))
      return ids;
    for (auto &phase : state.phases)
      if (phase.candidate_index == base &&
          phase.amount > engine.settings.amount_tolerance * total) {
        int closest = -1;
        double distance = separation * .25;
        for (auto &original : anchor.phases)
          if (original.candidate_index == base &&
              (phase.composition - original.composition).norm() < distance) {
            closest = original.id;
            distance = (phase.composition - original.composition).norm();
          }
        if (closest < 0)
          return std::vector<int>{};
        ids.push_back(closest);
      }
    std::sort(ids.begin(), ids.end());
    return ids;
  };
  auto different_solution_branches = [&](const State &first,
                                         const State &second) {
    auto a = branches(first), b = branches(second);
    return !a.empty() && !b.empty() && a != b;
  };
  auto immediate_neighbour = [&](const State &state) {
    const auto ids = engine.active_phases(state);
    // Finite equality tolerances can accept a probe still on a phase line.
    // A PT field needs two Gibbs degrees of freedom, including both P and T.
    if (result.section.type == DiagramType::PT &&
        ids.size() > engine.components.size())
      return false;
    // For an observed full assemblage, independent phase amounts cannot
    // exchange one zero phase for another across this edge. Such a probe
    // has skipped the narrow full assemblage.
    // Dependent amounts retain polymorph and compound-reaction replacements.
    if (observed_full_assemblage &&
        !has_composition_axis(result.section.type) &&
        variance.rank() == formulae.rows())
      return ids == engine.canonical_ids(line.assemblage) ||
             ids == engine.canonical_ids(absent);
    return has_composition_axis(result.section.type) ||
           (std::includes(line.assemblage.begin(), line.assemblage.end(),
                          ids.begin(), ids.end()) &&
            !(replaces_solution && variance.rank() + 1 == formulae.rows() &&
              ids.size() + 1 < line.assemblage.size()));
  };
  double next_offset = std::min(
      std::max(engine.settings.node_tolerance * 4, 1.e-3), length * .125);
  bool expanded_for_amount = false;
  auto distinguishes_sides = [&]() {
    if (!side_states[0].success || !side_states[1].success)
      return false;
    for (std::size_t slot = 0; slot < side_states.size(); ++slot)
      if (!neighbours[slot].empty() &&
          engine.active_phases(side_states[slot]) != neighbours[slot])
        return false;
    return engine.active_phases(side_states[0]) !=
           engine.active_phases(side_states[1]);
  };
  const auto probe_accuracy =
      equilibrium_coordinate_tolerance(result.section.type, range);
  for (int retry = 0; retry < 24; ++retry) {
    const double offset = next_offset;
    next_offset *= .5;
    // Below native coordinate accuracy these are effectively the same
    // equilibrium, even when floating-point coordinates still differ.
    const Eigen::Vector2d displacement =
        offset * side.cwiseAbs().cwiseProduct(range);
    if (result.section.type == DiagramType::PT &&
        (displacement.array() <= probe_accuracy.array()).all()) {
      if (engine.settings.verbose)
        std::cerr << "Boundary " << line.id
                  << ": neighbour probes reached native coordinate accuracy.\n";
      break;
    }
    std::array<std::optional<State>, 2> discovered;
    auto discover = [&](std::size_t slot, const Eigen::Vector2d &actual) {
      // Inverse-section discovery can change the unplotted P/T and active
      // faces between attempts. Retain its independent constrained searches;
      // the neighbouring-state optimisation here is for P-T probes.
      if (result.section.type != DiagramType::PT)
        return engine.stable_at(actual).state;
      if (!discovered[slot]) {
        auto work = engine.stable_at(actual, warm_sides[slot].state.success
                                                 ? &warm_sides[slot]
                                                 : nullptr);
        discovered[slot] = work.state;
        if (work.state.success)
          warm_sides[slot] = std::move(work);
      }
      return *discovered[slot];
    };
    double unresolved_fraction = 0.;
    for (int sign : {-1, 1}) {
      auto u = (centre + sign * offset * side).eval();
      if (!inside(u))
        continue;
      int slot = sign < 0 ? 0 : 1;
      side_states[static_cast<std::size_t>(slot)] = State{};
      auto actual = (origin + u.cwiseProduct(range)).eval();
      for (auto &subset : side_assemblages)
        try {
          auto work = engine.fixed_at(
              subset, anchor.phases, actual,
              Eigen::Vector2d(anchor.pressure, anchor.temperature));
          // A real entering phase can still lie below the reporting cutoff
          // on this probe. Shrinking then makes both labels identical
          // forever. Retain its unrounded fraction to choose one more distant
          // probe.
          if (!has_composition_axis(result.section.type) &&
              !replaces_solution && variance.rank() == formulae.rows() &&
              subset == line.assemblage && work.state.success) {
            const double total = total_amount(work.state.phases);
            for (const auto &phase : work.state.phases)
              if (phase.id == line.zero_phase && phase.amount > 0. &&
                  phase.amount <= engine.settings.amount_tolerance * total)
                unresolved_fraction =
                    std::max(unresolved_fraction, phase.amount / total);
          }
          // A finite-tolerance solve can retain a zero-amount boundary
          // phase. Remove its chemical equations and reverify the actual
          // field before using its compositions to identify a replacement.
          while (work.state.success &&
                 engine.active_phases(work.state).size() < work.ids.size()) {
            auto surviving = present_phase_ids(
                work.state.phases, engine.settings.amount_tolerance);
            work = engine.fixed_at(
                surviving, work.state.phases, actual,
                Eigen::Vector2d(work.state.pressure, work.state.temperature));
          }
          if (!work.state.success ||
              !engine.distinct(*work.assemblage, work.ids) ||
              (result.section.type == DiagramType::PT &&
               field_variance(*work.assemblage,
                              engine.active_phases(work.state).size()) < 2) ||
              !immediate_neighbour(work.state))
            continue;
          work.state.message = "Stable neighbouring field";
          side_states[static_cast<std::size_t>(slot)] = work.state;
          warm_sides[static_cast<std::size_t>(slot)] = std::move(work);
          break;
        } catch (const std::exception &) {
        }
      if (!side_states[static_cast<std::size_t>(slot)].success)
        side_states[static_cast<std::size_t>(slot)] =
            discover(static_cast<std::size_t>(slot), actual);
    }
    // Grow only for measured positive phase entry, bounded away from the
    // line's endpoints. The usual immediate-neighbour checks still reject
    // a probe that skips another phase field; subsequent retries shrink it.
    if (!expanded_for_amount && unresolved_fraction > 0. &&
        side_states[0].success && side_states[1].success &&
        engine.active_phases(side_states[0]) ==
            engine.active_phases(side_states[1])) {
      expanded_for_amount = true;
      const double larger = std::min(
          length * .125,
          offset * 4. * engine.settings.amount_tolerance / unresolved_fraction);
      if (larger > offset * 1.5) {
        if (engine.settings.verbose)
          std::cerr << "Growing neighbour probe for phase " << line.zero_phase
                    << ": fraction=" << unresolved_fraction
                    << ", offset=" << offset << " -> " << larger << '\n';
        next_offset = larger;
        continue;
      }
    }
    // Canonical IDs record how many copies survive, not which solution
    // branch they occupy. Preserve a verified composition replacement
    // instead of overwriting both sides with the same fresh LP root.
    if (side_states[0].success && side_states[1].success &&
        engine.active_phases(side_states[0]) ==
            engine.active_phases(side_states[1]) &&
        !different_solution_branches(side_states[0], side_states[1]))
      for (int sign : {-1, 1}) {
        auto u = (centre + sign * offset * side).eval();
        if (inside(u)) {
          auto actual = (origin + u.cwiseProduct(range)).eval();
          side_states[sign < 0 ? 0 : 1] = discover(sign < 0 ? 0 : 1, actual);
        }
      }
    // A normal probe can cross an entire thin neighbouring field. Its
    // immediate assemblage must be drawn from the phases on this boundary;
    // shrink the probe when a different boundary has already been crossed.
    // A solution replacement exchanges one phase. Compound reactions can
    // legitimately remove several phases together, so exclude them from
    // this phase-count check.
    for (auto &state : side_states)
      if (state.success && !immediate_neighbour(state))
        state = State{};
    if (distinguishes_sides())
      break;
    // Each side can need a different perturbation to escape an active face
    // while remaining inside its narrow field. Retain verified neighbours
    // when the next, smaller probe fails on just one side.
    for (std::size_t slot = 0; slot < side_states.size(); ++slot)
      if (side_states[slot].success)
        previous_sides[slot] = side_states[slot];
      else if (previous_sides[slot].success)
        side_states[slot] = previous_sides[slot];
    if (distinguishes_sides())
      break;
    if (side_states[0].success && side_states[1].success &&
        different_solution_branches(side_states[0], side_states[1])) {
      line.is_solution_replacement = true;
      break;
    }
  }
  for (std::size_t slot = 0; slot < side_states.size(); ++slot)
    if (!neighbours[slot].empty() &&
        engine.active_phases(side_states[slot]) != neighbours[slot])
      side_states[slot] = State{};
  if (side_states[0].success) {
    line.side_a = engine.active_phases(side_states[0]);
    add_sample(side_states[0]);
  }
  if (side_states[1].success) {
    line.side_b = engine.active_phases(side_states[1]);
    add_sample(side_states[1]);
  }
  if (ambiguous_boundary(line))
    result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                 ": neighbouring fields could not be "
                                 "distinguished within tolerances.");
}

bool Tracer::same_curve(const Boundary &first, const Boundary &second,
                        bool partial, bool require_equilibrium) {
  return same_boundary_curve(engine, result, first, second, partial,
                             require_equilibrium);
}

bool Tracer::complete_boundary(const Boundary &line) const {
  const auto [back, forward] = boundary_statuses(line);
  return line.start_node >= 0 && line.end_node >= 0 &&
         !unfinished_boundary(back) && !unfinished_boundary(forward);
}

void Tracer::retain_curve(Boundary &kept, Boundary &other) {
  if (complete_boundary(kept) != complete_boundary(other)) {
    if (complete_boundary(other)) {
      const int id = kept.id;
      kept = std::move(other);
      kept.id = id;
    }
    return;
  }
  if (result.section.type == DiagramType::PT) {
    if (other.points.size() > kept.points.size()) {
      const int id = kept.id;
      kept = std::move(other);
      kept.id = id;
    }
    return;
  }
  if (kept.zero_phase != other.zero_phase || kept.points.size() < 2)
    return;
  std::vector<std::pair<double, BoundaryPoint>> ordered;
  for (std::size_t i = 0; i < kept.points.size(); ++i)
    ordered.push_back({static_cast<double>(i), kept.points[i]});
  for (const auto &point : other.points) {
    const auto u = normalise(point);
    double nearest = std::numeric_limits<double>::infinity(), position = 0.;
    for (std::size_t i = 1; i < kept.points.size(); ++i) {
      const auto a = normalise(kept.points[i - 1]);
      const auto d = (normalise(kept.points[i]) - a).eval();
      const double fraction =
          d.squaredNorm() == 0.
              ? 0.
              : std::clamp((u - a).dot(d) / d.squaredNorm(), 0., 1.);
      const double distance = (u - a - fraction * d).squaredNorm();
      if (distance < nearest) {
        nearest = distance;
        position = static_cast<double>(i - 1) + fraction;
      }
    }
    ordered.push_back({position, point});
  }
  std::stable_sort(
      ordered.begin(), ordered.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });
  kept.points.clear();
  for (const auto &entry : ordered)
    if (kept.points.empty() ||
        (normalise(kept.points.back()) - normalise(entry.second)).norm() >
            1.e-10)
      kept.points.push_back(entry.second);
}

void Tracer::trace(const Seed &seed) {
  if (covered(seed))
    return;

  try {
    auto actual = (origin + seed.position.cwiseProduct(range)).eval();
    std::vector<Eigen::Vector2d> normals;
    if (seed.direction.norm() > 0)
      normals.push_back(seed.direction);
    normals.push_back(Eigen::Vector2d(0., 1.));
    normals.push_back(Eigen::Vector2d(1., 0.));
    std::shared_ptr<Assemblage> a;
    EquilibriumSolve s;
    Eigen::VectorXd d;
    double affinity = 0.;
    bool accepted = false;
    // At a node the new branch direction is initially unknown. Try both
    // coordinate sections: one can duplicate an equilibrium equation (for
    // example fixing T on a horizontal edge), leaving a singular corrector.
    for (auto &normal : normals)
      try {
        a = engine.make_at(seed.ids, seed.phases, actual, seed.physical);
        if (!valid_amount_edge(*a, seed.ids, seed.zero))
          return;
        s = solve_boundary(*a, seed.ids, seed.zero, seed.position, normal);
        if (!valid(*a, s) || !engine.distinct(*a, seed.ids) ||
            (seed.node >= 0 &&
             (normalised_coordinates(*a) - seed.position).norm() >
                 engine.settings.node_tolerance))
          continue;
        std::vector<Minimum> minima;
        affinity = engine.stability(*a, &minima);
        if (needs_face_refit(affinity))
          refit_composition_faces(a, seed.ids, seed.zero, seed.position, normal,
                                  s, affinity, &minima);
        if (affinity < -engine.settings.affinity_tolerance)
          continue;
        if (seed.node >= 0 &&
            (normalised_coordinates(*a) - seed.position).norm() >
                engine.settings.node_tolerance)
          continue;
        d = tangent(s, *a);
        if (!d.size())
          continue;
        accepted = true;
        break;
      } catch (const std::exception &) {
      }
    if (!accepted) {
      if (engine.settings.verbose) {
        // If P or T are nan, don't print them.
        if (std::isnan(a ? a->get_pressure() : 0.) ||
            std::isnan(a ? a->get_temperature() : 0.))
          std::cerr << "Rejected boundary corrector: node=" << seed.node
                    << " zero=" << seed.zero << " affinity=" << affinity << ": "
                    << s.message << '\n';
        else
          std::cerr << "Rejected boundary corrector: node=" << seed.node
                    << " zero=" << seed.zero << " affinity=" << affinity
                    << " P=" << (a ? a->get_pressure() : 0.)
                    << " T=" << (a ? a->get_temperature() : 0.) << ": "
                    << s.message << '\n';
      }
      return;
    }
    Seed corrected = seed;
    corrected.position = normalised_coordinates(*a);
    corrected.phases = engine.snapshot(*a, seed.ids);
    corrected.physical =
        Eigen::Vector2d(a->get_pressure(), a->get_temperature());
    if (covered(corrected))
      return;
    // Near a known critical point, nearly identical-copy roots can satisfy
    // reaction tolerances without locating a solvus. Its separation-based
    // branch recovery supplies finite, distinct compositions instead.
    if (solution_separation(corrected.phases) < .002)
      for (const auto &n : result.nodes)
        if (n.kind == "critical_point" &&
            (normalise(n) - corrected.position).norm() <
                engine.settings.node_tolerance * 2.)
          return;
    Boundary line;
    line.id = burnman::utils::checked_int(result.boundaries.size());
    line.zero_phase = seed.zero;
    line.assemblage = seed.ids;
    line.start_node = seed.node;
    line.end_node = seed.node;
    auto unit = engine.project_direction(d, *a).cwiseQuotient(range).eval();
    std::string forward_status, backward_status;
    const auto seed_point = point(*a, seed.ids, s, affinity);
    auto backward_a = engine.copy_assemblage(*a);
    if (seed.bridge_node >= 0 && unit.dot(seed.direction) < 0.)
      d = -d;
    auto forward = follow(a, corrected, s, d, line.end_node, forward_status);
    std::vector<BoundaryPoint> backward;
    if (seed.bridge_node >= 0) {
      line.start_node = seed.bridge_node;
      backward_status = "solution critical point";
    } else if (forward_status == "closed loop")
      backward_status = "closed loop";
    else
      backward = follow(backward_a, corrected, s, -d, line.start_node,
                        backward_status);
    std::reverse(backward.begin(), backward.end());
    line.points = std::move(backward);
    line.points.push_back(seed_point);
    line.points.insert(line.points.end(), forward.begin(), forward.end());
    if (seed.bridge_node >= 0)
      line.points.insert(line.points.begin(), seed.prefix.begin(),
                         seed.prefix.end() - 1);
    if (forward_status == "closed loop")
      line.points.push_back(line.points.front());
    line.termination = backward_status + "; " + forward_status;
    trim_boundary_endpoints(line, engine, range);
    if (line.points.size() < 2)
      return;
    double length = 0.;
    for (std::size_t k = 1; k < line.points.size(); ++k)
      length +=
          (normalise(line.points[k]) - normalise(line.points[k - 1])).norm();
    if (length < engine.settings.min_step * 2.)
      return;
    label(line);
    // Polymorph replacement can be represented by either zero amount; keep
    // one geometrical edge and one incident line at its nodes.
    for (auto &old : result.boundaries) {
      bool same_sides = same_boundary_sides(old, line);
      bool same_ends = line.start_node >= 0 && line.end_node >= 0 &&
                       (same_boundary_nodes(old, line));
      if (same_sides && same_ends && old.assemblage == line.assemblage &&
          same_curve(old, line)) {
        retain_curve(old, line);
        return;
      }
    }
    record_incident_line(result, line);
    if (backward_status.find("unresolved") != std::string::npos ||
        forward_status.find("unresolved") != std::string::npos ||
        backward_status == "corrector failed" ||
        forward_status == "corrector failed" ||
        backward_status == "trace step limit" ||
        forward_status == "trace step limit")
      result.diagnostics.push_back("Boundary " + std::to_string(line.id) +
                                   ": " + line.termination);
    result.boundaries.push_back(std::move(line));
  } catch (const std::exception &e) {
    if (engine.settings.verbose)
      std::cerr << "Rejected boundary seed: " << e.what() << '\n';
  }
}

void Tracer::search() {
  int seeds = 0;
  const int limit = engine.settings.max_lines * 30;
  while (!queue.empty() &&
         result.boundaries.size() <
             static_cast<std::size_t>(engine.settings.max_lines) &&
         seeds < limit) {
    auto seed = canonical_seed(queue.front());
    queue.pop_front();
    std::ostringstream key;
    for (int id : seed.ids)
      key << id << ',';
    key << ':' << seed.zero << ':' << seed.node;
    if (seed.node >= 0) {
      // Different removals of coexisting copies can have identical
      // canonical IDs. A failed composition must not suppress another one.
      auto &trials = attempted[key.str()];
      if (std::any_of(trials.begin(), trials.end(), [&](const auto &old) {
            return same_boundary_compositions(
                seed.phases, seed.zero, old, seed.zero,
                engine.settings.composition_tolerance,
                engine.settings.amount_tolerance);
          }))
        continue;
      trials.push_back(seed.phases);
    }
    trace(seed);
    ++seeds;
    if (engine.settings.verbose)
      std::cerr << "Traced " << result.boundaries.size() << " lines; "
                << queue.size() << " seeds remaining\n";
  }
}

void Tracer::recover() {
  for (int pass = 0; pass < engine.settings.max_recovery_passes; ++pass) {
    bool progress = false;
    const std::size_t count = result.boundaries.size();
    for (std::size_t index = 0; index < count; ++index) {
      auto &line = result.boundaries[index];
      auto [backward, forward] = boundary_statuses(line);
      for (bool back : {false, true}) {
        auto &status = back ? backward : forward;
        if (!line.points.empty()) {
          auto &p = back ? line.points.front() : line.points.back();
          if (model_excluded(p))
            continue;
        }
        // Older saved traces could finish at an identical-copy root and call
        // it a junction. Such a point does not describe two coexisting
        // phases.
        if (status != "solution critical point" && !line.points.empty() &&
            solution_separation(
                (back ? line.points.front() : line.points.back()).phases) <=
                engine.settings.composition_tolerance * 10.) {
          status = "reduced variance";
          (back ? line.start_node : line.end_node) = -1;
        }
        bool critical = status == "solution critical point";
        if (critical) {
          int n = back ? line.start_node : line.end_node;
          if (n >= 0 &&
              result.nodes[static_cast<std::size_t>(n)].incident_lines.size() >
                  1 &&
              result.nodes[static_cast<std::size_t>(n)].critical_mode.size())
            continue;
        }
        if (!unfinished_boundary(status) && status != "reduced variance" &&
            !critical)
          continue;
        if (line.points.size() < 2)
          continue;
        auto original = back ? line.points.front() : line.points.back();
        auto adjacent =
            back ? line.points[1] : line.points[line.points.size() - 2];
        Eigen::Vector2d hint = normalise(original) - normalise(adjacent);
        // A former rank stop may have been a trivial root with identical
        // solution copies. Rewind to the last genuinely distinct state.
        while (line.points.size() > 2 &&
               solution_separation(
                   (back ? line.points.front() : line.points.back()).phases) <=
                   engine.settings.composition_tolerance * 10.) {
          if (back)
            line.points.erase(line.points.begin());
          else
            line.points.pop_back();
        }
        while (line.points.size() > 1) {
          auto &p = back ? line.points.front() : line.points.back();
          auto &q = back ? line.points[1] : line.points[line.points.size() - 2];
          if ((normalise(p) - normalise(q)).norm() > 1.e-12)
            break;
          if (back)
            line.points.erase(line.points.begin());
          else
            line.points.pop_back();
        }
        auto saved = back ? line.points.front() : line.points.back();
        Eigen::Vector2d u = normalise(saved);
        Eigen::Vector2d along = hint;
        if (line.points.size() > 1) {
          auto &neighbour =
              back ? line.points[1] : line.points[line.points.size() - 2];
          along = u - normalise(neighbour);
        }
        if (along.norm() < 1.e-12)
          continue;
        along.normalize();
        if (unfinished_boundary(status))
          (back ? line.start_node : line.end_node) = -1;
        try {
          auto a = engine.make_at(
              line.assemblage, saved.phases,
              diagram_coordinates(saved, result.section.type),
              Eigen::Vector2d(saved.pressure, saved.temperature));

          auto s =
              solve_boundary(*a, line.assemblage, line.zero_phase, u, along);
          if (!valid(*a, s))
            continue;
          std::vector<Minimum> minima;
          double affinity = engine.stability(*a, &minima);
          if (needs_face_refit(affinity))
            refit_composition_faces(a, line.assemblage, line.zero_phase, u,
                                    along, s, affinity, &minima);
          if (affinity < -engine.settings.affinity_tolerance)
            continue;
          bool near_critical = false;
          if (unfinished_boundary(status) || status == "reduced variance") {
            auto states = engine.snapshot(*a, line.assemblage);
            auto z = find_phase(states, line.zero_phase);
            if (z != states.end())
              for (auto &other : states)
                if (other.id != z->id &&
                    other.candidate_index == z->candidate_index &&
                    (other.composition - z->composition).norm() < .04)
                  near_critical = true;
          }
          if (critical || near_critical) {
            std::vector<BoundaryPoint> extra;
            if (approach_critical(a, line.assemblage, line.zero_phase, s,
                                  extra)) {
              int n = node(*a, line.assemblage, {line.zero_phase},
                           "critical_point", &s.J);
              if (back) {
                std::reverse(extra.begin(), extra.end());
                line.points.insert(line.points.begin(), extra.begin(),
                                   extra.end());
              } else
                line.points.insert(line.points.end(), extra.begin(),
                                   extra.end());
              (back ? line.start_node : line.end_node) = n;
              status = "solution critical point";
              progress = true;
              continue;
            }
            if (critical)
              continue;
          }
          auto d = tangent(s, *a);
          if (!d.size())
            continue;
          if (engine.project_direction(d, *a).cwiseQuotient(range).dot(along) <
              0.)
            d = -d;
          Seed seed{line.assemblage,
                    line.zero_phase,
                    -2,
                    engine.snapshot(*a, line.assemblage),
                    normalised_coordinates(*a),
                    along};
          int end = -1;
          std::string ended;
          auto extra = follow(a, seed, s, d, end, ended);
          if (!extra.empty()) {
            progress = true;
            if (back) {
              std::reverse(extra.begin(), extra.end());
              line.points.insert(line.points.begin(), extra.begin(),
                                 extra.end());
            } else
              line.points.insert(line.points.end(), extra.begin(), extra.end());
            status = ended;
            (back ? line.start_node : line.end_node) = end;
          } else if (end >= 0) {
            progress = true;
            status = ended;
            (back ? line.start_node : line.end_node) = end;
          }
          if (engine.settings.verbose)
            std::cerr << "Recovered boundary " << line.id
                      << (back ? " start: " : " end: ") << extra.size()
                      << " accepted points; " << ended << '\n';
        } catch (const std::exception &error) {
          if (engine.settings.verbose)
            std::cerr << "Recovery of boundary " << line.id << ": "
                      << error.what() << '\n';
        }
      }
      line.termination = backward + "; " + forward;
    }
    search();
    if (!progress)
      break;
  }
}

bool Tracer::refine_segment(Boundary &line, std::size_t right_index,
                            bool reconnect) {
  const auto left = normalise(line.points[right_index - 1]);
  const auto right = normalise(line.points[right_index]);
  const auto accuracy =
      equilibrium_coordinate_tolerance(result.section.type, range);
  if (((right - left).cwiseAbs().cwiseProduct(range).array() <=
       accuracy.array())
          .all())
    return false;
  double length = (right - left).norm();
  // Inverse coordinates can leave duplicate endpoints at the native solve
  // accuracy. Direct P-T repairs still need to resolve short reaction arcs.
  const double coordinate_accuracy =
      engine.direct_coordinates()
          ? 0.
          : equilibrium_coordinate_tolerance(result.section.type, range)
                .cwiseQuotient(range)
                .norm();
  if (length < std::max(64. * std::numeric_limits<double>::epsilon(),
                        coordinate_accuracy))
    return false;
  // At a critical endpoint the two compositions coincide. The other endpoint
  // can supply a distinct starting pair for the inserted state.
  std::string failures;
  bool unstable = false;
  auto rejected = [&](int side, const std::string &reason) {
    failures += (side == 0   ? "left endpoint: "
                 : side == 1 ? "; right endpoint: "
                             : "; interpolated endpoints: ") +
                reason;
  };
  for (int side = 0; side < 3; ++side)
    try {
      const Eigen::Vector2d middle = .5 * (left + right);
      auto states =
          line.points[right_index - 1 + static_cast<std::size_t>(side == 1)]
              .phases;
      if (side == 2)
        states = average_phase_states(std::move(states),
                                      line.points[right_index].phases);
      auto a = engine.make_at(
          line.assemblage, states, origin + middle.cwiseProduct(range),
          Eigen::Vector2d(.5 * (line.points[right_index - 1].pressure +
                                line.points[right_index].pressure),
                          .5 * (line.points[right_index - 1].temperature +
                                line.points[right_index].temperature)));

      const Eigen::Vector2d normal = (right - left).normalized();
      auto s =
          solve_boundary(*a, line.assemblage, line.zero_phase, middle, normal);
      auto acceptable = [&]() {
        const auto position = normalised_coordinates(*a);
        return valid(*a, s) && engine.distinct(*a, line.assemblage) &&
               (position - middle).norm() <= length &&
               std::min((position - left).norm(), (position - right).norm()) >
                   64. * std::numeric_limits<double>::epsilon();
      };
      if (!acceptable()) {
        rejected(side, "Boundary checks failed. " + s.message);
        continue;
      }
      std::vector<Minimum> minima;
      double affinity = engine.stability(*a, &minima);
      if (needs_face_refit(affinity))
        refit_composition_faces(a, line.assemblage, line.zero_phase, middle,
                                normal, s, affinity, &minima);
      if (affinity < -engine.settings.affinity_tolerance || !acceptable()) {
        unstable = unstable || affinity < -engine.settings.affinity_tolerance;
        rejected(side, "Boundary checks failed after checking stability "
                       "(minimum affinity " +
                           std::to_string(affinity) + " J/mol). " + s.message);
        continue;
      }
      line.points.insert(line.points.begin() +
                             static_cast<std::ptrdiff_t>(right_index),
                         point(*a, line.assemblage, s, affinity));
      return true;
    } catch (const std::exception &error) {
      rejected(side, error.what());
    }
  if (unstable || reconnect)
    unstable_lines.insert(line.id);
  if (engine.settings.verbose) {
    const auto &p = line.points[right_index - 1];
    const auto &q = line.points[right_index];
    std::ostringstream message;
    message << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "Boundary " << line.id << ": could not refine segment "
            << right_index << " between ";
    auto endpoint = [&](const auto &point) {
      message << "(P=" << point.pressure << " Pa, T=" << point.temperature
              << " K";
      for (auto axis : diagram_axes(result.section.type))
        switch (axis) {
        case Coordinate::S:
          message << ", S=" << point.entropy << " J/K";
          break;
        case Coordinate::V:
          message << ", V=" << point.volume << " m^3";
          break;
        case Coordinate::X:
          message << ", X=" << point.composition_coordinate;
          break;
        default:
          break;
        }
      message << ')';
    };
    endpoint(p);
    message << " and ";
    endpoint(q);
    message << "; normalised segment length=" << length << ": " << failures
            << '\n';
    std::cerr << message.str();
  }
  return false;
}

State Tracer::field_state(const Eigen::Vector2d &location) {
  std::vector<std::pair<double, std::size_t>> nearby;
  for (std::size_t i = 0; i < result.samples.size(); ++i)
    if (result.samples[i].success)
      nearby.push_back(
          {(normalise(result.samples[i]) - normalise(location)).squaredNorm(),
           i});
  std::sort(nearby.begin(), nearby.end());
  for (std::size_t i = 0; i < std::min<std::size_t>(32, nearby.size()); ++i)
    try {
      const auto &previous = result.samples[nearby[i].second];
      auto ids = phase_ids(previous.phases);
      auto work = engine.fixed_at(
          ids, previous.phases, location,
          Eigen::Vector2d(previous.pressure, previous.temperature));
      if (work.state.success && engine.distinct(*work.assemblage, work.ids)) {
        work.state.message =
            "Closed field verified from a neighbouring equilibrium state";
        return work.state;
      }
    } catch (const std::exception &) {
    }
  return engine.stable_at(location).state;
}

bool Tracer::model_excluded(double first, double second) const {
  Eigen::Vector2d p = normalise(first, second);
  for (auto &region : result.excluded_regions) {
    bool inside = false;
    for (int i = 1; i < region.rows(); ++i) {
      auto a = normalise(region(i - 1, 0), region(i - 1, 1)),
           b = normalise(region(i, 0), region(i, 1));
      if ((a[1] > p[1]) != (b[1] > p[1]) &&
          p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        inside = !inside;
    }
    if (inside)
      return true;
  }
  return false;
}

void Tracer::model_domain() {
  if (result.excluded_regions.empty())
    result.excluded_regions = excluded_model_regions(engine, origin, range);
}

void Tracer::reconcile_boundaries(bool transfer_labels_only) {
  std::vector<Boundary> unique;
  for (auto &line : result.boundaries) {
    bool duplicate = false;
    for (auto &old : unique) {
      // The early pass only merges pairs whose unresolved labels can be
      // inherited. Other duplicates retain ordinary label recovery.
      if (transfer_labels_only &&
          ambiguous_boundary(line) == ambiguous_boundary(old))
        continue;
      bool same_sides = same_boundary_sides(old, line);
      bool same_ends = same_boundary_nodes(old, line);
      const bool partial = complete_boundary(line) != complete_boundary(old);
      auto &part = complete_boundary(line) ? old : line;
      auto &whole = complete_boundary(line) ? line : old;
      if (partial) {
        // An unfinished retrace is removable only when an anchored,
        // completed record covers its entire equilibrium branch.
        auto shares = [&](int node) {
          return node >= 0 &&
                 (node == whole.start_node || node == whole.end_node);
        };
        same_ends = (shares(part.start_node) || shares(part.end_node)) &&
                    (part.start_node < 0 || shares(part.start_node)) &&
                    (part.end_node < 0 || shares(part.end_node));
        if (line.zero_phase != old.zero_phase ||
            line.is_solution_replacement != old.is_solution_replacement)
          continue;
      } else if (line.start_node < 0 || line.end_node < 0)
        continue;
      bool uncertain = ambiguous_boundary(line) || ambiguous_boundary(old);
      const bool univariant = result.section.type == DiagramType::PT &&
                              boundary_variance(line, result) == 1 &&
                              boundary_variance(old, result) == 1;
      if ((!same_sides && !uncertain && !univariant) || !same_ends ||
          old.assemblage != line.assemblage)
        continue;
      const bool verify = uncertain || line.zero_phase != old.zero_phase ||
                          boundary_variance(line, result) == 1;
      if (same_curve(partial ? part : line, partial ? whole : old, partial,
                     verify || partial)) {
        // An equivalent retrace can have unreliable probes near a tiny
        // field. Transfer labels only after verifying both curves and
        // their phase compositions, not merely their chord geometry.
        auto &bad = ambiguous_boundary(line) ? line : old;
        const auto &good = ambiguous_boundary(line) ? old : line;
        if (uncertain && !ambiguous_boundary(good)) {
          bool aligned = bad.start_node >= 0 && good.start_node >= 0
                             ? bad.start_node == good.start_node
                             : bad.end_node == good.end_node;
          bad.side_a = aligned ? good.side_a : good.side_b;
          bad.side_b = aligned ? good.side_b : good.side_a;
          bad.is_solution_replacement = good.is_solution_replacement;
        }
        duplicate = true;
        retain_curve(old, line);
        if (!same_sides && univariant)
          label(old, true);
        break;
      }
    }
    if (!duplicate)
      unique.push_back(std::move(line));
  }
  result.boundaries = std::move(unique);
}

void Tracer::index_boundaries() {
  std::set<int> unstable;
  for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
    auto &line = result.boundaries[i];
    if (unstable_lines.count(line.id))
      unstable.insert(burnman::utils::checked_int(i));
    line.id = burnman::utils::checked_int(i);
  }
  unstable_lines = std::move(unstable);
}

void Tracer::apply_model_domain() {
  separate_junctions();
  model_domain();
  detail::join_frame_limits(result, engine);
  detail::join_model_limits(result);
  for (auto &state : result.samples)
    if (model_excluded(state)) {
      state.success = false;
      state.outside_model_domain = true;
      state.message = "Outside the required EOS model domain.";
    }
  if (!result.excluded_regions.empty()) {
    auto states = std::move(result.samples);
    result.samples.clear();
    result.fields.clear();
    field_index.clear();
    for (auto &state : states)
      add_sample(state);
  }
  erase_if(result.boundaries, [&](const Boundary &line) {
    // Pure composition-edge traces have no two-sided field. Keep
    // thermodynamic boundaries even when they lie near a P/T edge.
    if (has_composition_axis(result.section.type))
      for (double edge : {0., 1.})
        if (!line.points.empty() &&
            std::all_of(line.points.begin(), line.points.end(),
                        [&](const BoundaryPoint &p) {
                          return std::abs(normalise(p)[1] - edge) < 1.e-8;
                        }))
          return true;
    return !line.points.empty() &&
           std::all_of(
               line.points.begin(), line.points.end(),
               [&](const BoundaryPoint &p) { return model_excluded(p); });
  });
}

void Tracer::remove_retraced_spurs() {
  // Older junction correctors could return to the same node in both
  // directions, leaving a collinear out-and-back spur. Replace it only
  // when equilibrium and compositions verify that another edge covers it;
  // the ordinary duplicate removal below then keeps that physical edge.
  for (auto &line : result.boundaries) {
    if (line.start_node < 0 || line.start_node != line.end_node ||
        line.points.size() < 3)
      continue;
    const auto start = normalise(line.points.front());
    auto farthest = start;
    for (const auto &point : line.points)
      if ((normalise(point) - start).squaredNorm() >
          (farthest - start).squaredNorm())
        farthest = normalise(point);
    if ((farthest - start).norm() < 1.e-10 ||
        !std::all_of(line.points.begin(), line.points.end(),
                     [&](const BoundaryPoint &point) {
                       return segment_distance(normalise(point), start,
                                               farthest) < 1.e-10;
                     }))
      continue;
    for (const auto &other : result.boundaries)
      if (other.start_node != other.end_node &&
          (other.start_node == line.start_node ||
           other.end_node == line.start_node) &&
          other.assemblage == line.assemblage &&
          other.zero_phase == line.zero_phase &&
          same_boundary_sides(other, line) && same_curve(line, other, true)) {
        const int id = line.id;
        line = other;
        line.id = id;
        break;
      }
  }
}

void Tracer::recover_failed_seeds() {
  // An early LP seed can fail even when the subsequently discovered field
  // supplies an accurate warm start. Reverify that location instead of
  // retaining a stale failure after the boundary construction has succeeded.
  const std::size_t sample_count = result.samples.size();
  for (std::size_t i = 0; i < sample_count; ++i)
    if (!result.samples[i].success && !result.samples[i].outside_model_domain) {
      auto &old = result.samples[i];
      const auto location = diagram_coordinates(old, result.section.type);
      auto verified = field_state(location);
      // An inverse Newton walk can hit a bound from one rounded target yet
      // converge at a neighbouring target. Use that nearby equilibrium only
      // as a starting point, then solve and validate the original request.
      if (!verified.success && !engine.direct_coordinates()) {
        const auto axes = diagram_axes(result.section.type);
        const auto accuracy =
            equilibrium_coordinate_tolerance(result.section.type, range);
        for (std::size_t axis = 0; axis < axes.size() && !verified.success;
             ++axis) {
          if (axes[axis] != Coordinate::S && axes[axis] != Coordinate::V)
            continue;
          for (double sign : {1., -1.}) {
            auto nearby = location;
            nearby[static_cast<Eigen::Index>(axis)] +=
                sign * accuracy[static_cast<Eigen::Index>(axis)];
            if (!inside(normalise(nearby)))
              continue;
            auto warm = field_state(nearby);
            if (!warm.success)
              continue;
            auto work = engine.fixed_at(
                phase_ids(warm.phases), warm.phases, location,
                Eigen::Vector2d(warm.pressure, warm.temperature));
            if (work.state.success && engine.locally_stable(*work.assemblage) &&
                engine.distinct(*work.assemblage, work.ids)) {
              verified = std::move(work.state);
              break;
            }
          }
        }
      }
      if (verified.success)
        add_sample(verified, burnman::utils::checked_int(i));
    }
}

void Tracer::repair_and_refine_boundaries(bool initial) {
  reconcile_boundaries(initial);
  repair_field_boundaries(
      result, engine, [&](Boundary &line) { label(line); },
      [&](Boundary &line, std::size_t k) {
        return refine_segment(line, k, true);
      },
      [&](Boundary &line) { label(line, true); });
  // Short duplicates may become verifiable only after recovering side labels.
  reconcile_boundaries();
  index_boundaries();
  // Rebuild from surviving records; earlier failures may now be recovered.
  result.diagnostics.clear();
  result.spacing_diagnostics =
      line_resolution
          ? detail::refine_line_spacing(result, engine, *line_resolution,
                                        [&](Boundary &line, std::size_t k) {
                                          return refine_segment(line, k);
                                        })
          : std::vector<std::string>{};
}

void Tracer::retrace_unstable_boundaries() {
  if (line_resolution)
    for (int id : regional_retrace_lines(result, *line_resolution))
      unstable_lines.insert(id);
  // Retrace unstable chords or a requested region's conflicting arcs once;
  // ordinary phase-entry recovery supplies their missing connections.
  if (engine.settings.max_recovery_passes && !unstable_lines.empty()) {
    auto old = std::move(result.boundaries);
    result.boundaries.clear();
    for (auto &line : old)
      if (unstable_lines.count(line.id)) {
        for (bool start : {true, false}) {
          const auto &p = start ? line.points.front() : line.points.back();
          const auto &q = start ? line.points.back() : line.points.front();
          queue.push_back({line.assemblage, line.zero_phase,
                           start ? line.start_node : line.end_node, p.phases,
                           normalise(p),
                           (normalise(q) - normalise(p)).normalized()});
          queue.back().physical = Eigen::Vector2d(p.pressure, p.temperature);
        }
      } else
        result.boundaries.push_back(std::move(line));
    unstable_lines.clear();
    attempted.clear();
    expanded.clear();
    for (auto &n : result.nodes)
      n.incident_lines.clear();
    for (std::size_t i = 0; i < result.boundaries.size(); ++i) {
      auto &line = result.boundaries[i];
      line.id = burnman::utils::checked_int(i);
      for (int n : {line.start_node, line.end_node})
        if (n >= 0)
          result.nodes.at(static_cast<std::size_t>(n))
              .incident_lines.push_back(line.id);
    }
    const auto saved_settings = engine.settings;
    if (line_resolution)
      engine.settings =
          refinement_settings(result, *line_resolution, saved_settings);
    engine.settings.step = saved_settings.step;
    search();
    recover();
    engine.settings = saved_settings;
    repair_and_refine_boundaries();
  }
}

bool Tracer::prepare_previous(const Result &previous,
                              const std::optional<LineResolution> &resolution) {
  if (resolution)
    validate_line_resolution(result, *resolution);
  line_resolution = resolution;
  validate_previous_identity(previous);
  result = previous;
  result.section = engine.section;
  result.settings = engine.settings;
  // Older files can contain an entire saddle branch near a solvus. Such
  // compositions pass finite affinity tolerance but cannot bound stable
  // fields. Retain every arc with any locally stable saved state.
  erase_if(result.boundaries, [&](const Boundary &line) {
    return !engine.direct_coordinates() && !line.points.empty() &&
           std::all_of(
               line.points.begin(), line.points.end(), [&](const auto &p) {
                 auto a = engine.make_assemblage(line.assemblage, p.phases,
                                                 p.pressure, p.temperature);
                 return !engine.locally_stable(*a) ||
                        !engine.distinct(*a, line.assemblage);
               });
  });
  // An old phase-entry endpoint can contain an additional stationary copy
  // that is a saddle or belongs to the same mixing basin. It is not a
  // junction: reopen the original reaction from its last accepted state.
  for (auto &line : result.boundaries) {
    if (engine.direct_coordinates() || line.points.empty())
      continue;
    auto [back, forward] = boundary_statuses(line);
    for (bool start : {true, false}) {
      auto &status = start ? back : forward;
      const auto &p = start ? line.points.front() : line.points.back();
      const auto ids = phase_ids(p.phases);
      if (status != "junction" || ids.size() <= line.assemblage.size())
        continue;
      auto a = engine.make_assemblage(ids, p.phases, p.pressure, p.temperature);
      const int node_id = start ? line.start_node : line.end_node;
      const Node *event =
          node_id >= 0 ? &result.nodes.at(static_cast<std::size_t>(node_id))
                       : nullptr;
      // A near-critical third-copy root can pass finite reaction tolerances
      // while its double-zero constraints fail to isolate a junction.
      // Reopen it for native recovery rather than retaining that connection.
      const bool nonisolated_copy =
          event && event->pt_nullity > 0 && event->zero_phases.size() == 2 &&
          near_third_copy_event(ids, event->zero_phases[0],
                                event->zero_phases[1], p.phases);
      if (nonisolated_copy || !engine.locally_stable(*a) ||
          !engine.distinct(*a, ids)) {
        (start ? line.start_node : line.end_node) = -1;
        status = "unresolved phase-in event";
        while (line.points.size() > 1) {
          const auto &last = start ? line.points.front() : line.points.back();
          auto original = engine.make_assemblage(
              line.assemblage, last.phases, last.pressure, last.temperature);
          if (engine.locally_stable(*original) &&
              engine.distinct(*original, line.assemblage))
            break;
          if (start)
            line.points.erase(line.points.begin());
          else
            line.points.pop_back();
        }
        auto &last = start ? line.points.front() : line.points.back();
        erase_if(last.phases, [&](const auto &phase) {
          return std::find(line.assemblage.begin(), line.assemblage.end(),
                           phase.id) == line.assemblage.end();
        });
      }
    }
    line.termination = back + "; " + forward;
  }
  if (!engine.direct_coordinates())
    index_boundaries();
  separate_junctions();
  if (!engine.settings.required_eos_phases.empty())
    result.excluded_regions.clear();
  model_domain();
  detail::join_frame_limits(result, engine);
  detail::join_model_limits(result);
  auto samples = std::move(result.samples);
  result.samples.clear();
  result.fields.clear();
  for (auto &state : samples) {
    if (state.success && !engine.direct_coordinates()) {
      const auto ids = phase_ids(state.phases);
      auto a = engine.make_assemblage(ids, state.phases, state.pressure,
                                      state.temperature);
      if (!engine.locally_stable(*a) || !engine.distinct(*a, ids)) {
        state.success = false;
        state.message =
            "Saved solution compositions are not stable Gibbs minima.";
      }
    }
    add_sample(state);
  }
  for (std::size_t i = 0; i < result.nodes.size(); ++i)
    if (result.nodes[i].id != static_cast<int>(i))
      throw std::invalid_argument("Saved node IDs must be contiguous.");
  const bool dangling_junction = std::any_of(
      result.nodes.begin(), result.nodes.end(), [&](const Node &node) {
        return node.kind == "junction" &&
               std::count_if(result.boundaries.begin(), result.boundaries.end(),
                             [&](const Boundary &line) {
                               return line.start_node == node.id ||
                                      line.end_node == node.id;
                             }) == 1;
      });
  if (resolution && (previous.resolved || resolution->region) &&
      (!dangling_junction || !engine.settings.max_recovery_passes)) {
    // Regional requests use the existing graph; only an unstable local arc
    // reopens its endpoints. Completed diagrams also skip general searches.
    return false;
  }
  // A seed-only saved calculation still supplies initial boundary brackets.
  if (!previous.resolved && previous.fields.empty() &&
      result.boundaries.empty() && engine.settings.max_recovery_passes) {
    for (const auto &state : result.samples)
      if (state.success && inside(normalise(state))) {
        auto ids = engine.active_phases(state);
        grid.push_back(
            {state,
             engine.make_at(ids, state.phases,
                            diagram_coordinates(state, result.section.type),
                            Eigen::Vector2d(state.pressure, state.temperature)),
             ids});
      }
    for (std::size_t i = 1; i < grid.size(); ++i)
      bracket(grid[i - 1], grid[i]);
  }
  // Successful endpoint states seed all untraced branches at their junction.
  for (auto &line : result.boundaries) {
    if (line.points.empty())
      continue;
    auto [back, forward] = boundary_statuses(line);
    for (bool start : {true, false}) {
      auto status = start ? back : forward;
      int id = start ? line.start_node : line.end_node;
      if ((status != "junction" && status != "solution critical point") ||
          id < 0)
        continue;
      auto &point = start ? line.points.front() : line.points.back();
      auto &n = result.nodes.at(static_cast<std::size_t>(id));
      if ((normalise(point) - normalise(n)).norm() >
          engine.settings.node_tolerance)
        continue;
      // A solution copy is renumbered when its neighbour disappears. The
      // endpoint states identify the actual zero copies; the shared node's
      // IDs can refer to the opposite composition of the same solution.
      auto ids = phase_ids(point.phases);
      const auto zeros =
          zero_phases_at(n, point.phases, engine.settings.max_phase_instances);
      auto a = engine.make_at(
          ids, point.phases, diagram_coordinates(n, result.section.type),
          Eigen::Vector2d(point.pressure, point.temperature));
      if (status == "junction")
        expand(*a, ids, zeros, id);
      // Critical endpoints are first approached from both arms in recover().
    }
  }
  return true;
}
} // namespace burnman::phase_diagrams::detail
