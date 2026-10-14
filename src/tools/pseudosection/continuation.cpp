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

#include "burnman/tools/equilibration/equilibrate_objective.hpp"
#include "burnman/utils/index_utils.hpp"
#include "internal.hpp"

namespace burnman::pseudosections::detail {
namespace {
// Differentiate a field branch with respect to a normal displacement in the
// rescaled diagram. The other equations retain reaction and mass balance.
Eigen::VectorXd field_response(const Engine &engine, const Assemblage &a,
                               Eigen::MatrixXd j, const Eigen::Vector2d &range,
                               const Eigen::Vector2d &normal) {
  if (j.rows() < 2 || j.rows() != j.cols() || !j.allFinite())
    return {};
  if (!engine.free_vectors.empty()) {
    // Reduced mass equations omit elements dependent in this assemblage.
    // The bulk path must also lie in its full compositional space; otherwise
    // this branch exists only at one X and cannot label a field beside it.
    const auto prm = engine.parameters(a);
    if ((a.get_compositional_null_basis() *
         prm.free_compositional_vectors.transpose())
            .norm() > 1.e-9 * prm.free_compositional_vectors.norm())
      return {};
  }
  j.topRows(2) = range.cwiseInverse().asDiagonal() *
                 engine.coordinate_jacobian(a, j.cols());
  auto scales = engine.parameter_scales(a, j.cols(), range);
  j *= scales.asDiagonal();
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(j.rows());
  rhs.head(2) = normal;
  for (Eigen::Index row = 0; row < j.rows(); ++row) {
    double norm = j.row(row).norm();
    if (norm > 0.) {
      j.row(row) /= norm;
      rhs[row] /= norm;
    }
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);
  svd.setThreshold(1.e-10);
  if (svd.rank() != j.cols())
    return {};
  Eigen::VectorXd response = svd.solve(rhs);
  if (!response.allFinite() || (j * response - rhs).norm() > 1.e-7 * rhs.norm())
    return {};
  return scales.cwiseProduct(response);
}

int phase_entry_side(Engine &engine, Assemblage &a, const std::vector<int> &ids,
                     int zero, const optim::roots::DampedNewtonResult &s,
                     const Eigen::Vector2d &range,
                     const Eigen::Vector2d &normal, int &full_side) {
  auto entry = std::find(ids.begin(), ids.end(), zero);
  auto prm = engine.parameters(a);
  if (entry == ids.end() || ids.size() < 2 || s.x.size() != prm.n_parameters)
    return 0;
  auto states = engine.snapshot(a, ids);
  const double total = a.get_n_moles();
  // A junction is not a regular arc, even if the file repeats it as a midpoint.
  for (const auto &state : states)
    if (state.id != zero && state.amount <= total * 1.e-10)
      return 0;
  auto response = field_response(engine, a, s.J, range, normal);
  if (!response.size())
    return 0;
  Eigen::Index amount = prm.phase_amount_indices[entry - ids.begin()];
  double change = response[amount] / total;
  if (std::abs(change) < 1.e-9)
    return 0;
  const int sign = change > 0. ? 1 : -1;
  full_side = sign;
  // Check the opposite field, including composition-axis feasibility. Merely
  // deleting a phase can leave a branch that exists only on the line itself.
  auto absent = ids;
  absent.erase(absent.begin() + (entry - ids.begin()));
  auto other = engine.make_at(absent, states, engine.coordinates(a));
  auto other_prm = engine.parameters(*other);
  auto x = get_parameter_vector(*other, engine.free_vectors.empty() ? 0 : 1);
  ConstraintGroup fixed;
  const auto axes = diagram_axes(engine.section.type);
  for (Coordinate axis : axes)
    fixed.push_back(engine.coordinate_constraint(axis, 0., x.size()));
  if (!engine.free_vectors.empty())
    fixed.push_back(engine.coordinate_constraint(
        *engine.section.fixed_coordinate, 0., x.size()));
  auto j = J(x, *other, fixed, other_prm.reduced_free_composition_vectors);
  auto opposite = field_response(engine, *other, j, range, -sign * normal);
  if (!opposite.size())
    return 0;
  // The disappearing phase must become unfavourable on the absent side.
  // Differentiate its tangent-plane affinity along the linearised branch;
  // the stationary phase composition makes this an envelope derivative.
  auto formula = engine.phase_formulae({zero}, states).row(0).eval();
  auto material =
      clone(a.get_phase(static_cast<std::size_t>(entry - ids.begin())));
  auto affinity = [&]() {
    auto mu = engine.potentials(*other);
    double value = material->get_molar_gibbs();
    for (std::size_t k = 0; k < engine.components.size(); ++k)
      value -= formula[engine.components[k]] * mu[static_cast<Eigen::Index>(k)];
    return value;
  };
  const double initial = affinity();
  double step = 1.e-6;
  const auto slack =
      (other_prm.constraint_matrix * x + other_prm.constraint_vector).eval();
  const auto rate = (other_prm.constraint_matrix * opposite).eval();
  for (Eigen::Index row = 0; row < slack.size(); ++row)
    if (rate[row] > 1.e-12) {
      if (slack[row] >= -1.e-12)
        return 0;
      step = std::min(step, -.1 * slack[row] / rate[row]);
    }
  if (step < 1.e-10)
    return 0;
  set_composition_and_state_from_parameters(*other, x + step * opposite);
  material->set_state(other->get_pressure(), other->get_temperature());
  const double derivative = (affinity() - initial) / step;
  return std::isfinite(derivative) && derivative > 1.e-5 ? sign : 0;
}

} // namespace

std::array<std::vector<int>, 2>
boundary_sides(Engine &engine, Assemblage &a, const std::vector<int> &ids,
               int zero, const optim::roots::DampedNewtonResult &s,
               const Eigen::Vector2d &range, const Eigen::Vector2d &normal,
               std::array<State, 2> *limiting_states) {
  if (limiting_states)
    *limiting_states = {};
  auto entry = std::find(ids.begin(), ids.end(), zero);
  if (entry == ids.end())
    return {};
  auto absent = ids;
  absent.erase(absent.begin() + (entry - ids.begin()));
  int full_side = 0;
  int sign =
      phase_entry_side(engine, a, ids, zero, s, range, normal, full_side);
  if (sign)
    return {engine.canonical_ids(sign > 0 ? absent : ids),
            engine.canonical_ids(sign > 0 ? ids : absent)};
  if (engine.section.type != DiagramType::PT) {
    // Preserve the full branch's orientation even when the opposite field
    // needs another phase. Local probes must not skip this narrow branch.
    std::array<std::vector<int>, 2> partial;
    if (full_side)
      partial[full_side > 0 ? 1 : 0] = engine.canonical_ids(ids);
    return partial;
  }
  // A compound reaction has dependent phase amounts. Its two limiting
  // assemblages follow the reaction extent, and dG = dV dP - dS dT orients
  // them even though the full assemblage is singular at fixed P,T.
  auto states = engine.snapshot(a, ids);
  for (const auto &state : states)
    if (state.id != zero && state.amount <= a.get_n_moles() * 1.e-10)
      return {};
  auto formulae = engine.phase_formulae(ids, states);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(formulae.transpose(),
                                        Eigen::ComputeFullV);
  svd.setThreshold(1.e-9);
  if (formulae.rows() - svd.rank() != 1)
    return {};
  Eigen::VectorXd reaction = svd.matrixV().rightCols(1);
  Eigen::Index entering = entry - ids.begin();
  if (std::abs(reaction[entering]) < 1.e-9)
    return {};
  if (reaction[entering] < 0.)
    reaction = -reaction;
  double extent = std::numeric_limits<double>::infinity();
  Eigen::VectorXd g(ids.size()), volume(ids.size()), entropy(ids.size());
  for (Eigen::Index k = 0; k < reaction.size(); ++k) {
    auto phase = a.get_phase(static_cast<std::size_t>(k));
    g[k] = phase->get_molar_gibbs();
    volume[k] = phase->get_molar_volume();
    entropy[k] = phase->get_molar_entropy();
    if (reaction[k] < -1.e-9)
      extent = std::min(extent, states[static_cast<std::size_t>(k)].amount /
                                    -reaction[k]);
  }
  if (!std::isfinite(extent) || extent <= 0. ||
      std::abs(reaction.dot(g)) >
          engine.settings.affinity_tolerance * reaction.cwiseAbs().sum())
    return {};
  std::vector<int> after;
  for (Eigen::Index k = 0; k < reaction.size(); ++k)
    if (states[static_cast<std::size_t>(k)].amount + extent * reaction[k] >
        a.get_n_moles() * 1.e-10)
      after.push_back(ids[static_cast<std::size_t>(k)]);
  const double slope = reaction.dot(volume) * range[0] * normal[0] -
                       reaction.dot(entropy) * range[1] * normal[1];
  if (!std::isfinite(slope) || std::abs(slope) < 1.e-5 || after.empty())
    return {};
  if (limiting_states) {
    // Both limits are equilibria at the same P,T. Retain their compositions
    // and amounts as contour warm starts without inventing off-line samples.
    for (int end : {0, 1}) {
      auto warm = states;
      if (end)
        for (std::size_t k = 0; k < warm.size(); ++k)
          warm[k].amount =
              std::max(0., warm[k].amount +
                               extent * reaction[static_cast<Eigen::Index>(k)]);
      const auto &present = end ? after : absent;
      auto limit = engine.make_assemblage(present, warm, a.get_pressure(),
                                          a.get_temperature());
      auto &state = (*limiting_states)[static_cast<std::size_t>(
          slope > 0. ? 1 - end : end)];
      engine.record_coordinates(state, *limit);
      state.phases = engine.snapshot(*limit, present);
      state.gibbs = limit->get_n_moles() * limit->get_molar_gibbs();
      state.mass_balance_error = engine.mass_error(*limit);
      state.equilibrium_error =
          limit->get_n_reactions()
              ? limit->get_reaction_affinities().cwiseAbs().maxCoeff()
              : 0.;
      state.success =
          state.mass_balance_error <= engine.settings.mass_balance_tolerance &&
          state.equilibrium_error <= engine.settings.affinity_tolerance * .1;
      state.message =
          "Equilibrium reaction limiting assemblage on the boundary";
      if (!state.success)
        return {};
    }
  }
  return {engine.canonical_ids(slope > 0. ? after : absent),
          engine.canonical_ids(slope > 0. ? absent : after)};
}

std::unique_ptr<EqualityConstraint>
continuation_plane(const Engine &engine, const Assemblage &a, Eigen::Index n,
                   const Eigen::Vector2d &target,
                   const Eigen::Vector2d &weights) {
  auto axes = diagram_axes(engine.section.type);
  double x = engine.composition_coordinate(a);
  return std::make_unique<SectionConstraint>(
      engine.coordinate_constraint(axes[0], target[0], n, x),
      engine.coordinate_constraint(axes[1], target[1], n, x), weights);
}
Eigen::VectorXd continuation_tangent(const Engine &engine,
                                     const optim::roots::DampedNewtonResult &s,
                                     const Assemblage &a,
                                     const Eigen::Vector2d &range,
                                     int *coordinate_rank) {
  if (s.x.size() != engine.parameters(a).n_parameters) {
    if (coordinate_rank)
      *coordinate_rank = 0;
    return Eigen::VectorXd();
  }
  auto scales = engine.parameter_scales(a, s.x.size(), range);
  // Row 0 fixes the phase amount or contour target; row 1 fixes the moving
  // plane. Keep all remaining equations, including the extra fixed coordinate
  // in X sections. Rescaling balances their sizes without changing equilibrium.
  Eigen::MatrixXd j(s.J.rows() - 1, s.J.cols());
  j.topRows(1) = s.J.topRows(1);
  j.bottomRows(s.J.rows() - 2) = s.J.bottomRows(s.J.rows() - 2);
  j = j * scales.asDiagonal();
  for (int k = 0; k < j.rows(); ++k) {
    double norm = j.row(k).norm();
    if (norm > 0)
      j.row(k) /= norm;
  }
  // Very small phase amounts make some equation directions hard to distinguish.
  // A tighter rank threshold retains their coupling to P,T at empty sites.
  double rank_tolerance =
      engine.settings.active_solution_faces ? 1.e-12 : 1.e-9;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(j, Eigen::ComputeFullV);
  int rank = burnman::utils::checked_int(
      (svd.singularValues().array() > rank_tolerance).count());
  Eigen::MatrixXd null = svd.matrixV().rightCols(j.cols() - rank);
  // Count independent ways equilibrium can move in the selected diagram.
  // Changing phase amounts can move S/V at constant P,T; other changes may
  // leave both plotted coordinates unchanged. Exactly one direction defines a
  // line.
  Eigen::JacobiSVD<Eigen::MatrixXd> projection(
      range.cwiseInverse().asDiagonal() *
          engine.coordinate_jacobian(a, s.x.size()) * scales.asDiagonal() *
          null,
      Eigen::ComputeThinU | Eigen::ComputeThinV);
  int dim = burnman::utils::checked_int(
      (projection.singularValues().array() > 1.e-7).count());
  if (coordinate_rank)
    *coordinate_rank = dim;
  if (dim != 1)
    return Eigen::VectorXd();
  auto d = (null * projection.matrixV().col(0)).eval();
  auto physical = (scales.asDiagonal() * d).eval();
  return physical /
         engine.project_direction(physical, a).cwiseQuotient(range).norm();
}
namespace {
std::optional<std::vector<std::size_t>>
phase_matches(const std::vector<PhaseState> &a,
              const std::vector<PhaseState> &b, double tolerance) {
  std::vector<std::size_t> matches;
  std::vector<bool> used(b.size(), false);
  for (const auto &phase : a) {
    std::size_t match = b.size();
    double distance = tolerance;
    for (std::size_t j = 0; j < b.size(); ++j)
      if (!used[j] && phase.candidate_index == b[j].candidate_index &&
          phase.composition.size() == b[j].composition.size() &&
          (phase.composition - b[j].composition).norm() < distance) {
        match = j;
        distance = (phase.composition - b[j].composition).norm();
      }
    if (match == b.size())
      return std::nullopt;
    used[match] = true;
    matches.push_back(match);
  }
  return matches;
}
} // namespace

bool seed_phase_compositions(std::vector<PhaseState> &warm,
                             const std::vector<PhaseState> &seed,
                             double tolerance, double amount_tolerance) {
  const auto matches =
      phase_matches(warm, seed, std::numeric_limits<double>::infinity());
  if (!matches)
    return false;
  double warm_total = 0., seed_total = 0.;
  for (const auto &phase : warm)
    warm_total += phase.amount;
  for (const auto &phase : seed)
    seed_total += phase.amount;
  // Preserve the compositions of phases present in both reaction extents.
  // A zero-amount phase can start on a different active face; its composition
  // must be verified after correcting its own zero-amount constraint.
  for (std::size_t i = 0; i < warm.size(); ++i) {
    const auto &other = seed[(*matches)[i]];
    if (warm[i].amount > amount_tolerance * warm_total &&
        other.amount > amount_tolerance * seed_total &&
        (warm[i].composition - other.composition).norm() > tolerance)
      return false;
  }
  for (std::size_t i = 0; i < warm.size(); ++i)
    warm[i].composition = seed[(*matches)[i]].composition;
  return true;
}

bool same_phase_compositions(const std::vector<PhaseState> &a,
                             const std::vector<PhaseState> &b, double tolerance,
                             double amount_tolerance) {
  if (a.size() > b.size())
    return same_phase_compositions(b, a, tolerance, amount_tolerance);
  const auto matches = phase_matches(a, b, tolerance);
  if (!matches)
    return false;
  double total = 0.;
  for (const auto &phase : b)
    total += phase.amount;
  for (std::size_t j = 0; j < b.size(); ++j)
    if (std::find(matches->begin(), matches->end(), j) == matches->end() &&
        b[j].amount > amount_tolerance * total)
      return false;
  return true;
}

bool same_boundary_compositions(const std::vector<PhaseState> &a, int zero_a,
                                const std::vector<PhaseState> &b, int zero_b,
                                double tolerance, double amount_tolerance) {
  auto zero = [](const auto &states, int id) {
    return std::find_if(states.begin(), states.end(),
                        [&](const auto &phase) { return phase.id == id; });
  };
  const auto x = zero(a, zero_a), y = zero(b, zero_b);
  return x != a.end() && y != b.end() &&
         x->candidate_index == y->candidate_index &&
         x->composition.size() == y->composition.size() &&
         (x->composition - y->composition).norm() < tolerance &&
         same_phase_compositions(a, b, tolerance, amount_tolerance);
}

std::vector<int> zero_phases_at(const Node &node,
                                const std::vector<PhaseState> &states,
                                int copies) {
  std::vector<int> zeros;
  for (int zero : node.zero_phases) {
    const PhaseState *best = nullptr;
    for (const auto &phase : states)
      if (phase.candidate_index == zero / copies &&
          std::find(zeros.begin(), zeros.end(), phase.id) == zeros.end() &&
          (!best || std::abs(phase.amount) < std::abs(best->amount)))
        best = &phase;
    if (best)
      zeros.push_back(best->id);
  }
  return zeros;
}

std::optional<BoundaryPoint> corrected_point(Engine &engine, Assemblage &a,
                                             const std::vector<int> &ids,
                                             ConstraintList &c,
                                             const Eigen::Vector2d &target,
                                             const Eigen::Vector2d &allowance) {
  auto solved = engine.solve(a, c);
  if (!solved.success || !solved.x.allFinite() ||
      engine.mass_error(a) > engine.settings.mass_balance_tolerance ||
      !engine.distinct(a, ids) ||
      ((engine.coordinates(a) - target).cwiseAbs().array() > allowance.array())
          .any())
    return std::nullopt;
  const double affinity = engine.stability(a);
  if (affinity < -engine.settings.affinity_tolerance)
    return std::nullopt;
  BoundaryPoint point;
  engine.record_coordinates(point, a);
  point.phases = engine.snapshot(a, ids);
  point.mass_balance_error = engine.mass_error(a);
  point.minimum_affinity = affinity;
  point.residual = solved.F.tail(solved.F.size() - 2).norm();
  return point;
}

// Correct an existing branch without starting a new bulk phase search.
std::optional<BoundaryPoint>
boundary_point_at(Engine &engine, const Boundary &line,
                  const std::vector<PhaseState> &warm,
                  const Eigen::Vector2d &target, const Eigen::Vector2d &weights,
                  const Eigen::Vector2d &allowance) {
  auto assembly = engine.make_at(line.assemblage, warm, target);
  const auto prm = engine.parameters(*assembly);
  const auto zero = std::find(line.assemblage.begin(), line.assemblage.end(),
                              line.zero_phase);
  if (zero == line.assemblage.end())
    return std::nullopt;
  auto c = constraints(
      std::make_unique<PhaseFractionConstraint>(zero - line.assemblage.begin(),
                                                0., prm),
      continuation_plane(engine, *assembly, prm.n_parameters, target, weights));
  return corrected_point(engine, *assembly, line.assemblage, c, target,
                         allowance);
}

// Use the same intensive seed (and active solution faces) for both events,
// while preserving each event's solution-copy identities and zero constraints.
std::optional<BoundaryPoint> junction_point_at(
    Engine &engine, const Node &node, std::vector<PhaseState> warm,
    const std::vector<PhaseState> &seed, const Eigen::Vector2d &allowance) {
  if (node.zero_phases.size() != 2)
    return std::nullopt;
  const auto zeros = zero_phases_at(node, warm, engine.settings.max_phase_instances);
  if (zeros.size() != 2)
    return std::nullopt;
  if (!seed_phase_compositions(warm, seed))
    return std::nullopt;
  for (auto &phase : warm) {
    if (std::find(zeros.begin(), zeros.end(), phase.id) != zeros.end())
      phase.amount = 0.;
  }
  const auto target = diagram_coordinates(node, engine.section.type);
  auto assembly = engine.make_at(node.assemblage, warm, target);
  const auto prm = engine.parameters(*assembly);
  auto zero = [&](int id) -> std::unique_ptr<EqualityConstraint> {
    const auto entry =
        std::find(node.assemblage.begin(), node.assemblage.end(), id);
    if (entry == node.assemblage.end())
      throw std::invalid_argument(
          "Junction zero phase is absent from its assemblage.");
    return std::make_unique<PhaseFractionConstraint>(
        entry - node.assemblage.begin(), 0., prm);
  };
  auto c = constraints(zero(zeros[0]), zero(zeros[1]));
  return corrected_point(engine, *assembly, node.assemblage, c, target,
                         allowance);
}

// An event found behind a point accepted within the affinity tolerance must
// replace the overshot tail. Verify the event on the earlier branch before
// trimming; coordinate order alone is insufficient for curved traces.
bool trim_boundary_endpoints(Boundary &line, Engine &engine,
                              const Eigen::Vector2d &range) {
  if (engine.section.type != DiagramType::PT ||
      line.points.size() < 3 || line.zero_phase < 0 ||
      (line.start_node >= 0 && line.start_node == line.end_node))
    return false;
  bool changed = false;
  const auto statuses = boundary_statuses(line);
  for (bool front : {false, true}) {
    const auto &status = front ? statuses.first : statuses.second;
    if (unfinished_boundary(status) || status == "closed loop")
      continue;
    if (front)
      std::reverse(line.points.begin(), line.points.end());
    const auto endpoint = line.points.back();
    const auto target = diagram_coordinates(endpoint, engine.section.type);
    auto q = [&](const auto &p) -> Eigen::Vector2d {
      return (diagram_coordinates(p, engine.section.type) - target)
          .cwiseQuotient(range);
    };
    std::size_t tail = line.points.size() - 2;
    const auto accuracy = equilibrium_pt_tolerance();
    while (tail > 0 &&
           ((diagram_coordinates(line.points[tail], engine.section.type) - target)
                .cwiseAbs().array() <= accuracy.array()).all())
      --tail;
    // Ordinary forward arrivals and repeated endpoint records need no solve.
    // Search the earlier branch only when attaching the event turns back.
    const bool turns_back = tail > 0 &&
        -q(line.points[tail]).dot(q(line.points[tail]) -
                                  q(line.points[tail - 1])) < 0.;
    for (std::size_t k = tail; turns_back && k > 0; --k) {
      const auto a = q(line.points[k - 1]), b = q(line.points[k]);
      const Eigen::Vector2d delta = b - a;
      if (delta.squaredNorm() == 0.)
        continue;
      const double along = -a.dot(delta) / delta.squaredNorm();
      if (along < 0. || along > 1. ||
          (a.cwiseProduct(range).cwiseAbs().array() <= accuracy.array()).all() ||
          (b.cwiseProduct(range).cwiseAbs().array() <= accuracy.array()).all() ||
          segment_distance(Eigen::Vector2d::Zero(), a, b) >
              std::max(engine.settings.node_tolerance, .5 * delta.norm()))
        continue;
      try {
        auto point = boundary_point_at(
            engine, line, line.points[k - 1].phases, target,
            delta.normalized().cwiseQuotient(range), equilibrium_pt_tolerance());
        if (!point ||
            !same_phase_compositions(point->phases, endpoint.phases,
                                     engine.settings.composition_tolerance,
                                     engine.settings.amount_tolerance))
          continue;
        line.points.resize(k);
        line.points.push_back(endpoint);
        changed = true;
        break;
      } catch (const std::exception &) {
        // Keep a trace whose earlier branch cannot be verified at the event.
      }
    }
    if (front)
      std::reverse(line.points.begin(), line.points.end());
  }
  return changed;
}

// Chords are a quick comparison only when both traces already have matching
// neighbours. Otherwise compare corrected equilibria, including solution
// copies.
bool same_boundary_curve(Engine &engine, const Result &result,
                         const Boundary &first, const Boundary &second,
                         bool partial, bool require_equilibrium) {
  if (first.points.size() < 2 || second.points.size() < 2)
    return false;
  const auto ranges = result.coordinate_ranges();
  const Eigen::Vector2d origin(ranges[0][0], ranges[1][0]),
      range(ranges[0][1] - origin[0], ranges[1][1] - origin[1]);
  auto q = [&](const auto &p) -> Eigen::Vector2d {
    return (diagram_coordinates(p, result.section.type) - origin)
        .cwiseQuotient(range);
  };
  auto near = [&](const Boundary &one, const Boundary &two) {
    for (const auto &p : one.points) {
      double distance = std::numeric_limits<double>::infinity();
      for (std::size_t k = 1; k < two.points.size(); ++k)
        distance =
            std::min(distance, segment_distance(q(p), q(two.points[k - 1]),
                                                q(two.points[k])));
      if (distance > engine.settings.node_tolerance * 2.)
        return false;
    }
    return true;
  };
  const bool univariant = result.section.type == DiagramType::PT &&
                          boundary_variance(first, result) == 1 &&
                          boundary_variance(second, result) == 1;
  if (partial && (first.start_node < 0 || first.end_node < 0)) {
    if (!near(first, second))
      return false;
    // A corrector can extend the same reaction beyond a completed record.
    // That does not prove the accepted tail is covered by the saved curve.
    for (bool start : {true, false}) {
      if ((start ? first.start_node : first.end_node) >= 0)
        continue;
      const auto p = q(start ? first.points.front() : first.points.back());
      bool covered = false;
      for (std::size_t k = 1; k < second.points.size(); ++k) {
        const auto a = q(second.points[k - 1]), b = q(second.points[k]);
        const Eigen::Vector2d delta = b - a;
        if (delta.squaredNorm() == 0.)
          continue;
        const double along = (p - a).dot(delta) / delta.squaredNorm();
        const double roundoff = 64. * std::numeric_limits<double>::epsilon();
        covered = covered || (along >= -roundoff && along <= 1. + roundoff &&
                              segment_distance(p, a, b) <=
                                  engine.settings.node_tolerance * 2.);
      }
      if (!covered)
        return false;
    }
  }
  if (!univariant && !partial && !require_equilibrium && near(first, second) &&
      near(second, first))
    return true;
  auto verified = [&](const Boundary &one, const Boundary &two) {
    if (one.points.size() < 2 || two.points.empty())
      return false;
    double reach = engine.settings.step * 3.;
    for (std::size_t k = 1; k < two.points.size(); ++k)
      reach = std::max(reach, (q(two.points[k]) - q(two.points[k - 1])).norm());
    std::vector<BoundaryPoint> inserted;
    if (one.points.size() == 2)
      try {
        Eigen::Vector2d tangent = q(one.points.back()) - q(one.points.front());
        if (tangent.norm() == 0.)
          return false;
        const Eigen::Vector2d target =
            origin + .5 * (q(one.points.front()) + q(one.points.back()))
                              .cwiseProduct(range);
        auto midpoint = boundary_point_at(
            engine, one, one.points.front().phases, target,
            tangent.normalized().cwiseQuotient(range), range * tangent.norm());
        if (!midpoint)
          return false;
        inserted = {one.points.front(), std::move(*midpoint),
                    one.points.back()};
      } catch (const std::exception &) {
        return false;
      }
    int checked = 0;
    const auto &samples = inserted.empty() ? one.points : inserted;
    for (std::size_t i = partial ? 0 : 1;
         i < samples.size() - (partial ? 0 : 1); ++i) {
      const auto &target = samples[i];
      if (solution_separation(target.phases) < .002 || two.points.empty())
        continue;
      const auto closest =
          std::min_element(two.points.begin(), two.points.end(),
                           [&](const auto &a, const auto &b) {
                             return (q(a) - q(target)).squaredNorm() <
                                    (q(b) - q(target)).squaredNorm();
                           });
      if ((q(*closest) - q(target)).norm() > reach)
        return false;
      Eigen::Vector2d tangent =
          q(samples[std::min(i + 1, samples.size() - 1)]) -
          q(samples[i ? i - 1 : 0]);
      if (tangent.norm() < 1.e-12 && partial)
        tangent = q(target) - q(samples[i ? i - 1 : 0]);
      if (tangent.norm() < 1.e-12)
        continue;
      try {
        if (univariant) {
          const auto weights = tangent.normalized().cwiseQuotient(range).eval();
          auto left = boundary_point_at(
              engine, one, target.phases,
              diagram_coordinates(target, result.section.type), weights,
              range * engine.settings.node_tolerance);
          if (!left)
            return false;
          auto warm = closest->phases;
          warm.erase(std::remove_if(warm.begin(), warm.end(),
                                    [&](const auto &p) {
                                      return std::find(two.assemblage.begin(),
                                                       two.assemblage.end(),
                                                       p.id) ==
                                             two.assemblage.end();
                                    }),
                     warm.end());
          auto right = boundary_point_at(
              engine, two, warm,
              diagram_coordinates(*left, result.section.type), weights,
              range * engine.settings.node_tolerance);
          // Match the independently corrected compositions before sharing a
          // seed. Otherwise a different intensive branch could be forced to
          // the reference root merely by copying its starting compositions.
          const bool same = right && same_phase_compositions(
                                         left->phases, right->phases,
                                         engine.settings.composition_tolerance,
                                         engine.settings.amount_tolerance);
          if (!same || ((diagram_coordinates(*right, result.section.type) -
                         diagram_coordinates(*left, result.section.type))
                            .cwiseAbs()
                            .array() > equilibrium_pt_tolerance().array())
                           .any()) {
            if (same)
              warm = right->phases;
            if (!seed_phase_compositions(warm, left->phases,
                                         engine.settings.composition_tolerance,
                                         engine.settings.amount_tolerance))
              return false;
            right = boundary_point_at(
                engine, two, warm,
                diagram_coordinates(*left, result.section.type), weights,
                equilibrium_pt_tolerance());
            if (!right ||
                !same_phase_compositions(left->phases, right->phases,
                                         engine.settings.composition_tolerance,
                                         engine.settings.amount_tolerance))
              return false;
          }
          ++checked;
          continue;
        }
        auto point =
            boundary_point_at(engine, two, closest->phases,
                              diagram_coordinates(target, result.section.type),
                              tangent.normalized().cwiseQuotient(range),
                              range * engine.settings.node_tolerance);
        if (!point ||
            (q(*point) - q(target)).norm() > engine.settings.node_tolerance)
          return false;
        const bool same =
            partial ? same_boundary_compositions(
                          point->phases, two.zero_phase, target.phases,
                          one.zero_phase, engine.settings.composition_tolerance,
                          engine.settings.amount_tolerance)
                    : same_phase_compositions(
                          point->phases, target.phases,
                          engine.settings.composition_tolerance * 100.,
                          engine.settings.amount_tolerance);
        if (!same)
          return false;
        ++checked;
      } catch (const std::exception &) {
        return false;
      }
    }
    return checked > 0;
  };
  return verified(first, second) && (partial || verified(second, first));
}
} // namespace burnman::pseudosections::detail
