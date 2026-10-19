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

#include "burnman/tools/phase_diagram/continuation.hpp"
#include "burnman/tools/equilibration/equality_constraint_variants.hpp"
#include "burnman/tools/equilibration/equilibrate_objective.hpp"
#include "burnman/tools/phase_diagram/boundary_repair.hpp"
#include "burnman/tools/phase_diagram/domain.hpp"
#include "burnman/tools/phase_diagram/stability.hpp"
#include "burnman/tools/polytope.hpp"
#include "burnman/utils/index_utils.hpp"

namespace burnman::phase_diagrams::detail {
Eigen::Vector2d equilibrium_coordinate_tolerance(DiagramType type,
                                                 const Eigen::Vector2d &range) {
  Eigen::Vector2d tolerance;
  const auto axes = diagram_axes(type);
  for (int i = 0; i < 2; ++i)
    switch (axes[static_cast<std::size_t>(i)]) {
    case Coordinate::P:
      tolerance[i] = equilibrium_pt_tolerance()[0];
      break;
    case Coordinate::T:
      tolerance[i] = equilibrium_pt_tolerance()[1];
      break;
    case Coordinate::S:
    case Coordinate::V:
      tolerance[i] = std::max(1.e-18, std::abs(range[i]) * 1.e-9);
      break;
    case Coordinate::X:
      tolerance[i] = 1.e-10;
      break;
    }
  return tolerance;
}
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
  // Latent S/V intervals can couple phase amounts to the intensive variables
  // through a small but nonzero singular direction. Dropping it turns a
  // regular field into a spurious reaction replacement. Use the same inverse
  // rank accuracy as continuation_tangent and retain the residual check below.
  svd.setThreshold(engine.direct_coordinates() ? 1.e-10 : 1.e-12);
  if (svd.rank() != j.cols())
    return {};
  Eigen::VectorXd response = svd.solve(rhs);
  if (!response.allFinite() || (j * response - rhs).norm() > 1.e-7 * rhs.norm())
    return {};
  return scales.cwiseProduct(response);
}

int phase_entry_side(Engine &engine, Assemblage &a, const std::vector<int> &ids,
                     int zero, const EquilibriumSolve &s,
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
  int response_sign = 0;
  if (response.size()) {
    const Eigen::Index amount = prm.phase_amount_indices[entry - ids.begin()];
    const double change = response[amount] / total;
    if (std::abs(change) > 1.e-9)
      response_sign = change > 0. ? 1 : -1;
  }
  full_side = response_sign;
  if (!response_sign) {
    // An arbitrarily thin regular field can have an ill-conditioned full
    // response while its absent branch remains regular. Dependent phase
    // amounts instead need the compound-reaction limiting states below.
    const auto formulae = engine.phase_formulae(ids, states);
    Eigen::FullPivLU<Eigen::MatrixXd> rank(formulae);
    rank.setThreshold(1.e-9);
    if (rank.rank() < formulae.rows())
      return 0;
  }
  auto absent = ids;
  absent.erase(absent.begin() + (entry - ids.begin()));
  auto other =
      engine.make_at(absent, states, engine.coordinates(a),
                     Eigen::Vector2d(a.get_pressure(), a.get_temperature()));
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
  const auto opposite = field_response(engine, *other, j, range, normal);
  if (!opposite.size())
    return 0;

  // Express the entering phase's elemental formula as a combination of the
  // absent branch's endmembers. Their partial-Gibbs derivatives then give its
  // affinity derivative without subtracting large Gibbs energies or stepping
  // outside a nearly empty composition face. This is independent of the
  // chemical-potential gauge when the formula lies in the endmember span.
  const auto stoich = other->get_stoichiometric_matrix();
  const auto elements = other->get_elements();
  Eigen::MatrixXd formulae(stoich.rows(), engine.components.size());
  const auto formula = engine.phase_formulae({zero}, states).row(0).eval();
  Eigen::VectorXd target(engine.components.size());
  for (std::size_t k = 0; k < engine.components.size(); ++k) {
    const auto it = std::find(
        elements.begin(), elements.end(),
        engine.elements[static_cast<std::size_t>(engine.components[k])]);
    if (it == elements.end())
      formulae.col(static_cast<Eigen::Index>(k)).setZero();
    else
      formulae.col(static_cast<Eigen::Index>(k)) =
          stoich.col(it - elements.begin());
    target[static_cast<Eigen::Index>(k)] = formula[engine.components[k]];
  }
  const Eigen::VectorXd weights =
      formulae.transpose().completeOrthogonalDecomposition().solve(target);
  if ((formulae.transpose() * weights - target).norm() >
      1.e-10 * std::max(1., target.norm()))
    return 0;
  Eigen::VectorXd derivatives(stoich.rows());
  Eigen::Index offset = 0;
  const auto counts = other->get_endmembers_per_phase();
  for (std::size_t k = 0; k < counts.size(); ++k) {
    const Eigen::Index n = counts[k];
    const auto phase = other->get_phase(k);
    if (n == 1)
      derivatives[offset] = phase->get_molar_volume() * opposite[0] -
                            phase->get_molar_entropy() * opposite[1];
    else {
      const auto solution = other->get_phase<Solution>(k);
      Eigen::VectorXd change(n);
      change.tail(n - 1) = opposite.segment(
          other_prm.phase_amount_indices[static_cast<Eigen::Index>(k)] + 1,
          n - 1);
      change[0] = -change.tail(n - 1).sum();
      derivatives.segment(offset, n) =
          solution->get_partial_volumes().matrix() * opposite[0] -
          solution->get_partial_entropies().matrix() * opposite[1] +
          solution->get_gibbs_hessian().matrix() * change;
    }
    offset += n;
  }
  const auto material =
      a.get_phase(static_cast<std::size_t>(entry - ids.begin()));
  const double derivative = material->get_molar_volume() * opposite[0] -
                            material->get_molar_entropy() * opposite[1] -
                            weights.dot(derivatives);
  if (!std::isfinite(derivative) || std::abs(derivative) <= 1.e-5)
    return 0;
  const int sign = derivative < 0. ? 1 : -1;
  if (response_sign && response_sign != sign)
    return 0;
  const auto slack =
      (other_prm.constraint_matrix * x + other_prm.constraint_vector).eval();
  const auto rate = (other_prm.constraint_matrix * (-sign * opposite)).eval();
  for (Eigen::Index row = 0; row < slack.size(); ++row)
    if (slack[row] >= -1.e-12 && rate[row] > 1.e-12)
      return 0;
  full_side = sign;
  return sign;
}

// At an intensive invariant, changing reaction extents gives a latent S/V
// edge. Its full assemblage need not be an areal field in the selected axes.
// Test its limiting assemblages' coordinate responses and the affinity of
// each omitted phase, instead of assigning a field by phase count or by a
// finite probe that can jump across a very narrow neighbour.
std::array<std::vector<int>, 2>
latent_boundary_sides(Engine &engine, const Assemblage &a,
                      const std::vector<int> &ids, const Eigen::Vector2d &range,
                      const Eigen::Vector2d &normal) {
  std::array<std::vector<int>, 2> sides;
  std::array<bool, 2> conflict{};
  const auto states = engine.snapshot(a, ids);
  for (std::size_t removed = 0; removed < ids.size(); ++removed) {
    auto subset = ids;
    subset.erase(subset.begin() + static_cast<std::ptrdiff_t>(removed));
    try {
      auto work = engine.fixed_at(
          subset, states, engine.coordinates(a),
          Eigen::Vector2d(a.get_pressure(), a.get_temperature()));
      if (!work.state.success || work.ids.size() != subset.size())
        continue;
      auto &field = *work.assemblage;
      const auto prm = engine.parameters(field);
      const auto x =
          get_parameter_vector(field, engine.free_vectors.empty() ? 0 : 1);
      ConstraintGroup fixed;
      for (Coordinate axis : diagram_axes(engine.section.type))
        fixed.push_back(engine.coordinate_constraint(axis, 0., x.size()));
      if (!engine.free_vectors.empty())
        fixed.push_back(engine.coordinate_constraint(
            *engine.section.fixed_coordinate, 0., x.size()));
      const auto jacobian =
          J(x, field, fixed, prm.reduced_free_composition_vectors);
      auto response = field_response(engine, field, jacobian, range, normal);
      if (!response.size())
        continue;
      auto material = clone(a.get_phase(removed));
      const auto formula =
          engine.phase_formulae({ids[removed]}, states).row(0).eval();
      auto affinity = [&]() {
        material->set_state(field.get_pressure(), field.get_temperature());
        const auto mu = engine.potentials(field);
        double value = material->get_molar_gibbs();
        for (std::size_t k = 0; k < engine.components.size(); ++k)
          value -=
              formula[engine.components[k]] * mu[static_cast<Eigen::Index>(k)];
        return value;
      };
      const double initial = affinity();
      for (int sign : {-1, 1}) {
        set_composition_and_state_from_parameters(field, x);
        const auto direction = (sign * response).eval();
        const auto slack =
            (prm.constraint_matrix * x + prm.constraint_vector).eval();
        const auto rate = (prm.constraint_matrix * direction).eval();
        bool feasible = true;
        double step = 1.e-6;
        for (Eigen::Index row = 0; row < slack.size(); ++row)
          if (rate[row] > 1.e-12) {
            if (slack[row] >= -1.e-12) {
              feasible = false;
              break;
            }
            step = std::min(step, -.1 * slack[row] / rate[row]);
          }
        if (!feasible || step < 1.e-10)
          continue;
        set_composition_and_state_from_parameters(field, x + step * direction);
        const double derivative = (affinity() - initial) / step;
        if (!std::isfinite(derivative) || derivative <= 1.e-5)
          continue;
        const auto branch = engine.canonical_ids(subset);
        const std::size_t slot = sign > 0 ? 1 : 0;
        if (!sides[slot].empty() && sides[slot] != branch)
          conflict[slot] = true;
        else
          sides[slot] = branch;
      }
    } catch (const std::exception &) {
    }
  }
  for (std::size_t slot = 0; slot < sides.size(); ++slot)
    if (conflict[slot])
      sides[slot].clear();
  return sides;
}

} // namespace

std::array<std::vector<int>, 2>
boundary_sides(Engine &engine, Assemblage &a, const std::vector<int> &ids,
               int zero, const EquilibriumSolve &s,
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
    else if (!has_composition_axis(engine.section.type))
      partial = latent_boundary_sides(engine, a, ids, range, normal);
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
      auto limit = engine.make_at(
          present, warm, engine.coordinates(a),
          Eigen::Vector2d(a.get_pressure(), a.get_temperature()));
      auto &state = (*limiting_states)[static_cast<std::size_t>(
          slope > 0. ? 1 - end : end)];
      engine.record_coordinates(state, *limit);
      state.phases = engine.snapshot(*limit, present);
      state.gibbs = limit->get_n_moles() * limit->get_molar_gibbs();
      state.mass_balance_error = engine.mass_error(*limit);
      state.equilibrium_error = reaction_residual(*limit);
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
                                     const EquilibriumSolve &s,
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
  const Eigen::MatrixXd projected = range.cwiseInverse().asDiagonal() *
                                    engine.coordinate_jacobian(a, s.x.size()) *
                                    scales.asDiagonal() * null;
  Eigen::JacobiSVD<Eigen::MatrixXd> projection(
      projected, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
phase_matches(const PhaseStates &a, const PhaseStates &b, double tolerance) {
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

bool seed_phase_compositions(PhaseStates &warm, const PhaseStates &seed,
                             double tolerance, double amount_tolerance) {
  const auto matches =
      phase_matches(warm, seed, std::numeric_limits<double>::infinity());
  if (!matches)
    return false;
  const double warm_total = total_amount(warm), seed_total = total_amount(seed);
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
  for (std::size_t i = 0; i < warm.size(); ++i) {
    warm[i].composition = seed[(*matches)[i]].composition;
    warm[i].composition_basis = seed[(*matches)[i]].composition_basis;
  }
  return true;
}

bool share_composition_faces(PhaseStates &first, PhaseStates &second) {
  const auto matches =
      phase_matches(first, second, std::numeric_limits<double>::infinity());
  if (!matches || first.size() != second.size())
    return false;
  for (std::size_t k = 0; k < first.size(); ++k) {
    auto &a = first[k], &b = second[(*matches)[k]];
    const auto n = a.composition.size();
    if (n <= 1)
      continue;
    Eigen::MatrixXd basis;
    if (!a.composition_basis || !b.composition_basis)
      basis = Eigen::MatrixXd::Identity(n, n);
    else {
      const auto &left = *a.composition_basis, &right = *b.composition_basis;
      if (left.cols() != n || right.cols() != n)
        return false;
      Eigen::MatrixXd joined(left.rows() + right.rows(), n);
      joined.topRows(left.rows()) = left;
      joined.bottomRows(right.rows()) = right;
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(joined.transpose());
      qr.setThreshold(1.e-10);
      basis.resize(qr.rank(), n);
      for (Eigen::Index row = 0; row < basis.rows(); ++row)
        basis.row(row) = joined.row(qr.colsPermutation().indices()[row]);
    }
    a.composition_basis = b.composition_basis = std::move(basis);
  }
  return true;
}

std::optional<PhaseStates> reaction_seed_amounts(const Engine &engine,
                                                 const std::vector<int> &ids,
                                                 const PhaseStates &warm,
                                                 int zero) {
  auto present = ids;
  present.erase(std::remove(present.begin(), present.end(), zero),
                present.end());
  if (present.empty())
    return std::nullopt;
  const auto formulae = engine.phase_formulae(present, warm);
  Eigen::MatrixXd reduced(formulae.rows(), engine.components.size());
  for (std::size_t k = 0; k < engine.components.size(); ++k)
    reduced.col(static_cast<Eigen::Index>(k)) =
        formulae.col(engine.components[k]);
  try {
    const auto lp = polytope::gibbs_linear_program(
        reduced, Eigen::VectorXd::Zero(reduced.rows()), engine.b);
    auto states = warm;
    for (auto &phase : states) {
      const auto it = std::find(present.begin(), present.end(), phase.id);
      phase.amount =
          it == present.end() ? 0. : lp.amounts[it - present.begin()];
    }
    return states;
  } catch (const polytope::InfeasibleBulk &) {
    return std::nullopt;
  }
}

bool same_phase_compositions(const PhaseStates &a, const PhaseStates &b,
                             double tolerance, double amount_tolerance) {
  if (a.size() > b.size())
    return same_phase_compositions(b, a, tolerance, amount_tolerance);
  const auto matches = phase_matches(a, b, tolerance);
  if (!matches)
    return false;
  const double total = total_amount(b);
  for (std::size_t j = 0; j < b.size(); ++j)
    if (std::find(matches->begin(), matches->end(), j) == matches->end() &&
        b[j].amount > amount_tolerance * total)
      return false;
  return true;
}

bool same_boundary_compositions(const PhaseStates &a, int zero_a,
                                const PhaseStates &b, int zero_b,
                                double tolerance, double amount_tolerance) {
  const auto x = find_phase(a, zero_a), y = find_phase(b, zero_b);
  return x != a.end() && y != b.end() &&
         x->candidate_index == y->candidate_index &&
         x->composition.size() == y->composition.size() &&
         (x->composition - y->composition).norm() < tolerance &&
         same_phase_compositions(a, b, tolerance, amount_tolerance);
}

bool same_solution_branches(const Engine &engine, const PhaseStates &before,
                            const PhaseStates &after, double pressure,
                            double temperature) {
  const double before_tolerance =
      engine.settings.amount_tolerance * total_amount(before);
  const double after_tolerance =
      engine.settings.amount_tolerance * total_amount(after);
  for (const auto &phase : before) {
    const auto next = find_phase(after, phase.id);
    if (next == after.end() || std::abs(phase.amount) > before_tolerance ||
        std::abs(next->amount) > after_tolerance)
      continue;
    if (phase.candidate_index != next->candidate_index ||
        phase.composition.size() != next->composition.size() ||
        !engine.same_composition_basin(phase.candidate_index, phase.composition,
                                       next->composition, pressure,
                                       temperature))
      return false;
  }
  return true;
}

std::vector<int> zero_phases_at(const Node &node, const PhaseStates &states,
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

std::optional<BoundaryPoint>
corrected_point(Engine &engine, Assemblage &a, const std::vector<int> &ids,
                ConstraintList &c, const Eigen::Vector2d &target,
                const Eigen::Vector2d &allowance, double tolerance_scale,
                bool require_stability) {
  auto solved = engine.solve(a, c, true, tolerance_scale);
  if (!solved.success || !solved.x.allFinite() ||
      engine.mass_error(a) > engine.settings.mass_balance_tolerance ||
      !engine.locally_stable(a) || !engine.distinct(a, ids) ||
      ((engine.coordinates(a) - target).cwiseAbs().array() > allowance.array())
          .any())
    return std::nullopt;
  const double affinity = engine.stability(a);
  if (require_stability && affinity < -engine.settings.affinity_tolerance)
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
boundary_point_at(Engine &engine, const Boundary &line, const PhaseStates &warm,
                  const Eigen::Vector2d &target, const Eigen::Vector2d &weights,
                  const Eigen::Vector2d &allowance, double tolerance_scale,
                  bool require_stability) {
  // S/V do not determine a physical starting P,T without an inverse solve.
  // The nearest accepted boundary state supplies both, including active faces.
  const auto nearest = std::min_element(
      line.points.begin(), line.points.end(),
      [&](const auto &a, const auto &b) {
        return ((diagram_coordinates(a, engine.section.type) - target)
                    .cwiseProduct(weights))
                   .squaredNorm() <
               ((diagram_coordinates(b, engine.section.type) - target)
                    .cwiseProduct(weights))
                   .squaredNorm();
      });
  std::optional<Eigen::Vector2d> physical;
  if (nearest != line.points.end())
    physical = Eigen::Vector2d(nearest->pressure, nearest->temperature);
  auto assembly = engine.make_at(line.assemblage, warm, target, physical);
  const auto zero = std::find(line.assemblage.begin(), line.assemblage.end(),
                              line.zero_phase);
  if (zero == line.assemblage.end())
    return std::nullopt;
  std::optional<BoundaryPoint> point;
  for (int pass = 0; pass < 3; ++pass) {
    const auto prm = engine.parameters(*assembly);
    auto c = constraints(std::make_unique<PhaseFractionConstraint>(
                             zero - line.assemblage.begin(), 0., prm),
                         continuation_plane(engine, *assembly, prm.n_parameters,
                                            target, weights));
    point = corrected_point(engine, *assembly, line.assemblage, c, target,
                            allowance, tolerance_scale, require_stability);
    if (!point || tolerance_scale == 1. ||
        !engine.settings.active_solution_faces ||
        point->minimum_affinity >=
            -std::min(engine.settings.affinity_tolerance, 1.e-5))
      return point;
    // Before rejecting a weakly unstable arc, release a trace composition
    // that the ordinary continuation face may have pinned to zero.
    std::vector<Minimum> minima;
    engine.stability(*assembly, &minima);
    auto states = engine.snapshot(*assembly, line.assemblage);
    for (auto &phase : states) {
      const auto &minimum =
          minima[static_cast<std::size_t>(phase.candidate_index)];
      if (minimum.affinity <
              -std::min(engine.settings.affinity_tolerance, 1.e-5) &&
          (minimum.p - phase.composition).norm() < .05)
        phase.composition = minimum.p;
    }
    auto trial = engine.make_assemblage(line.assemblage, states,
                                        assembly->get_pressure(),
                                        assembly->get_temperature(), 1.e-9);
    bool changed = false;
    for (std::size_t k = 0; k < line.assemblage.size(); ++k) {
      auto old = engine.composition_basis(*assembly->get_phase(k),
                                          states[k].candidate_index),
           next = engine.composition_basis(*trial->get_phase(k),
                                           states[k].candidate_index);
      changed = changed || old.rows() != next.rows() ||
                old.cols() != next.cols() ||
                (old.rows() == next.rows() && old.cols() == next.cols() &&
                 (old - next).norm() > 1.e-8);
    }
    if (!changed)
      return point;
    engine.set_coordinate(*trial, engine.composition_coordinate(*assembly));
    assembly = trial;
  }
  return point;
}

// Use the same intensive seed (and active solution faces) for both events,
// while preserving each event's solution-copy identities and zero constraints.
std::optional<BoundaryPoint>
junction_point_at(Engine &engine, const Node &node, PhaseStates warm,
                  const PhaseStates &seed, const Eigen::Vector2d &allowance) {
  if (node.zero_phases.size() != 2)
    return std::nullopt;
  const auto zeros =
      zero_phases_at(node, warm, engine.settings.max_phase_instances);
  if (zeros.size() != 2)
    return std::nullopt;
  if (!seed_phase_compositions(warm, seed))
    return std::nullopt;
  for (auto &phase : warm) {
    if (std::find(zeros.begin(), zeros.end(), phase.id) != zeros.end())
      phase.amount = 0.;
  }
  const auto target = diagram_coordinates(node, engine.section.type);
  auto assembly =
      engine.make_at(node.assemblage, warm, target,
                     Eigen::Vector2d(node.pressure, node.temperature));
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
                         allowance, engine.direct_coordinates() ? 1. : 1.e-4);
}

// An event found behind a point accepted within the affinity tolerance must
// replace the overshot tail. Verify the event on the earlier branch before
// trimming; coordinate order alone is insufficient for curved traces.
void trim_boundary_endpoints(Boundary &line, Engine &engine,
                             const Eigen::Vector2d &range) {
  if (has_composition_axis(engine.section.type) || line.points.size() < 3 ||
      line.zero_phase < 0 ||
      (line.start_node >= 0 && line.start_node == line.end_node))
    return;
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
    const auto accuracy =
        equilibrium_coordinate_tolerance(engine.section.type, range);
    while (
        tail > 0 &&
        ((diagram_coordinates(line.points[tail], engine.section.type) - target)
             .cwiseAbs()
             .array() <= accuracy.array())
            .all())
      --tail;
    // Ordinary forward arrivals and repeated endpoint records need no solve.
    // Search the earlier branch only when attaching the event turns back.
    const bool turns_back =
        tail > 0 &&
        -q(line.points[tail])
                .dot(q(line.points[tail]) - q(line.points[tail - 1])) < 0.;
    for (std::size_t k = tail; turns_back && k > 0; --k) {
      const auto a = q(line.points[k - 1]), b = q(line.points[k]);
      const Eigen::Vector2d delta = b - a;
      if (delta.squaredNorm() == 0.)
        continue;
      const double along = -a.dot(delta) / delta.squaredNorm();
      if (along < 0. || along > 1. ||
          (a.cwiseProduct(range).cwiseAbs().array() <= accuracy.array())
              .all() ||
          (b.cwiseProduct(range).cwiseAbs().array() <= accuracy.array())
              .all() ||
          segment_distance(Eigen::Vector2d::Zero(), a, b) >
              std::max(engine.settings.node_tolerance, .5 * delta.norm()))
        continue;
      try {
        auto point = boundary_point_at(
            engine, line, line.points[k - 1].phases, target,
            delta.normalized().cwiseQuotient(range), accuracy);
        if (!point ||
            !same_phase_compositions(point->phases, endpoint.phases,
                                     engine.settings.composition_tolerance,
                                     engine.settings.amount_tolerance))
          continue;
        line.points.resize(k);
        line.points.push_back(endpoint);
        break;
      } catch (const std::exception &) {
        // Keep a trace whose earlier branch cannot be verified at the event.
      }
    }
    if (front)
      std::reverse(line.points.begin(), line.points.end());
  }
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
    return follows_polyline(one.points, two.points, q,
                            engine.settings.node_tolerance * 2.);
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
          erase_if(warm, [&](const auto &p) {
            return std::find(two.assemblage.begin(), two.assemblage.end(),
                             p.id) == two.assemblage.end();
          });
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
} // namespace burnman::phase_diagrams::detail
