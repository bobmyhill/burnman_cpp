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
  int sign = phase_entry_side(engine, a, ids, zero, s, range, normal, full_side);
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
} // namespace burnman::pseudosections::detail
