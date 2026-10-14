// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.

#include "burnman/tools/pseudosection/common.hpp"
#include <cmath>
#include <map>

namespace burnman::pseudosections::detail {
bool intersects_region(const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                       const std::array<std::array<double, 2>, 2> &ranges,
                       double parallel_tolerance) {
  double first = 0., last = 1.;
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    const auto &bounds = ranges[static_cast<std::size_t>(axis)];
    const double delta = b[axis] - a[axis];
    if (delta == 0. || std::abs(delta) < parallel_tolerance) {
      if (a[axis] < bounds[0] || a[axis] > bounds[1])
        return false;
    } else {
      double lo = (bounds[0] - a[axis]) / delta,
             hi = (bounds[1] - a[axis]) / delta;
      first = std::max(first, std::min(lo, hi));
      last = std::min(last, std::max(lo, hi));
    }
  }
  return first <= last;
}

PhaseStates average_phase_states(PhaseStates states, const PhaseStates &other) {
  for (auto &state : states) {
    const auto match = find_phase(other, state.id);
    if (match == other.end())
      continue;
    const double amount = state.amount + match->amount;
    if (amount > 0.)
      state.composition = (state.amount * state.composition +
                           match->amount * match->composition) /
                          amount;
    else
      state.composition = .5 * (state.composition + match->composition);
    state.amount = .5 * amount;
  }
  return states;
}

std::vector<int> canonical_phase_ids(const std::vector<int> &ids, int copies) {
  std::map<int, int> counts;
  for (int id : ids)
    ++counts[id / copies];
  std::vector<int> canonical;
  for (auto [base, count] : counts)
    for (int copy = 0; copy < count; ++copy)
      canonical.push_back(base * copies + copy);
  return canonical;
}

double solution_separation(const PhaseStates &states) {
  double minimum = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < states.size(); ++i)
    for (std::size_t j = i + 1; j < states.size(); ++j)
      if (states[i].candidate_index == states[j].candidate_index)
        minimum = std::min(
            minimum, (states[i].composition - states[j].composition).norm());
  return minimum;
}
} // namespace burnman::pseudosections::detail
