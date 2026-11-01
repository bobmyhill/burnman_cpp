// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include "burnman/utils/types/mineral_params.hpp"

namespace burnman::eos::detail {
// Internal reference states, shared with the original scientific regression
// tests.
struct AAReference {
  double energy, temperature, pressure, bulk_modulus;
};
AAReference aa_reference(double volume, const types::MineralParams &params);

struct BroshCaloric {
  double gibbs, entropy, heat_capacity;
};
BroshCaloric brosh_sgte(double temperature, const types::MineralParams &params);
struct BroshThermal {
  double theta, interpolation;
};
BroshThermal brosh_thermal(double pressure, const types::MineralParams &params);
} // namespace burnman::eos::detail
