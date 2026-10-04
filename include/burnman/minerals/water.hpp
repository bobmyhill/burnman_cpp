#pragma once
#include "burnman/core/mineral.hpp"
namespace burnman::minerals {
/// Pitzer & Sterner (1994) H2O fluid, with NIST ideal-gas thermal reference.
/// Supports 500 <= T <= 1700 K and 0 < P <= 5 GPa. Selects the stable
/// mechanically stable density root. Thermodynamic derivatives are in SI.
std::shared_ptr<Mineral> water_fluid();
} // namespace burnman::minerals
