#pragma once
#include "burnman/core/mineral.hpp"
namespace burnman::minerals {
/// Ideal-gas thermal reference used with the PS1994 pressure-dependent EOS.
enum class WaterThermalReference { HollandPowell2011, NIST };

/// Ideal-gas properties at the 1 bar standard pressure, in SI molar units.
struct WaterIdealGasProperties {
  double enthalpy;        ///< J/mol, on the enthalpy-of-formation scale.
  double entropy;         ///< J/(mol K), absolute entropy.
  double heat_capacity_p; ///< J/(mol K).
  double gibbs;           ///< J/mol, H - T S.
};

/// Evaluate the thermal reference independently of the PS1994 fluid domain.
/// The Holland-Powell reference includes its 298.15 K standard state. The
/// NIST Shomate fit is intended for 500-1700 K. Requires finite positive T.
WaterIdealGasProperties water_ideal_gas_reference(
    double temperature,
    WaterThermalReference reference = WaterThermalReference::HollandPowell2011);

/// Pitzer & Sterner (1994) H2O fluid, with the Holland & Powell (2011)
/// Table 2a ideal-gas thermal reference (H_0 = -241810 J/mol).
/// Supports 500 <= T <= 1700 K and 0 < P <= 5 GPa. Selects the stable
/// density root. Thermodynamic derivatives are in SI.
std::shared_ptr<Mineral> water_fluid();

/// Select an alternative thermal reference without changing the PS1994 EOS.
std::shared_ptr<Mineral> water_fluid(WaterThermalReference reference);
} // namespace burnman::minerals
