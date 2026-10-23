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

#pragma once

#include "burnman/core/seismic.hpp"
#include <Eigen/Core>
#include <limits>
#include <memory>
#include <optional>

namespace burnman {
class Material;
}

namespace burnman::geotherm {

/** Tabulated depth-temperature geotherm.
 * Depths use metres and temperatures use kelvin. Coordinates must be finite,
 * nonempty and sorted in increasing order; repeated coordinates are allowed.
 * Queries outside the tabulation throw. At a repeated coordinate the shallower
 * row is selected.
 */
class Geotherm {
public:
  Geotherm(const Eigen::ArrayXd &depths_array,
           const Eigen::ArrayXd &temperatures_array);
  virtual ~Geotherm() = default;

  Eigen::ArrayXd depths_array, temperatures_array;
  double minimum_depth = std::numeric_limits<double>::quiet_NaN();
  double maximum_depth = std::numeric_limits<double>::quiet_NaN();

  virtual Eigen::ArrayXd temperatures(const Eigen::ArrayXd &depths) const;
  double temperatures(double depth) const;

protected:
  Geotherm() = default;
  explicit Geotherm(const Eigen::ArrayXXd &table);
};

class BrownShankland : public Geotherm {
public:
  BrownShankland();
};
class Anderson : public Geotherm {
public:
  Anderson();
};
class StaceyContinental : public Geotherm {
public:
  StaceyContinental();
};
class StaceyOceanic : public Geotherm {
public:
  StaceyOceanic();
};
class Katsura2022 : public Geotherm {
public:
  Katsura2022();
};
class Plesa2022Mars6cm3 : public Geotherm {
public:
  Plesa2022Mars6cm3();
};

/** Pressure-temperature geotherm with a depth-to-pressure model.
 * A null constructor argument selects the shared seismic::prem_model().
 * Only temperatures_from_pressures() uses the pressure tabulation directly;
 * temperatures(depths) first calls depth_to_pressure_model->pressure(depths).
 */
class GeothermFromPressures : public Geotherm {
public:
  GeothermFromPressures(const Eigen::ArrayXd &pressures_array,
                        const Eigen::ArrayXd &temperatures_array,
                        std::shared_ptr<seismic::Seismic1DModel>
                            depth_to_pressure_model = nullptr);

  Eigen::ArrayXd pressures_array;
  std::shared_ptr<seismic::Seismic1DModel> depth_to_pressure_model;
  double minimum_pressure = std::numeric_limits<double>::quiet_NaN();
  double maximum_pressure = std::numeric_limits<double>::quiet_NaN();

  using Geotherm::temperatures;
  Eigen::ArrayXd temperatures(const Eigen::ArrayXd &depths) const override;
  virtual Eigen::ArrayXd
  temperatures_from_pressures(const Eigen::ArrayXd &pressures) const;
  double temperatures_from_pressures(double pressure) const;

protected:
  explicit GeothermFromPressures(
      std::shared_ptr<seismic::Seismic1DModel> depth_to_pressure_model);
  GeothermFromPressures(
      const Eigen::ArrayXXd &table,
      std::shared_ptr<seismic::Seismic1DModel> depth_to_pressure_model);
};

class Anzellini2013 : public GeothermFromPressures {
public:
  explicit Anzellini2013(std::shared_ptr<seismic::Seismic1DModel>
                             depth_to_pressure_model = nullptr);
};

/** Material adiabat anchored at (P_0, T_0), with constant entropy of the bulk.
 * The material is shared and its state is set to the anchor on evaluation,
 * whenever it is evaluated. Ordinary materials retain their composition;
 * EquilibratedComposite re-minimises at each state. For an Assemblage, entropy
 * includes n_moles so reaction-induced changes in formula-unit counts do not
 * change the prescribed entropy. Concurrent evaluations of a shared material
 * require external synchronisation.
 */
class AdiabaticGeotherm : public GeothermFromPressures {
public:
  AdiabaticGeotherm(std::shared_ptr<Material> material, double T_0, double P_0,
                    std::shared_ptr<seismic::Seismic1DModel>
                        depth_to_pressure_model = nullptr);

  std::shared_ptr<Material> material;
  double T_0, P_0;
  using GeothermFromPressures::temperatures_from_pressures;
  Eigen::ArrayXd
  temperatures_from_pressures(const Eigen::ArrayXd &pressures) const override;
};

/// Solve constant bulk entropy using a gamma*T/K_S estimate and a bracketed
/// root. P_0 defaults to the first requested pressure. Requests may be
/// unordered or repeated. The material is left at (P_0, T_0); empty requests
/// leave it unchanged.
Eigen::ArrayXd adiabatic_profile(const Eigen::ArrayXd &pressures,
                                 Material &rock, double T_0,
                                 std::optional<double> P_0 = std::nullopt);

} // namespace burnman::geotherm
