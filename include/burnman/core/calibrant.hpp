// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#pragma once

#include <Eigen/Core>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace burnman {
/** Pressure standard with the Python BurnMan Calibrant calling convention.
 * All public quantities are SI. Catalogue parameter dictionaries retain the
 * published/native parameter units used by their pressure functions.
 */
class Calibrant {
public:
  using Params = std::map<std::string, double>;
  using Function = std::function<double(double, double, const Params &)>;
  using CovarianceResult = std::pair<double, Eigen::Matrix3d>;

  Calibrant(Function calibrant_function,
            const std::string &calibrant_function_return_type, Params params);
  Params params;
  const std::string &return_type() const { return return_type_; }
  double calibrant_function(double value, double temperature,
                            const Params &parameters) const;
  double pressure_function(double volume, double temperature,
                           const Params &parameters) const;
  double volume_function(double pressure, double temperature,
                         const Params &parameters) const;
  double pressure(double volume, double temperature) const;
  double volume(double pressure, double temperature) const;
  /// Input order [V,T], output order [P,V,T].
  CovarianceResult pressure(double volume, double temperature,
                            const Eigen::Matrix2d &VT_covariance) const;
  /// Input order [P,T], output order [V,P,T].
  CovarianceResult volume(double pressure, double temperature,
                          const Eigen::Matrix2d &PT_covariance) const;

private:
  Function function_;
  std::string return_type_;
};
} // namespace burnman
