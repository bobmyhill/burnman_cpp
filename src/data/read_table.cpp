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

#include "read_table.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace burnman::data {
Eigen::ArrayXXd read_table(const char *text, Eigen::Index columns) {
  std::istringstream input(text);
  std::vector<double> values;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line.substr(0, line.find('#')));
    double value;
    std::vector<double> parsed;
    while (row >> value)
      parsed.push_back(value);
    if (parsed.empty())
      continue;
    if (static_cast<Eigen::Index>(parsed.size()) != columns)
      throw std::runtime_error("Invalid bundled seismic table.");
    values.insert(values.end(), parsed.begin(), parsed.end());
  }
  const auto rows = static_cast<Eigen::Index>(values.size()) / columns;
  return Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(values.data(), rows,
                                                         columns);
}

Eigen::ArrayXXd read_table(const char *text) {
  std::istringstream input(text);
  std::vector<std::pair<double, double>> values;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line.substr(0, line.find('#')));
    double coordinate, temperature;
    if (row >> coordinate >> temperature)
      values.emplace_back(coordinate, temperature);
  }
  Eigen::ArrayXXd table(static_cast<Eigen::Index>(values.size()), 2);
  for (Eigen::Index i = 0; i < table.rows(); ++i) {
    table(i, 0) = values[static_cast<std::size_t>(i)].first;
    table(i, 1) = values[static_cast<std::size_t>(i)].second;
  }
  return table;
}
} // namespace burnman::data
