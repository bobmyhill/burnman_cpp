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

#include "bindings.hpp"
#include "burnman/core/combined_mineral.hpp"

namespace burnman::python {
void bind_combined(py::module_ &m) {
  m.def(
      "CombinedMineral",
      [](const std::vector<Mineral> &minerals, const Eigen::ArrayXd &amounts,
         const Eigen::Vector3d &adjustment, const std::string &name) {
        return std::make_shared<Mineral>(
            make_combined_mineral(minerals, amounts, adjustment, name));
      },
      py::arg("minerals"), py::arg("molar_amounts"),
      py::arg("energy_adjustment") = Eigen::Vector3d::Zero().eval(),
      py::arg("name") = "Combined mineral",
      "Create a native mineral from a signed linear combination of "
      "endmembers.");
}
} // namespace burnman::python
