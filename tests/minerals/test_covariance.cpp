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

#include "burnman/minerals/datasets.hpp"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

using namespace burnman::minerals;

TEST_CASE("Endmember energy covariances retain names and cross correlations",
          "[native_datasets][covariance]") {
  using Loader = CovarianceData (*)();
  const std::array<Loader, 3> loaders{HP11::cov, HGP18::cov, HPx_ds636::cov};
  const std::array<Eigen::Index, 3> sizes{256, 289, 291};
  for (std::size_t k = 0; k < loaders.size(); ++k) {
    auto first = loaders[k]();
    auto second = loaders[k]();
    const auto &names = first.endmember_names;
    const auto &matrix = first.covariance_matrix;
    REQUIRE(matrix.rows() == sizes[k]);
    REQUIRE(matrix.cols() == sizes[k]);
    REQUIRE(names.size() == static_cast<std::size_t>(sizes[k]));
    CHECK(matrix.allFinite());
    CHECK((matrix - matrix.transpose()).norm() == 0.);
    const auto ky = std::find(names.begin(), names.end(), "ky");
    const auto sill = std::find(names.begin(), names.end(), "sill");
    REQUIRE(ky != names.end());
    REQUIRE(sill != names.end());
    const auto i = ky - names.begin(), j = sill - names.begin();
    CHECK(matrix(i, i) + matrix(j, j) - 2. * matrix(i, j) ==
          (k == 2 ? 5600. : 5200.));
    first.covariance_matrix.setZero();
    first.endmember_names[0] = "changed";
    CHECK(second.endmember_names[0] == "fo");
    CHECK(second.covariance_matrix(0, 0) > 0.);
  }
}
