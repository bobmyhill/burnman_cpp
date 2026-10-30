# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Native Voigt tensor conversions and polar decomposition."""

from .._core import (
    contract_compliances,
    contract_stiffnesses,
    contract_strains,
    contract_stresses,
    expand_strains,
    expand_stresses,
    polar_decomposition,
    voigt_array_from_cijs,
    voigt_index_to_ij,
    voigt_notation_to_compliance_tensor,
    voigt_notation_to_stiffness_tensor,
)
