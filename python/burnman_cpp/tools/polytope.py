# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""Native cddlib/GMP vertex enumeration and solution simplification."""

from .._core import (
    MaterialPolytope,
    composite_polytope_at_constrained_composition,
    simplify_composite_with_composition,
    solution_polytope_from_endmember_occupancies,
    transform_solution_to_new_basis,
)

__all__ = [
    "MaterialPolytope",
    "composite_polytope_at_constrained_composition",
    "simplify_composite_with_composition",
    "solution_polytope_from_endmember_occupancies",
    "transform_solution_to_new_basis",
]
