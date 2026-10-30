# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Numerical helpers used by the anisotropic contributions."""

import numpy as np


def is_positive_definite(matrix):
    """Whether a matrix has strictly positive eigenvalues, following BurnMan."""
    return bool(np.all(np.linalg.eigvals(matrix) > 0.0))
