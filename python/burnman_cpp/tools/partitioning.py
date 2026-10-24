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

"""Iron partitioning between ferropericlase and bridgmanite (SI units)."""

from .._core.partitioning import calculate_nakajima_fp_pv_partition_coefficient

__all__ = ["calculate_nakajima_fp_pv_partition_coefficient"]
