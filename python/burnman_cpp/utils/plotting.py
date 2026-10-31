# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Polygon centres using the native pole-of-inaccessibility algorithm."""

from .._core import visual_center_of_polygon as _native_visual_center


def visual_center_of_polygon(polygon_rings, precision=1.0, with_distance=False):
    """Return the polygon centre, optionally with its distance to the edges."""
    centre, distance = _native_visual_center(polygon_rings, precision)
    return (centre, distance) if with_distance else centre
