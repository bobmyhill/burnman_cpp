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

import numpy as np
import pytest

import burnman_cpp as bm


def rectangle(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_visual_center_of_square():
    point, distance = bm.visual_center_of_polygon([rectangle(0, 0, 2, 2)])
    assert point == pytest.approx([1.0, 1.0], abs=1e-2)
    assert distance == pytest.approx(1.0, abs=1e-2)


def test_visual_center_avoids_hole():
    rings = [rectangle(0, 0, 10, 10), rectangle(4, 4, 6, 6)]
    point, distance = bm.visual_center_of_polygon(rings, precision=1e-6)
    assert bm.signed_distance_to_polygon(point, rings) == pytest.approx(distance)
    assert distance == pytest.approx(4 * np.sqrt(2) / (1 + np.sqrt(2)), abs=1e-3)


def test_visual_center_rejects_degenerate_polygon():
    with pytest.raises(ValueError):
        bm.visual_center_of_polygon([np.zeros((2, 2))])
