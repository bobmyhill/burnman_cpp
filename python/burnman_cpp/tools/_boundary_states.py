# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Read verified field-side equilibria from the saved limiting curves."""

import numpy as np


def field_boundary_states(data):
    """Yield (boundary key, position, state) from this field's saved curve."""
    from .phase_diagram import _section_axes
    from .phase_diagram_labels import _limiting_curve

    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    boundaries = {b["id"]: b for b in data["boundaries"]}

    def sample(polygon):
        for line_id in polygon.boundary_ids:
            curve = _limiting_curve(polygon, boundaries[line_id], axes, ranges)
            if curve is None:
                continue
            for index, state in enumerate(curve["points"]):
                if state["success"] and state["phases"]:
                    yield ("boundary", line_id, index), tuple(
                        state[a] for a in axes
                    ), state

    return sample
