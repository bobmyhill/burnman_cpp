# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Field-side limiting amounts from saved boundary compositions and chemistry."""

from collections import Counter

import numpy as np


def field_boundary_states(data, phases):
    """Return a sampler yielding (boundary key, position, limiting state).

    A replacement boundary can save the finite amounts of the opposite field
    (e.g. sillimanite rather than kyanite). Keep the field's saved composition
    roots, then recover its unique nonnegative amounts by linear mass balance.
    No compositions, P/T or equilibria are solved. Incomplete, nonunique or
    incompatible limits are omitted so interpolation can use field samples.
    """
    from .phase_diagram import _section_axes
    from .phase_diagram_labels import _field_observations
    from .. import _core

    copies = data.get("settings", {}).get("max_phase_instances", 3)
    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    fields = {f["id"]: f for f in data.get("fields", [])}
    boundaries = {b["id"]: b for b in data.get("boundaries", [])}
    formulae = [
        p.endmember_formulae if isinstance(p, _core.Solution) else [p.formula]
        for p in phases
    ]
    elements = sorted({e for group in formulae for formula in group for e in formula})
    chemistry = [
        np.array([[formula.get(e, 0.0) for formula in group] for e in elements])
        for group in formulae
    ]

    def sample(polygon):
        counts = Counter(i // copies for i in polygon.phases)
        roots = [
            {
                key: {p[0] for p in states}
                for key, _, states in _field_observations(
                    polygon,
                    candidate,
                    count,
                    copies,
                    data.get("samples", []),
                    fields,
                    boundaries,
                    axes,
                    ranges,
                )
            }
            for candidate, count in counts.items()
        ]
        for line_id in polygon.boundary_ids:
            for index, state in enumerate(boundaries[line_id]["points"]):
                key = ("boundary", line_id, index)
                if not roots or any(key not in group for group in roots):
                    continue
                ids = set().union(*(group[key] for group in roots))
                states = state["phases"]
                selected = [i for i, p in enumerate(states) if p["id"] in ids]
                matrix = np.column_stack(
                    [
                        chemistry[p["candidate_index"]] @ np.asarray(p["composition"])
                        for p in states
                    ]
                )
                amounts = np.array([p["amount"] for p in states])
                bulk = matrix @ amounts
                target = matrix[:, selected]
                retained = amounts[selected].copy()
                remainder = bulk - target @ retained
                tolerance = 1.0e-9 * max(np.linalg.norm(bulk), 1.0)
                if np.linalg.norm(remainder) > tolerance:
                    if boundaries[line_id].get("reaction_kind") != "replacement":
                        continue
                    # Changing amounts on a latent S/V edge also changes its
                    # plotted coordinate; that limit needs its own saved state.
                    if "S" in data["diagram_type"] or "V" in data["diagram_type"]:
                        continue
                    correction, _, rank, _ = np.linalg.lstsq(
                        target, remainder, rcond=None
                    )
                    if rank < len(selected):
                        continue
                    retained += correction
                if (retained < -1.0e-10 * max(amounts.sum(), 1.0)).any():
                    continue
                retained = np.maximum(retained, 0.0)
                if np.linalg.norm(target @ retained - bulk) > tolerance:
                    continue
                limiting = [
                    dict(states[i], amount=float(n)) for i, n in zip(selected, retained)
                ]
                # A replacement changes total volume. Recompute it when
                # evaluating a volume fraction instead of using the other side.
                yield key, tuple(state[a] for a in axes), dict(
                    state, phases=limiting, volume=0.0
                )

    return sample
