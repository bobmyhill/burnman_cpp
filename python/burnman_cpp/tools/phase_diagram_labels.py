# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Composition continuity labels for coexisting solution branches."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json

import numpy as np

from .. import _core
from .phase_diagram import _phase_diagram_data, _section_axes


def label_phase_diagram_solutions(result, *, tolerance=1.0e-8, close_domain=True):
    """Return a saved-diagram dictionary with secondary solution labels.

    Accept a native diagram, dictionary or JSON path. For each solution that
    occurs more than once in any closed field, track its compositions within
    fields and across their shared reaction lines. Labels such as ``fsp_1``
    and ``fsp_2`` distinguish coexisting branches, including in neighbouring
    fields where only one branch survives. No fixed endmember cutoff, phase
    model, equilibrium solve or grid is needed. Original IDs and data remain
    unchanged; save the returned dictionary with ``save_phase_diagram``.

    Compositions are matched one-to-one along a minimum spanning tree in
    rescaled diagram coordinates. Shared boundary compositions join field
    branches only if that cannot give two instances in one field the same
    label. Disconnected branches retain separate labels. Numeric suffixes are
    descriptive identifiers, rather than formal mineral species names.

    ``secondary_phase_labels`` contains per-field labels, composition ranges,
    group membership and continuity diagnostics. Missing composition evidence
    raises ValueError. Sparse saved states or branch coalescence can make
    continuity ambiguous; conflicting links are recorded and kept separate.
    Plot with ``plot_phase_diagram(result, secondary_labels=True)`` to use
    saved labels or compute them on demand. Plotting defaults are unchanged.
    """
    data = _phase_diagram_data(result)
    geometry = _core.phase_diagram_field_polygons(data, tolerance, close_domain)
    labels = _secondary_phase_labels(data, geometry, tolerance, close_domain)
    output = deepcopy(data)
    output["secondary_phase_labels"] = labels
    return output


def _source_digest(data):
    original = {k: v for k, v in data.items() if k != "secondary_phase_labels"}
    return sha256(
        json.dumps(original, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _secondary_phase_labels(data, geometry, tolerance, close_domain):
    """Reuse annotations only while their source states and geometry agree."""
    digest = _source_digest(data)
    previous = data.get("secondary_phase_labels", {})
    if (
        previous.get("schema_version") == 1
        and previous.get("source_digest") == digest
        and previous.get("tolerance") == tolerance
        and previous.get("close_domain") == close_domain
    ):
        return deepcopy(previous)

    copies = data.get("settings", {}).get("max_phase_instances", 3)
    polygons = [p for p in geometry.polygons if p.phases and not p.outside_model_domain]
    repeated = {
        candidate
        for polygon in polygons
        for candidate, count in Counter(i // copies for i in polygon.phases).items()
        if count > 1
    }
    _, axes = _section_axes(data)
    ranges = np.asarray([data[axis + "_range"] for axis in axes], dtype=float)
    boundaries = {b["id"]: b for b in data.get("boundaries", [])}
    samples = data.get("samples", [])
    fields = {f["id"]: f for f in data.get("fields", [])}
    names = data.get("phase_names", [])
    labels, groups, diagnostics = {}, [], []

    for candidate in sorted(repeated):
        nodes, shared = [], defaultdict(list)
        base = names[candidate * copies].split(" #")[0]
        for polygon in polygons:
            phase_ids = [i for i in polygon.phases if i // copies == candidate]
            if not phase_ids:
                continue
            records = _field_observations(
                polygon,
                candidate,
                len(phase_ids),
                copies,
                samples,
                fields,
                boundaries,
                axes,
                ranges,
            )
            if not records:
                raise ValueError(
                    f"Saved compositions required for {base} in field "
                    f"{polygon.field_number}; no unambiguous equilibrium states found."
                )
            tracks = _track_compositions(records)
            first_node = len(nodes)
            for slot, phase_id in enumerate(phase_ids):
                values = np.asarray(
                    [
                        record[2][track.index(slot)][1]
                        for record, track in zip(records, tracks)
                    ]
                )
                nodes.append(
                    dict(
                        field=polygon.field_number,
                        phase_id=phase_id,
                        mean=values.mean(axis=0),
                        low=values.min(axis=0),
                        high=values.max(axis=0),
                    )
                )
            for (key, _, phases), track in zip(records, tracks):
                if key[0] == "boundary":
                    for (phase_id, _), slot in zip(phases, track):
                        shared[(*key, phase_id)].append(first_node + slot)

        # Exact shared boundary states supply cross-field links. Their solver
        # copy IDs are used only within that one saved boundary equilibrium.
        votes = Counter(
            tuple(sorted(pair))
            for members in shared.values()
            for pair in combinations(members, 2)
        )
        parents = list(range(len(nodes)))
        occupied = [{node["field"]} for node in nodes]

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        conflicts = 0
        for (a, b), _ in sorted(votes.items(), key=lambda item: (-item[1], item[0])):
            a, b = root(a), root(b)
            if a == b:
                continue
            if occupied[a] & occupied[b]:
                conflicts += 1
                continue
            parents[b] = a
            occupied[a].update(occupied[b])
        if conflicts:
            diagnostics.append(
                f"{base}: kept {conflicts} conflicting continuity links separate "
                "to distinguish coexisting instances."
            )
        components = defaultdict(list)
        for i in range(len(nodes)):
            components[root(i)].append(i)
        ordered = sorted(
            components.values(),
            key=lambda members: tuple(
                np.mean([nodes[i]["mean"] for i in members], axis=0)
            ),
        )
        for index, members in enumerate(ordered, 1):
            label = f"{base}_{index}"
            for i in members:
                node = nodes[i]
                labels[node["field"], node["phase_id"]] = label
            groups.append(
                dict(
                    label=label,
                    candidate_index=candidate,
                    name=base,
                    fields=sorted(nodes[i]["field"] for i in members),
                    mean_composition=np.mean(
                        [nodes[i]["mean"] for i in members], axis=0
                    ).tolist(),
                    minimum_composition=np.min(
                        [nodes[i]["low"] for i in members], axis=0
                    ).tolist(),
                    maximum_composition=np.max(
                        [nodes[i]["high"] for i in members], axis=0
                    ).tolist(),
                )
            )
    return dict(
        schema_version=1,
        method="composition_continuity",
        source_digest=digest,
        tolerance=tolerance,
        close_domain=close_domain,
        fields=[
            dict(
                field_number=p.field_number,
                phases=[
                    dict(
                        phase_id=i,
                        label=labels.get((p.field_number, i), names[i].split(" #")[0]),
                    )
                    for i in p.phases
                ],
            )
            for p in polygons
        ],
        groups=groups,
        diagnostics=diagnostics,
    )


def _limiting_curve(polygon, boundary, axes, ranges):
    """Select a saved field-side curve, using orientation for equal names."""
    curves = boundary.get("limiting_curves", [])
    matches = [
        i
        for i, curve in enumerate(curves)
        if curve["assemblage"] == list(polygon.phases)
    ]
    if len(matches) == 1:
        return curves[matches[0]]
    if len(matches) != 2:
        return None
    span = np.diff(ranges).ravel()
    points = np.asarray([[p[a] for a in axes] for p in boundary["points"]])
    for first, last in zip(points[:-1], points[1:]):
        along = (last - first) / span
        if np.linalg.norm(along) == 0.0:
            continue
        right = np.array([along[1], -along[0]]) / np.linalg.norm(along)
        centre = (first + last) / 2.0
        for step in (1e-9, 1e-8, 1e-7, 1e-6):
            for side, sign in ((0, 1.0), (1, -1.0)):
                if polygon.contains_rectangle(
                    centre + sign * step * span * right, span * step * 0.01
                ):
                    return curves[side]
    return None


def _field_observations(
    polygon, candidate, count, copies, samples, fields, boundaries, axes, ranges
):
    """Select the field's limiting compositions, retaining zero-amount roots."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    records, ambiguous = [], []
    sample_ids = set(fields.get(polygon.field_id, {}).get("sample_indices", []))
    if polygon.sample_index >= 0:
        sample_ids.add(polygon.sample_index)
    states = [(("sample", i), samples[i], None) for i in sorted(sample_ids)]
    for line_id in polygon.boundary_ids:
        boundary = boundaries[line_id]
        curve = _limiting_curve(polygon, boundary, axes, ranges)
        points = curve["points"] if curve is not None else boundary["points"]
        if boundary.get("limiting_curves") and curve is None:
            continue
        states += [
            (("boundary", line_id, i), point, boundary)
            for i, point in enumerate(points)
            if curve is None or point.get("success", False)
        ]
    half_size = np.diff(ranges).ravel() * 1.0e-12
    for key, state, boundary in states:
        position = np.array([state[axis] for axis in axes], dtype=float)
        # Admit seed states on the calculation frame by probing just inside
        # it. Internal reaction boundaries still need their own limiting states.
        probe = np.clip(
            position, ranges[:, 0] + 2 * half_size, ranges[:, 1] - 2 * half_size
        )
        if boundary is None and (
            not state.get("success", False)
            or (position < ranges[:, 0] - half_size).any()
            or (position > ranges[:, 1] + half_size).any()
            or not polygon.contains_rectangle(probe, half_size)
        ):
            continue
        phases = []
        for phase in state.get("phases", []):
            if phase.get("candidate_index", phase["id"] // copies) != candidate:
                continue
            composition = np.asarray(phase.get("composition", []), dtype=float)
            if (
                composition.ndim != 1
                or not composition.size
                or not np.isfinite(composition).all()
            ):
                raise ValueError(
                    "Secondary labels require finite saved phase compositions."
                )
            phases.append((phase["id"], composition))
        if len(phases) < count:
            continue
        if (
            boundary is not None
            and len(phases) > count
            and boundary.get("reaction_kind") == "phase_out"
        ):
            phases = [p for p in phases if p[0] != boundary.get("zero_phase")]
        phases.sort(key=lambda phase: tuple(phase[1]))
        record = (key, (position - ranges[:, 0]) / np.diff(ranges).ravel(), phases)
        if len(phases) == count:
            records.append(record)
        elif len(phases) > count:
            ambiguous.append(record)
    # Replacement boundaries can retain both limiting compositions. Select
    # the field's subset by matching the nearest unambiguous saved state.
    if records:
        reference_positions = np.asarray([r[1] for r in records])
        for key, position, phases in ambiguous:
            near = np.argmin(np.linalg.norm(reference_positions - position, axis=1))
            reference = [p[1] for p in records[near][2]]
            _, selected = linear_sum_assignment(
                cdist(reference, [p[1] for p in phases])
            )
            records.append((key, position, [phases[i] for i in sorted(selected)]))
    return sorted(
        records,
        key=lambda r: (tuple(r[1]), tuple(np.concatenate([p[1] for p in r[2]]))),
    )


def _track_compositions(records):
    """Transport distinct branch identities locally, allowing drifting solvi."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist

    count = len(records[0][2])
    tracks = [None] * len(records)
    tracks[0] = list(range(count))
    if len(records) == 1:
        return tracks
    distances = cdist([r[1] for r in records], [r[1] for r in records])
    distances += np.finfo(float).eps
    np.fill_diagonal(distances, 0.0)
    tree = minimum_spanning_tree(csr_matrix(distances)).tocoo()
    neighbours = defaultdict(list)
    for a, b in zip(tree.row, tree.col):
        neighbours[a].append(b)
        neighbours[b].append(a)
    pending = [0]
    while pending:
        parent = pending.pop()
        for child in sorted(neighbours[parent]):
            if tracks[child] is not None:
                continue
            left, right = linear_sum_assignment(
                cdist(
                    [p[1] for p in records[parent][2]],
                    [p[1] for p in records[child][2]],
                )
            )
            tracks[child] = [0] * count
            for a, b in zip(left, right):
                tracks[child][b] = tracks[parent][a]
            pending.append(child)
    return tracks
