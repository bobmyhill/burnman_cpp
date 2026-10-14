"""Pseudosection figures, refinement and local patches using native calculations."""

from collections.abc import Mapping
from copy import deepcopy
import warnings

from .. import _core, constants


def save_pseudosection(result, path, *, metadata=None):
    """Save a native result or saved dictionary as JSON; return the file path.

    All coordinates retain their native SI units. Optional ``metadata`` adds
    top-level entries, e.g. a model set or oxide amounts. Result entries take
    precedence, so ``metadata=previous`` retains extra information when saving
    a refined result without replacing its updated lines or settings.

    Parent directories are created as needed. Non-finite values are rejected
    before writing, leaving an existing file unchanged if serialization fails.
    """
    import json
    from pathlib import Path

    data = dict(result) if isinstance(result, Mapping) else result.to_dict()
    data = dict(metadata or {}) | data
    if "diagnostics" in data:
        data["diagnostics"] = _core._normalize_pseudosection_diagnostics(
            data["diagnostics"]
        )
    text = json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def load_pseudosection(path):
    """Load a pseudosection JSON file, preserving all entries and metadata.

    Return a dictionary usable for plotting, contours and refinement, with
    coordinates in native SI units. Candidate models are supplied separately
    when calculating new points. Legacy Region indices in diagnostics are
    translated to the same one-based Field numbers used by plots and keys;
    other entries are preserved.
    Use ``PseudosectionResult.from_dict(data)`` if a native object is needed.
    """
    import json
    from pathlib import Path

    with Path(path).open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("A saved pseudosection must be a JSON object.")
    if "diagnostics" in data:
        data["diagnostics"] = _core._normalize_pseudosection_diagnostics(
            data["diagnostics"]
        )
    return data


def _value(record, name, default=None):
    return (
        record.get(name, default)
        if isinstance(record, Mapping)
        else getattr(record, name, default)
    )


def _pseudosection_data(result):
    from os import PathLike

    if isinstance(result, (str, PathLike)):
        return load_pseudosection(result)
    return result if isinstance(result, Mapping) else result.to_dict()


def _section_axes(result):
    diagram, names, *_ = _coordinate_transforms(result)
    names["X"] = "composition"
    return diagram, [names[axis] for axis in diagram]


def _section_window(result, region, display):
    import numpy as np

    data = _pseudosection_data(result)
    _, axes = _section_axes(data)
    bounds = np.asarray(region, dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or (bounds[:, 0] >= bounds[:, 1]).any()
    ):
        raise ValueError("region must contain finite increasing x and y bounds.")
    *_, to_native = _coordinate_transforms(data, **display)
    window = np.sort(to_native(bounds.T), axis=0).T
    domain = np.asarray([data[axis + "_range"] for axis in axes])
    if (window[:, 0] < domain[:, 0]).any() or (window[:, 1] > domain[:, 1]).any():
        raise ValueError("The patch region must be contained in the original domain.")
    return data, axes, window


def _boundary_coordinates(boundary, nodes, names):
    records = list(boundary["points"]) + [
        nodes[n] for n in (boundary["start_node"], boundary["end_node"]) if n in nodes
    ]
    return [[r[name] for name in names] for r in records]


def _padded_section_region(data, points, padding, display):
    import numpy as np

    _, axes = _section_axes(data)
    domain = np.asarray([data[a + "_range"] for a in axes], dtype=float)
    points = np.clip(np.asarray(points), domain[:, 0], domain[:, 1])
    low, high = points.min(axis=0), points.max(axis=0)
    width = high - low
    minimum = (domain[:, 1] - domain[:, 0]) * data.get("settings", {}).get(
        "step", 0.025
    )
    pad = np.where(width > 0, width * padding, minimum / 2)
    low, high = np.maximum(domain[:, 0], low - pad), np.minimum(
        domain[:, 1], high + pad
    )
    *_, coordinates, _ = _coordinate_transforms(data, **display)
    shown = np.sort(coordinates(np.array([low, high])), axis=0).T
    return tuple(tuple(v) for v in shown)


def pseudosection_field_regions(result, field_numbers=None, *, padding=0.05, **display):
    """Return ``{Field number: ((xmin, xmax), (ymin, ymax))}`` for local patches.

    Select provisional, unlabelled and open fields by default, or supply the
    one-based Field numbers used in plots. Include their complete boundaries
    and endpoint nodes. Pad each side by ``padding`` times that axis span and
    clip to the original domain. A zero-width field uses the tracing step as
    its minimum window width. Regions use the plot units and axis order;
    ``**display`` accepts the same unit options as ``plot_pseudosection``.
    No equilibrium solves are performed.
    """
    import numpy as np

    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative.")
    data = _pseudosection_data(result)
    _, axes = _section_axes(data)
    names = ["composition_coordinate" if a == "composition" else a for a in axes]
    polygons = _core.pseudosection_field_polygons(data).polygons
    selected = None if field_numbers is None else set(field_numbers)
    boundaries = {b["id"]: b for b in data["boundaries"]}
    nodes = {n["id"]: n for n in data["nodes"]}
    regions = {}
    for p in polygons:
        if (
            p.field_number not in selected
            if selected is not None
            else not (
                p.is_provisional
                or p.has_open_boundary
                or (not p.phases and not p.outside_model_domain)
            )
        ):
            continue
        points = [list(v) for v in p.vertices]
        for bid in p.boundary_ids:
            points.extend(_boundary_coordinates(boundaries[bid], nodes, names))
        regions[p.field_number] = _padded_section_region(data, points, padding, display)
    if selected is not None and selected - regions.keys():
        raise ValueError(f"Unknown Field numbers: {sorted(selected - regions.keys())}.")
    return regions


def pseudosection_problem_regions(result, *, padding=0.05, **display):
    """Return patch windows labelled ``'Field N'`` or ``'Boundary N'``.

    Include provisional, unlabelled and open fields, plus boundaries named in
    saved ``Boundary N`` diagnostics. Each boundary is included once, using
    its native ID without adding one. Field numbers remain one-based.
    Windows include all saved boundary points and endpoint nodes, with the
    same padding, domain clipping, units and zero-width handling as
    ``pseudosection_field_regions``. No equilibrium solves are performed.
    """
    import re

    data = _pseudosection_data(result)
    regions = {
        f"Field {n}": region
        for n, region in pseudosection_field_regions(
            data, padding=padding, **display
        ).items()
    }
    flagged = {
        int(match[1])
        for message in data.get("diagnostics", [])
        if (match := re.match(r"Boundary\s+(\d+)\b", message))
    }
    _, axes = _section_axes(data)
    names = ["composition_coordinate" if a == "composition" else a for a in axes]
    nodes = {n["id"]: n for n in data["nodes"]}
    for boundary in data["boundaries"]:
        if boundary["id"] in flagged:
            points = _boundary_coordinates(boundary, nodes, names)
            if points:
                regions[f"Boundary {boundary['id']}"] = _padded_section_region(
                    data, points, padding, display
                )
    return regions


def calculate_pseudosection_patch(
    previous, phases, region, settings=None, *, resolution=None, **display
):
    """Calculate an independent, locally detailed diagram inside ``previous``.

    ``previous`` is a native result, dictionary or JSON path; its bulk, diagram
    axes, composition endpoints and fixed coordinate are reused. ``region``
    is ``((xmin, xmax), (ymin, ymax))`` in plot units, as in regional refinement.
    Supply the original candidate models and settings with increased axis seed
    counts and/or smaller tracing steps. Defaults reuse the saved settings.
    Optional ``resolution=(nx, ny)`` also refines the new patch's line spacing.
    This calculation starts afresh in the small window. It does not change the
    original graph; use ``merge_pseudosections`` to insert the returned result.
    """
    data, axes, window = _section_window(previous, region, display)
    options = {
        a + "_range": data[a + "_range"]
        for a in ("pressure", "temperature", "entropy", "volume", "composition")
        if a + "_range" in data
    }
    options.update({a + "_range": tuple(bounds) for a, bounds in zip(axes, window)})
    fixed = data.get("fixed_coordinate")
    if fixed:
        name = dict(P="pressure", T="temperature", S="entropy", V="volume")[fixed]
        options.pop(name + "_range", None)
        options[name] = data["fixed_value"]
    if settings is None:
        settings = _core.PseudosectionSettings.from_dict(data.get("settings", {}))
    result = _core.pseudosection(
        data["composition_start"],
        phases,
        settings=settings,
        diagram=data.get("diagram_type", "PT"),
        composition_end=data.get("composition_end", {}),
        **options,
    )
    if resolution is not None:
        result = refine_pseudosection(
            data["composition_start"],
            phases,
            result,
            settings,
            resolution=resolution,
            **display,
        )
    return result


def _box_intersection(a, b, box):
    """Parameter interval of a segment inside an axis-aligned rectangle."""
    lo, hi = 0.0, 1.0
    for x, y, (low, high) in zip(a, b, box):
        if x == y:
            if not low <= x <= high:
                return None
        else:
            enter, leave = sorted(((low - x) / (y - x), (high - x) / (y - x)))
            lo, hi = max(lo, enter), min(hi, leave)
    return (lo, hi) if lo < hi else None


class _PatchSegment:
    def __init__(self, line, points, start, end, statuses, *, base):
        self.line, self.points = deepcopy(line), list(points)
        self.nodes, self.tokens, self.statuses = (
            [start, end],
            [None, None],
            list(statuses),
        )
        self.base = base
        self.origins = {(0 if base else 1, line["id"])}

    def reverse(self):
        self.points.reverse()
        self.nodes.reverse()
        self.tokens.reverse()
        self.statuses.reverse()
        self.line["side_a"], self.line["side_b"] = self.line.get(
            "side_b", []
        ), self.line.get("side_a", [])


def merge_pseudosections(base, patch, *, join_tolerance=None, phases=None):
    """Replace the part of ``base`` inside ``patch``; return ``(dict, report)``.

    Inputs accept native results, dictionaries or JSON paths and are not
    modified. They must use the same bulk, candidate ordering, phase-instance
    policy, diagram axes and fixed coordinate. Repeated calls support multiple
    and overlapping rectangles: the last patch wins. Original samples inside
    the rectangle are replaced, and graph/sample indices are rebuilt.

    Match cut ends to patch-frame ports with the same reaction (zero phase,
    assemblage and replacement flag), nearest first, one-to-one. Distances
    are measured to the cut chord in units of the patch's axis spans. Optional
    ``join_tolerance`` limits this distance. Unmatched internal ends stay open
    and diagnostic. Points retain their solved thermodynamic states; seam
    joins add chords, not interpolated equilibrium states. Thus a geometric
    merge does not itself certify the thermodynamics of the seam.

    Rebuild polygon diagnostics and retain unrelated source warnings. Supplying
    the original ``phases`` additionally runs the native endpoint and sampled
    assemblage checks, without resuming tracing. The returned report describes
    joins; outstanding problems are in ``dict['diagnostics']``.
    """
    import math
    import re
    import numpy as np

    base, patch = _pseudosection_data(base), _pseudosection_data(patch)
    if join_tolerance is not None and (
        not math.isfinite(join_tolerance) or join_tolerance < 0
    ):
        raise ValueError("join_tolerance must be finite and nonnegative.")
    for key in (
        "diagram_type",
        "composition_start",
        "composition_end",
        "phase_names",
        "fixed_coordinate",
        "fixed_value",
    ):
        if base.get(key) != patch.get(key):
            raise ValueError(f"Incompatible pseudosections: {key} differs.")
    if (
        "model_set" in base
        and "model_set" in patch
        and base["model_set"] != patch["model_set"]
    ):
        raise ValueError("Incompatible pseudosections: model_set differs.")
    defaults = _core.PseudosectionSettings().to_dict()
    for key in (
        "max_phase_instances",
        "exclude_invalid_eos",
        "active_solution_faces",
        "required_eos_phases",
    ):
        if base.get("settings", {}).get(key, defaults[key]) != patch.get(
            "settings", {}
        ).get(key, defaults[key]):
            raise ValueError(f"Incompatible pseudosections: setting {key} differs.")
    _, axes = _section_axes(base)
    names = ["composition_coordinate" if a == "composition" else a for a in axes]
    box = np.asarray([patch[a + "_range"] for a in axes], dtype=float)
    domain = np.asarray([base[a + "_range"] for a in axes], dtype=float)
    if (
        not np.isfinite(box).all()
        or (box[:, 0] >= box[:, 1]).any()
        or (box[:, 0] < domain[:, 0]).any()
        or (box[:, 1] > domain[:, 1]).any()
    ):
        raise ValueError(
            "The patch must have increasing ranges contained in the base domain."
        )
    scale = box[:, 1] - box[:, 0]

    def point(record):
        return np.array([record[name] for name in names])

    def inside(record):
        q = point(record)
        return bool(((q >= box[:, 0]) & (q <= box[:, 1])).all())

    def frame(q, bounds):
        return bool((np.abs((q[:, None] - bounds) / scale[:, None]) <= 1.0e-8).any())

    def signature(line):
        return (
            line["zero_phase"],
            tuple(sorted(line["assemblage"])),
            line.get("is_solution_replacement", False),
        )

    def statuses(line):
        values = line.get(
            "termination", "unresolved endpoint; unresolved endpoint"
        ).split(";")
        return [s.strip() for s in values] if len(values) == 2 else values * 2

    nodes = {
        (source, n["id"]): deepcopy(n)
        for source, data in enumerate((base, patch))
        for n in data["nodes"]
    }
    segments, cuts, ports = [], {}, {}

    def cut(segment, end, a, b):
        token = len(cuts)
        segment.tokens[end] = ("cut", token)
        segment.nodes[end] = None
        segment.statuses[end] = "unresolved patch seam"
        cuts[segment.tokens[end]] = (point(a), point(b), signature(segment.line))

    for line in base["boundaries"]:
        pts = line["points"]
        if not pts:
            continue
        ends = [(0, line.get(key, -1)) for key in ("start_node", "end_node")]
        current = (
            None
            if inside(pts[0])
            else _PatchSegment(line, [pts[0]], *ends, statuses(line), base=True)
        )
        for a, b in zip(pts, pts[1:]):
            interval = _box_intersection(point(a), point(b), box)
            if interval is None:
                if inside(b):
                    if current is not None:
                        cut(current, 1, a, b)
                        segments.append(current)
                        current = None
                elif current is not None:
                    current.points.append(b)
                else:
                    current = _PatchSegment(line, [b], *ends, statuses(line), base=True)
                    cut(current, 0, a, b)
                continue
            _, leave = interval
            if current is not None:
                cut(current, 1, a, b)
                segments.append(current)
                current = None
            if leave < 1:
                current = _PatchSegment(line, [b], *ends, statuses(line), base=True)
                cut(current, 0, a, b)
        if current is not None:
            segments.append(current)

    for line in patch["boundaries"]:
        if not line["points"]:
            continue
        segment = _PatchSegment(
            line,
            line["points"],
            (1, line.get("start_node", -1)),
            (1, line.get("end_node", -1)),
            statuses(line),
            base=False,
        )
        for end in (0, 1):
            node = nodes.get(segment.nodes[end])
            if node and node["kind"] == "domain_edge":
                token = ("port", len(ports))
                segment.tokens[end] = token
                ports[token] = (point(node), signature(line), segment.nodes[end])
        segments.append(segment)

    def distance(q, a, b):
        delta, offset = (b - a) / scale, (q - a) / scale
        u = np.clip(offset @ delta / (delta @ delta), 0, 1) if delta @ delta else 0
        return float(np.linalg.norm(offset - u * delta))

    pairs = sorted(
        (distance(q, a, b), ct, pt)
        for ct, (a, b, reaction) in cuts.items()
        for pt, (q, other, _) in ports.items()
        if reaction == other
    )
    owners = {token: s for s in segments for token in s.tokens if token is not None}
    used, gaps = set(), []
    for gap, ct, pt in pairs:
        if (
            ct in used
            or pt in used
            or (join_tolerance is not None and gap > join_tolerance)
        ):
            continue
        x, y = owners[ct], owners[pt]
        used.update((ct, pt))
        gaps.append(gap)
        if x is y:
            x.points.append(x.points[0])
            x.nodes, x.tokens, x.statuses = (
                [None, None],
                [None, None],
                ["closed loop"] * 2,
            )
            continue
        if x.tokens[1] != ct:
            x.reverse()
        if y.tokens[0] != pt:
            y.reverse()
        # Patch side labels take precedence, in the joined direction.
        joined = _PatchSegment(
            x.line if not x.base else y.line,
            x.points + y.points,
            x.nodes[0],
            y.nodes[1],
            [x.statuses[0], y.statuses[1]],
            base=x.base and y.base,
        )
        joined.tokens = [x.tokens[0], y.tokens[1]]
        joined.origins = x.origins | y.origins
        slot = min(segments.index(x), segments.index(y))
        segments.remove(x)
        segments.remove(y)
        segments.insert(slot, joined)
        for token in joined.tokens:
            if token is not None:
                owners[token] = joined

    messages = []
    for s in segments:
        for end, token in enumerate(s.tokens):
            if token in ports and token not in used:
                key = ports[token][2]
                if not frame(point(nodes[key]), domain):
                    nodes[key]["kind"] = "patch_seam"
                    s.statuses[end] = "unresolved patch seam"
            if s.nodes[end] not in nodes:
                s.nodes[end] = None
    active = {n for s in segments for n in s.nodes if n is not None}
    node_map = {key: i for i, key in enumerate(sorted(active))}
    new_nodes = [
        dict(nodes[key], id=i, incident_lines=[]) for key, i in node_map.items()
    ]
    lines = []
    for i, s in enumerate(segments):
        line = dict(
            s.line,
            id=i,
            points=deepcopy(s.points),
            start_node=node_map.get(s.nodes[0], -1),
            end_node=node_map.get(s.nodes[1], -1),
            termination="; ".join(s.statuses),
        )
        lines.append(line)
        for key in set(s.nodes) - {None}:
            new_nodes[node_map[key]]["incident_lines"].append(i)
        if any(
            "unresolved" in status or "failed" in status or status == "trace step limit"
            for status in s.statuses
        ):
            messages.append(f"Boundary {i}: {line['termination']}")
        if not line.get("side_a") or not line.get("side_b"):
            messages.append(
                f"Boundary {i}: neighbouring fields could not be distinguished within tolerances."
            )

    samples, fields, by_phases = [], [], {}
    for source, data in enumerate((base, patch)):
        index_map = {}
        for i, sample in enumerate(data["samples"]):
            if source == 0 and inside(sample):
                continue
            index_map[i] = len(samples)
            samples.append(deepcopy(sample))
        for field in data["fields"]:
            indices = [index_map[i] for i in field["sample_indices"] if i in index_map]
            if not indices:
                continue
            key = tuple(sorted(field["phases"]))
            if key not in by_phases:
                by_phases[key] = dict(
                    id=len(fields), phases=list(key), sample_indices=[]
                )
                fields.append(by_phases[key])
            by_phases[key]["sample_indices"].extend(indices)

    merged = deepcopy(base)
    merged.update(nodes=new_nodes, boundaries=lines, samples=samples, fields=fields)

    # Partition each old excluded ring into the four strips outside the patch.
    # Only coordinates are clipped here; accepted equilibrium states above
    # are deliberately never interpolated.
    def clip_ring(ring, axis, bound, less):
        out = []
        for a, b in zip(ring, ring[1:] + ring[:1]):
            a, b = np.asarray(a), np.asarray(b)
            a_in, b_in = (
                (a[axis] <= bound, b[axis] <= bound)
                if less
                else (a[axis] >= bound, b[axis] >= bound)
            )
            if a_in:
                out.append(a.tolist())
            if a_in != b_in:
                out.append(
                    (a + (b - a) * ((bound - a[axis]) / (b[axis] - a[axis]))).tolist()
                )
        return out

    excluded = []
    for ring in base.get("excluded_regions", []):
        rest = deepcopy(ring)
        for axis, edge, less in (
            (0, 0, True),
            (0, 1, False),
            (1, 0, True),
            (1, 1, False),
        ):
            outside = clip_ring(rest, axis, box[axis, edge], less)
            if len(outside) >= 3:
                excluded.append(outside)
            rest = clip_ring(rest, axis, box[axis, edge], not less)
    merged["excluded_regions"] = excluded + deepcopy(patch.get("excluded_regions", []))
    for key in ("equilibrium_solves", "minimization_calls"):
        merged[key] = base.get(key, 0) + patch.get(key, 0)
    # These messages refer to the old graph. Regenerate them below rather than
    # retaining obsolete Field/Boundary/Node numbers after clipping.
    graph_message = re.compile(r"^(Field |Region |Unpaired |No closed region )")
    for source, data in enumerate((base, patch)):
        if source == 0 and np.array_equal(box, domain):
            continue
        for d in data.get("diagnostics", []):
            boundary = re.match(r"Boundary (\d+): (.*)", d)
            if boundary:
                old_line = next(
                    (b for b in data["boundaries"] if b["id"] == int(boundary[1])), None
                )
                if old_line and boundary[2] not in (
                    old_line.get("termination"),
                    "neighbouring fields could not be distinguished within tolerances.",
                ):
                    messages.extend(
                        f"Boundary {i}: {boundary[2]}"
                        for i, s in enumerate(segments)
                        if (source, int(boundary[1])) in s.origins
                    )
            elif not graph_message.match(d):
                messages.append(d)
    geometry = _core.pseudosection_field_polygons(merged)
    messages.extend(geometry.diagnostics)
    merged["diagnostics"] = list(dict.fromkeys(messages))
    merged["resolved"] = not merged["diagnostics"]
    if phases is not None:
        checked = _core.verify_pseudosection_critical_endpoints(merged, phases)
        merged.update(checked.to_dict())
    else:
        # Preserve unmatched-sample warnings unless their assemblage was
        # removed by clipping or now has a resolved perimeter.
        bordered = {
            tuple(p.phases) for p in geometry.polygons if not p.has_open_boundary
        }
        for data in (base, patch):
            old_fields = {f["id"]: f for f in data["fields"]}
            for d in data.get("diagnostics", []):
                match = re.match(
                    r"No closed region resolved for sampled (?:field|assemblage) (\d+)",
                    d,
                )
                if match and int(match[1]) in old_fields:
                    key = tuple(sorted(old_fields[int(match[1])]["phases"]))
                    if key in by_phases and key not in bordered:
                        merged["diagnostics"].append(
                            "No closed region resolved for sampled assemblage "
                            f"{by_phases[key]['id']}; increase seed density or inspect failed solves."
                        )
        for node in new_nodes:
            if node["kind"] == "junction" and len(node["incident_lines"]) == 1:
                merged["diagnostics"].append(
                    f"Unpaired internal junction at node {node['id']}."
                )
            if (
                node["kind"] == "critical_point"
                and len(node["incident_lines"]) == 1
                and not node.get("is_verified_critical_endpoint", False)
            ):
                merged["diagnostics"].append(
                    f"Unpaired solution critical endpoint at node {node['id']}; "
                    "its adjoining field remains unresolved."
                )
        merged["diagnostics"] = list(dict.fromkeys(merged["diagnostics"]))
        merged["resolved"] = not merged["diagnostics"]
    report = [
        f"Joined {len(gaps)} line ends; largest join gap {max(gaps, default=0):.3g} patch widths; "
        f"{len(cuts.keys() - used)} base line ends left open."
    ]
    return merged, report


def _coordinate_units(result, quantities, options):
    """SI-to-display scales for coordinates and contour values."""
    pressure_unit = options.get("pressure_unit", "kbar")
    temperature_unit = options.get("temperature_unit", "C")
    entropy_unit = options.get("entropy_unit", "J/K")
    volume_unit = options.get("volume_unit", "m3")
    pressure_scales = {"Pa": 1.0, "GPa": 1.0e9, "kbar": 1.0e8}
    if pressure_unit not in pressure_scales:
        raise ValueError("pressure_unit must be 'Pa', 'GPa' or 'kbar'.")
    if temperature_unit not in ("K", "C"):
        raise ValueError("temperature_unit must be 'K' or 'C'.")
    t_offset = 273.15 if temperature_unit == "C" else 0.0
    entropy_scales = {"J/K": 1.0, "kJ/K": 1000.0, "kB/atom": 1.0}
    volume_scales = {"m3": 1.0, "cm3": 1.0e-6, "kg/m3": 1.0}
    if entropy_unit not in entropy_scales or volume_unit not in volume_scales:
        raise ValueError(
            "entropy_unit must be 'J/K', 'kJ/K' or 'kB/atom'; "
            "volume_unit must be 'm3', 'cm3' or 'kg/m3'."
        )
    density = "V" in quantities and volume_unit == "kg/m3"
    per_atom = "S" in quantities and entropy_unit == "kB/atom"
    bulk_mass = None
    if density or per_atom:
        if "X" in _value(result, "diagram_type", "PT"):
            raise ValueError("Density and per-atom entropy require a constant bulk.")
        bulk = _value(result, "composition_start", {})
        if not bulk:
            raise ValueError("Density and per-atom entropy require composition_start.")
        composition = _core.Composition(bulk, "molar")
        bulk_mass = sum(composition.mass_composition.values())
        entropy_scales["kB/atom"] = (
            sum(composition.atomic_composition.values()) * constants.gas_constant
        )
    return (
        dict(
            P=pressure_scales[pressure_unit],
            T=1.0,
            S=entropy_scales[entropy_unit],
            V=volume_scales[volume_unit],
            X=1.0,
        ),
        t_offset,
        bulk_mass,
    )


def _coordinate_transforms(
    result,
    pressure_unit="kbar",
    temperature_unit="C",
    entropy_unit="J/K",
    volume_unit="m3",
    swap_axes=False,
):
    """Shared native-to-display transforms for phase lines and overlays."""
    import numpy as np

    diagram = _value(result, "diagram_type", "PT")
    if diagram not in ("PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"):
        raise ValueError("diagram_type must select two of P, T, S, V and X.")
    scales, t_offset, bulk_mass = _coordinate_units(
        result,
        diagram,
        dict(
            pressure_unit=pressure_unit,
            temperature_unit=temperature_unit,
            entropy_unit=entropy_unit,
            volume_unit=volume_unit,
        ),
    )
    density = "V" in diagram and volume_unit == "kg/m3"
    per_atom = "S" in diagram and entropy_unit == "kB/atom"
    names = dict(
        P="pressure",
        T="temperature",
        S="entropy",
        V="volume",
        X="composition_coordinate",
    )
    axis_scale = np.array([scales[axis] for axis in diagram])
    axis_offset = np.array([t_offset if axis == "T" else 0.0 for axis in diagram])
    order = [0, 1] if swap_axes else [1, 0]
    volume_axis = diagram.index("V") if density else None

    def coordinates(vertices):
        vertices = (np.asarray(vertices) - axis_offset) / axis_scale
        if density:
            vertices[:, volume_axis] = bulk_mass / vertices[:, volume_axis]
        return vertices[:, order]

    def native_coordinates(vertices):
        vertices = np.asarray(vertices)[:, order].copy()
        if density:
            vertices[:, volume_axis] = bulk_mass / vertices[:, volume_axis]
        return vertices * axis_scale + axis_offset

    return diagram, names, order, density, per_atom, coordinates, native_coordinates


def refine_pseudosection(
    composition,
    phases,
    previous,
    settings=None,
    *,
    resolution=None,
    region=None,
    pressure_unit="kbar",
    temperature_unit="C",
    entropy_unit="J/K",
    volume_unit="m3",
    swap_axes=False,
):
    """Resume a saved diagram and optionally refine all its phase lines.

    ``previous`` accepts a native result, its full dictionary or a JSON path.
    Use the same bulk and candidate models as the saved calculation.
    ``resolution=(nx, ny)`` gives axis point counts in plot axis order.
    ``(101, 201)`` means 100 divisions across x and 200 across y, limiting
    consecutive line-point spacing to the full axis span divided by (n - 1).
    Counts must be integers of at least 2. Plot unit options and ``swap_axes``
    select the displayed axes, including uniform spacing in density when used.
    All added points are solved and checked in C++. Failed refinements leave
    accepted points intact and report unmet spacing in ``result.diagnostics``.
    Completed diagrams reuse their saved lines without reopening each junction.
    Field labels use perimeter evidence and saved minimizations. Conflicting
    evidence is reported as provisional labelling, without interior solves or
    automatic curve edits.
    Set ``settings.verbose = True`` to report refinement and checking progress.
    Save the returned result with ``save_pseudosection``. Omitted resolution
    retains ordinary boundary recovery and the saved settings by default.
    ``region=((xmin, xmax), (ymin, ymax))`` limits spacing refinement to
    segments intersecting that window, in the selected plot units and axis
    order. Counts then subdivide the window rather than the whole diagram.
    Unstable arcs and arcs around provisional loops are retraced, using smaller
    steps within the window. The full graph and its original frame are retained.
    Regional refinement requires ``resolution`` and honours ``max_recovery_passes``.
    """
    from os import PathLike
    import numpy as np

    if isinstance(previous, (str, PathLike)):
        previous = load_pseudosection(previous)
    if resolution is None:
        if region is not None:
            raise ValueError("A refinement region requires resolution.")
        return _core.refine_pseudosection(composition, phases, previous, settings)
    counts = np.asarray(resolution, dtype=float)
    if (
        counts.shape != (2,)
        or not np.isfinite(counts).all()
        or (counts < 2).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError(
            "resolution must contain two integer axis point counts (nx, ny), each at least 2."
        )
    _, _, order, density, _, _, native_coordinates = _coordinate_transforms(
        previous, pressure_unit, temperature_unit, entropy_unit, volume_unit, swap_axes
    )
    native_region = None
    if region is not None:
        bounds = np.asarray(region, dtype=float)
        if (
            bounds.shape != (2, 2)
            or not np.isfinite(bounds).all()
            or (bounds[:, 0] >= bounds[:, 1]).any()
        ):
            raise ValueError("region must contain finite increasing x and y bounds.")
        native_region = np.sort(native_coordinates(bounds.T), axis=0).T.tolist()
    return _core.refine_pseudosection(
        composition,
        phases,
        previous,
        settings,
        resolution=tuple(int(count) for count in counts[order]),
        reciprocal_volume=density,
        region=native_region,
    )


def plot_pseudosection(
    result,
    ax=None,
    *,
    cmap="viridis",
    assemblage_colors=None,
    colorbar=True,
    fill_alpha=0.75,
    line_color="black",
    line_width=0.9,
    line_capstyle="round",
    line_joinstyle="round",
    show_nodes=False,
    show_unresolved=True,
    close_domain=True,
    merge_fields=True,
    tolerance=1.0e-8,
    pressure_unit="kbar",
    temperature_unit="C",
    entropy_unit="J/K",
    volume_unit="m3",
    swap_axes=False,
    composition_label="X",
    label_assemblages=True,
    phase_aliases=None,
    label_fontsize=7.0,
    label_key_path="pseudosection_labels.md",
):
    """Plot phase lines and colour closed fields by number of coexisting phases.

    Accept a native ``PseudosectionResult`` or a dictionary loaded from the
    basalt example's JSON. Return ``(figure, axes)``. Matplotlib is imported
    only when this function is called; install ``burnman_cpp[examples]``.

    C++ finds closed loops through saved stable vertices and traced lines,
    preserving holes and disconnected regions. Incidental chord crossings
    do not create vertices or subdivide a field. Each loop is filled independently,
    allowing coarse curves to overlap. The calculation frame closes edge fields;
    unfinished interior lines are never extended to manufacture closure.
    ``merge_fields`` is retained for compatibility and has no effect.
    Boundaries where solution compositions switch remain visible even when
    both fields have the same assemblage names.
    Connected segments separating the same two assemblages are drawn as one
    continuous line between endpoint vertices. Junctions and separate paths
    remain distinct; the equilibrium points are retained.
    ``line_capstyle`` accepts ``'butt'``, ``'round'`` or ``'projecting'``;
    ``line_joinstyle`` accepts ``'miter'``, ``'round'`` or ``'bevel'``.
    Both default to ``'round'``.
    Fields use agreed exterior-loop labels, then common admissible assemblages
    around the loop and saved interior minimizations. Boundary lengths never
    resolve conflicting assemblages. Provisional labels remain coloured and are
    identified by ``is_provisional`` and polygon diagnostics. Fields without
    any assemblage evidence stay uncoloured and issue a warning.
    Each coexisting solution instance counts as one phase.
    Native ``excluded_regions`` are drawn with grey hatching: the solid model
    cannot be evaluated there within its required mechanically stable EOS domain.
    Such regions have no assemblage count and are distinct from solve failures.

    The result selects any pair of P, T, S, V and X axes automatically. Composition diagrams
    place the dimensionless bulk mixing coordinate X on the horizontal axis;
    ``composition_label`` supplies its axis label. Pressure/temperature units
    apply to the physical axes, including temperature on the vertical TX axis.

    Pressure units are ``'Pa'``, ``'GPa'`` or ``'kbar'``; temperature units are
    ``'K'`` or ``'C'``. Total entropy units are ``'J/K'`` or ``'kJ/K'``;
    total volume units are ``'m3'`` or ``'cm3'``. For constant-composition
    sections, ``entropy_unit='kB/atom'`` divides entropy by the supplied bulk
    atom amount times the gas constant; ``volume_unit='kg/m3'`` displays
    density, using the bulk mass divided by total volume. ``swap_axes=True``
    places the first section coordinate on the horizontal axis. Both totals
    retain the supplied bulk amount scale. ``tolerance`` is a fraction of the calculation domain,
    used to merge numerically identical vertices. The colourbar uses integer
    ticks and discrete colours. All thermodynamics and polygon work are native.

    ``assemblage_colors`` maps tuples of native phase IDs or base model names
    to Matplotlib colours; order is ignored. Repeat a model name for each
    coexisting solution instance, e.g. ('g', 'fsp', 'fsp'). Unlisted fields
    retain phase-count colours. Custom colours suppress the phase-count
    colourbar. Named keys use model names before phase_aliases are applied.

    Label each identified closed field with its assemblage. ``phase_aliases``
    maps model names to abbreviations (e.g. ``{'g': 'gt'}``); repeated solution
    instances appear as ``2hb``. C++ selects interior label positions, avoiding
    holes. Every assemblage and numeric label uses ``label_fontsize`` (points).
    Labels that cannot fit, including their background, become unique field
    numbers. These one-based ``polygon.field_number`` values also identify
    fields in diagnostics and contours. Numbers that also cannot fit use
    leader lines. ``label_key_path``
    writes a separate Markdown assemblage key; pass ``None`` to suppress the
    file and read the same mapping from ``axes.pseudosection_label_key``.
    Set ``label_assemblages=False`` to suppress labels and the key.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.collections import PatchCollection, LineCollection
    from matplotlib.colors import BoundaryNorm
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Patch
    from matplotlib.transforms import Bbox
    from pathlib import Path as FilePath

    if not np.isfinite(label_fontsize) or label_fontsize <= 0.0:
        raise ValueError("label_fontsize must be positive and finite.")
    display_options = dict(
        pressure_unit=pressure_unit,
        temperature_unit=temperature_unit,
        entropy_unit=entropy_unit,
        volume_unit=volume_unit,
        swap_axes=swap_axes,
    )
    diagram, names, order, density, per_atom, coordinates, native_coordinates = (
        _coordinate_transforms(result, **display_options)
    )
    geometry = _core.pseudosection_field_polygons(
        result, tolerance, close_domain, merge_fields
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    fig = ax.figure
    ax.pseudosection_geometry = geometry
    ax._pseudosection_display_options = display_options

    def point_coordinates(point):
        return coordinates([[_value(point, names[axis]) for axis in diagram]])[0]

    phase_names = _value(result, "phase_names", [])
    patches, counts, phase_groups, domain_patches = [], [], [], []
    for polygon in geometry.polygons:
        if polygon.n_phases <= 0 and not polygon.outside_model_domain:
            continue
        paths = []
        for ring in [polygon.vertices, *polygon.holes]:
            points = coordinates(ring)
            codes = np.full(len(points), Path.LINETO, dtype=np.uint8)
            codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
            paths.append(Path(points, codes))
        patch = PathPatch(Path.make_compound_path(*paths))
        if polygon.outside_model_domain:
            domain_patches.append(patch)
        else:
            patches.append(patch)
            counts.append(polygon.n_phases)
            phase_groups.append(polygon.phases)
    if patches:
        low, high = min(counts), max(counts)
        palette = plt.get_cmap(cmap, high - low + 1)
        norm = BoundaryNorm(np.arange(low - 0.5, high + 1.5), palette.N)
        collection = PatchCollection(
            patches,
            cmap=palette,
            norm=norm,
            edgecolors="none",
            alpha=fill_alpha,
            zorder=0,
        )
        if assemblage_colors:
            colours = {
                tuple(sorted((key,) if isinstance(key, str) else key)): colour
                for key, colour in assemblage_colors.items()
            }
            field_colours = []
            for phases, count in zip(phase_groups, counts):
                field_names = tuple(
                    sorted(
                        phase_names[i].split(" #")[0]
                        for i in phases
                        if 0 <= i < len(phase_names)
                    )
                )
                field_colours.append(
                    colours.get(
                        tuple(sorted(phases)),
                        colours.get(field_names, palette(norm(count))),
                    )
                )
            collection.set_facecolors(field_colours)
        else:
            collection.set_array(np.asarray(counts, dtype=float))
        ax.add_collection(collection)
        if colorbar and not assemblage_colors:
            bar = fig.colorbar(collection, ax=ax, ticks=sorted(set(counts)), pad=0.025)
            bar.set_label("Number of phases")
    if domain_patches:
        ax.add_collection(
            PatchCollection(
                domain_patches,
                facecolors="#eeeeee",
                edgecolors="#888888",
                hatch="///",
                linewidths=0.5,
                zorder=0,
            )
        )
    uncoloured = sum(
        p.n_phases <= 0 and not p.outside_model_domain for p in geometry.polygons
    )
    unlabelled = sum(p.n_phases > 0 and not p.phases for p in geometry.polygons)
    issues = []
    if uncoloured:
        issues.append(
            f"{uncoloured} regions have unresolved phase counts and were left uncoloured"
        )
    if label_assemblages and unlabelled:
        issues.append(
            f"{unlabelled} regions have ambiguous assemblages and were left unlabelled"
        )
    if issues:
        warnings.warn(
            "; ".join(issues)
            + "; inspect pseudosection_field_polygons(result).diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )

    aliases = dict(phase_aliases or {})

    def assemblage_text(phases):
        names = []
        for phase in phases:
            if phase < 0 or phase >= len(phase_names):
                return ""
            name = phase_names[phase].split(" #")[0]
            names.append(aliases.get(name, name))
        ordered = dict.fromkeys(names)
        return " ".join(
            (str(names.count(name)) if names.count(name) > 1 else "") + name
            for name in ordered
        )

    # Render the same native subdivision used by the fills and labels. Drawing
    # the original traces would reintroduce the dissolved internal segments.
    if geometry.boundary_lines:
        outlines = LineCollection(
            [coordinates(s) for s in geometry.boundary_lines],
            colors=line_color,
            linewidths=line_width,
            capstyle=line_capstyle,
            joinstyle=line_joinstyle,
            zorder=2,
        )
        # Matplotlib otherwise simplifies long outlines while leaving the
        # filled polygons intact. Retain the same calculated points in both.
        for path in outlines.get_paths():
            path.should_simplify = False
        ax.add_collection(outlines)
    incomplete = []
    for line in _value(result, "boundaries", []):
        points = _value(line, "points", [])
        if not points:
            continue
        if "closed loop" not in _value(line, "termination", ""):
            for i, name in [(0, "start_node"), (-1, "end_node")]:
                if _value(line, name, -1) < 0:
                    incomplete.append(points[i])
    if show_nodes:
        visible_nodes = set(geometry.boundary_nodes)
        for node in _value(result, "nodes", []):
            if (
                _value(node, "incident_lines", [])
                and _value(node, "kind") == "junction"
                and _value(node, "id") in visible_nodes
            ):
                ax.plot(
                    *point_coordinates(node),
                    "o",
                    color=line_color,
                    ms=3,
                    zorder=3,
                )
    if show_unresolved:
        for node in _value(result, "nodes", []):
            if (
                _value(node, "kind") == "critical_point"
                and len(_value(node, "incident_lines", [])) == 1
                and not _value(node, "is_verified_critical_endpoint", False)
            ):
                incomplete.append(node)
        if incomplete:
            ax.scatter(
                [point_coordinates(p)[0] for p in incomplete],
                [point_coordinates(p)[1] for p in incomplete],
                marker="o",
                facecolors="none",
                edgecolors="crimson",
                s=15,
                label="Unresolved boundary endpoint",
                zorder=4,
            )
        failures = [
            s
            for s in _value(result, "samples", [])
            if not _value(s, "success", False)
            and not _value(s, "outside_model_domain", False)
        ]
        if failures:
            ax.scatter(
                [point_coordinates(s)[0] for s in failures],
                [point_coordinates(s)[1] for s in failures],
                marker="x",
                color="crimson",
                s=15,
                label="Unresolved equilibrium",
                zorder=4,
            )
        handles, labels = ax.get_legend_handles_labels()
        if domain_patches:
            handles.append(Patch(facecolor="#eeeeee", edgecolor="#888888", hatch="///"))
            labels.append("Outside solid EOS domain")
        if uncoloured:
            handles.append(Patch(facecolor="white", edgecolor=".65"))
            labels.append("Unresolved field")
        if handles:
            ax.legend(handles, labels, fontsize=label_fontsize, loc="upper left")
    pressure_range = _value(
        result, "pressure_range", _value(result, "calculation_pressure_range_Pa")
    )
    temperature_range = _value(
        result, "temperature_range", _value(result, "temperature_range_K")
    )
    ranges = dict(
        P=pressure_range,
        T=temperature_range,
        S=_value(result, "entropy_range"),
        V=_value(result, "volume_range"),
        X=_value(result, "composition_range"),
    )
    limits = coordinates(np.column_stack([ranges[axis] for axis in diagram]))
    labels = dict(
        P=f"Pressure ({pressure_unit})",
        T=f"Temperature ({'°C' if temperature_unit == 'C' else 'K'})",
        S=(r"Entropy ($k_B$/atom)" if per_atom else f"Entropy ({entropy_unit})"),
        V=(
            "Density (kg/m³)"
            if density
            else f"Volume ({'cm³' if volume_unit == 'cm3' else 'm³'})"
        ),
        X=composition_label,
    )
    ax.set(
        xlim=sorted(limits[:, 0]),
        ylim=sorted(limits[:, 1]),
        xlabel=labels[diagram[order[0]]],
        ylabel=labels[diagram[order[1]]],
    )
    ax.pseudosection_label_key = {}
    if label_assemblages:
        # Text metrics depend on the final axes limits, colorbar and layout.
        # Thermodynamic and polygon containment calculations remain native.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        padding = 0.18 * label_fontsize * fig.dpi / 72.0
        occupied, numbered = [], []
        text_style = dict(
            ha="center",
            va="center",
            fontsize=label_fontsize,
            zorder=5,
            bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none", alpha=0.8),
        )

        def box_for(text):
            bounds = text.get_window_extent(renderer)
            return Bbox.from_extents(
                bounds.x0 - padding,
                bounds.y0 - padding,
                bounds.x1 + padding,
                bounds.y1 + padding,
            )

        def fits(polygon, bounds):
            corners = native_coordinates(
                ax.transData.inverted().transform(bounds.get_points())
            )
            centre = corners.mean(axis=0)
            half = np.abs(corners[1] - corners[0]) * 0.5
            return polygon.contains_rectangle(centre, half)

        for polygon in geometry.polygons:
            number = polygon.field_number
            if polygon.n_phases <= 0 or not polygon.phases:
                continue
            name = assemblage_text(polygon.phases)
            if not name:
                continue
            location = coordinates([polygon.label_position])[0]
            text = ax.text(*location, name, **text_style)
            fitted = False
            variants = [name] + [
                textwrap.fill(
                    name, width=width, break_long_words=False, break_on_hyphens=False
                )
                for width in (32, 24, 20, 16, 12)
            ]
            for variant in dict.fromkeys(variants):
                text.set_text(variant)
                bounds = box_for(text)
                if fits(polygon, bounds):
                    occupied.append(bounds)
                    fitted = True
                    break
            if not fitted:
                text.remove()
                numbered.append((number, polygon, name, location))

        # Place full names first. Numeric callouts can then avoid every full
        # label and every earlier number, while keeping the chosen font size.
        for number, polygon, name, location in numbered:
            ax.pseudosection_label_key[number] = name
            text = ax.text(*location, str(number), **text_style)
            bounds = box_for(text)
            if fits(polygon, bounds) and not any(
                bounds.overlaps(old) for old in occupied
            ):
                occupied.append(bounds)
                continue
            anchor = ax.transData.transform(location)
            half = np.array([bounds.width, bounds.height]) * 0.5
            axes_box = ax.get_window_extent()
            placed = None
            spacing = max(bounds.width, bounds.height) + 2.0
            for radius in range(1, 41):
                angles = np.linspace(
                    0.0, 2.0 * np.pi, max(8, radius * 8), endpoint=False
                )
                for angle in angles:
                    centre = anchor + spacing * radius * np.array(
                        [np.cos(angle), np.sin(angle)]
                    )
                    candidate = Bbox.from_extents(*(centre - half), *(centre + half))
                    if (
                        candidate.x0 < axes_box.x0
                        or candidate.x1 > axes_box.x1
                        or candidate.y0 < axes_box.y0
                        or candidate.y1 > axes_box.y1
                        or any(candidate.overlaps(old) for old in occupied)
                    ):
                        continue
                    placed = (centre, candidate)
                    break
                if placed is not None:
                    break
            if placed is None:
                # Extremely crowded diagrams still retain a readable number
                # and a key entry; no font shrinking or lost assemblage names.
                occupied.append(bounds)
                continue
            text.remove()
            centre, bounds = placed
            position = ax.transData.inverted().transform(centre)
            ax.annotate(
                str(number),
                xy=location,
                xytext=position,
                arrowprops=dict(
                    arrowstyle="-", lw=0.45, color=".25", shrinkA=2.0, shrinkB=0.0
                ),
                **text_style,
            )
            occupied.append(bounds)

        if label_key_path is not None and ax.pseudosection_label_key:
            destination = FilePath(label_key_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                "# Pseudosection field assemblages",
                "",
                f"All field labels use {label_fontsize:g} pt.",
                "",
                "Numbers identify individual closed regions after any assemblage merging. A fine leader line connects",
                "displaced numbers to fields too narrow to hold the number itself.",
                "",
                "| Field number | Assemblage |",
                "| ---: | --- |",
            ]
            for number, name in ax.pseudosection_label_key.items():
                escaped_name = name.replace("|", r"\|")
                rows.append(f"| {number} | {escaped_name} |")
            destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return fig, ax


def plot_pseudosection_contours(
    result,
    contours,
    ax,
    *,
    color="crimson",
    line_width=0.8,
    line_style="solid",
    label=None,
    variable=None,
    value=None,
    label_format=None,
    label_placement="auto",
    label_fontsize=7.0,
    inline_labels=True,
    **display_options,
):
    """Overlay previously calculated constraint contours on an axes.

    This performs no equilibrium solves. Contours are the result of
    pseudosection_contours or its JSON dictionary. Display units and axis
    order inherit those of plot_pseudosection on the supplied axes.
    For other axes, supply the units used to draw their phase diagram.
    The label names this contour level in the legend and, optionally, along
    sufficiently long segments; all text uses label_fontsize.

    Alternatively supply ``value`` and ``variable`` ('P', 'T', 'S', 'V' or
    'X') to label an SI contour target in the panel's display units, even
    when that variable is not a diagram axis. Without variable, value is
    displayed unchanged, useful for composition ratios. An explicit label
    overrides automatic formatting. ``label_format`` is a Python numeric
    format specification, e.g. '.2f'.

    ``label_placement='auto'`` labels separate segments automatically.
    'single' places one label on the longest segment in display dimensions;
    'coexistence' first prefers fields with the most coexisting phases. These
    options help when contours nearly overlap in lower-variance fields.
    Return (figure, axes).
    """
    import numpy as np
    from matplotlib.contour import ContourSet

    if not np.isfinite(label_fontsize) or label_fontsize <= 0.0:
        raise ValueError("label_fontsize must be positive and finite.")
    if label_placement not in ("auto", "single", "coexistence"):
        raise ValueError("label_placement must be 'auto', 'single' or 'coexistence'.")
    if variable is not None and variable not in ("P", "T", "S", "V", "X"):
        raise ValueError("variable must be 'P', 'T', 'S', 'V' or 'X'.")
    options = dict(getattr(ax, "_pseudosection_display_options", {}))
    for key, option in display_options.items():
        if option is not None:
            if key in options and options[key] != option:
                raise ValueError("Contour display units must match the phase diagram.")
            options[key] = option
    diagram, names, _, _, _, coordinates, _ = _coordinate_transforms(result, **options)
    if _value(contours, "diagram_type") != diagram:
        raise ValueError("Contours and pseudosection must use the same diagram type.")
    if label is None and value is not None:
        display_value = float(value)
        if not np.isfinite(display_value):
            raise ValueError("Contour label values must be finite.")
        if variable is not None:
            scales, t_offset, mass = _coordinate_units(result, variable, options)
            if variable == "V" and options.get("volume_unit", "m3") == "kg/m3":
                if display_value <= 0.0:
                    raise ValueError("Density labels require a positive volume.")
                display_value = mass / display_value
            else:
                display_value = (
                    display_value - (t_offset if variable == "T" else 0.0)
                ) / scales[variable]
        if label_format is None:
            label_format = (
                ".3g"
                if variable == "S"
                or (variable == "V" and options.get("volume_unit", "m3") != "kg/m3")
                else "g"
            )
        label = format(display_value, label_format)
    segments, phase_counts = [], []
    for line in _value(contours, "lines", []):
        points = _value(line, "points", [])
        if len(points) >= 2:
            segments.append(
                coordinates([[_value(p, names[c]) for c in diagram] for p in points])
            )
            phase_counts.append(len(_value(line, "phases", [])))
    if segments:
        lines = ContourSet(
            ax,
            [0.0],
            [segments],
            colors=color,
            linewidths=line_width,
            linestyles=line_style,
            zorder=3,
        )
        ax.collections[-1].set_label(label)
        if label is not None and inline_labels:
            positions = None
            if label_placement != "auto":
                lengths = [
                    np.linalg.norm(
                        np.diff(ax.transData.transform(segment), axis=0), axis=1
                    )
                    for segment in segments
                ]
                candidates = [
                    i
                    for i in range(len(segments))
                    if label_placement != "coexistence"
                    or phase_counts[i] == max(phase_counts)
                ]
                index = max(candidates, key=lambda i: lengths[i].sum())
                if lengths[index].sum() > 0.0:
                    cumulative = np.r_[0.0, np.cumsum(lengths[index])]
                    halfway = cumulative[-1] / 2.0
                    midpoint = np.searchsorted(cumulative, halfway, side="right") - 1
                    fraction = (halfway - cumulative[midpoint]) / lengths[index][
                        midpoint
                    ]
                    left, right = segments[index][midpoint : midpoint + 2]
                    positions = [left + fraction * (right - left)]
                else:
                    return ax.figure, ax
            lines.clabel(
                fmt={0.0: str(label)},
                fontsize=label_fontsize,
                inline=True,
                manual=positions,
            )
    return ax.figure, ax
