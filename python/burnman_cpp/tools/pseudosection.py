"""Pseudosection figures and refinement; calculations use the C++ library."""

from collections.abc import Mapping
import warnings

from .. import _core


def _value(record, name, default=None):
    return (
        record.get(name, default)
        if isinstance(record, Mapping)
        else getattr(record, name, default)
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

    pressure_scales = {"Pa": 1.0, "GPa": 1.0e9, "kbar": 1.0e8}
    if pressure_unit not in pressure_scales:
        raise ValueError("pressure_unit must be 'Pa', 'GPa' or 'kbar'.")
    if temperature_unit not in ("K", "C"):
        raise ValueError("temperature_unit must be 'K' or 'C'.")
    p_scale = pressure_scales[pressure_unit]
    t_offset = 273.15 if temperature_unit == "C" else 0.0
    if isinstance(result, Mapping):
        result = dict(result)
    diagram = _value(result, "diagram_type", "PT")
    if diagram not in ("PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"):
        raise ValueError("diagram_type must select two of P, T, S, V and X.")
    entropy_scales = {"J/K": 1.0, "kJ/K": 1000.0, "kB/atom": 1.0}
    volume_scales = {"m3": 1.0, "cm3": 1.0e-6, "kg/m3": 1.0}
    if entropy_unit not in entropy_scales or volume_unit not in volume_scales:
        raise ValueError(
            "entropy_unit must be 'J/K', 'kJ/K' or 'kB/atom'; "
            "volume_unit must be 'm3', 'cm3' or 'kg/m3'."
        )
    density = "V" in diagram and volume_unit == "kg/m3"
    per_atom = "S" in diagram and entropy_unit == "kB/atom"
    if density or per_atom:
        if "X" in diagram:
            raise ValueError("Density and per-atom entropy require a constant bulk.")
        bulk = _value(result, "composition_start", {})
        if not bulk:
            raise ValueError("Density and per-atom entropy require composition_start.")
        composition = _core.Composition(bulk, "molar")
        bulk_mass = sum(composition.mass_composition.values())
        entropy_scales["kB/atom"] = (
            sum(composition.atomic_composition.values()) * 8.31446261815324
        )
    names = dict(
        P="pressure",
        T="temperature",
        S="entropy",
        V="volume",
        X="composition_coordinate",
    )
    scales = dict(
        P=p_scale,
        T=1.0,
        S=entropy_scales[entropy_unit],
        V=volume_scales[volume_unit],
        X=1.0,
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
    Field checks still trigger boundary recovery if inconsistencies are found.
    Set ``settings.verbose = True`` to report refinement and checking progress.
    Save the returned result with ``result.to_dict()``. Omitted resolution
    retains ordinary boundary recovery and the saved settings by default.
    """
    import json
    from os import PathLike
    from pathlib import Path
    import numpy as np

    if isinstance(previous, (str, PathLike)):
        previous = json.loads(Path(previous).read_text())
    if resolution is None:
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
    _, _, order, density, _, _, _ = _coordinate_transforms(
        previous, pressure_unit, temperature_unit, entropy_unit, volume_unit, swap_axes
    )
    return _core.refine_pseudosection(
        composition,
        phases,
        previous,
        settings,
        resolution=tuple(int(count) for count in counts[order]),
        reciprocal_volume=density,
    )


def plot_pseudosection(
    result,
    ax=None,
    *,
    cmap="viridis",
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

    C++ joins the traced polylines into planar polygons, preserving holes and
    disconnected regions. By default the calculation frame closes fields at
    its edges. Unfinished lines are never extended to manufacture closure.
    Adjacent identified regions with identical assemblages are merged and
    their internal line segments removed. Set ``merge_fields=False`` to see
    the original subdivision. Disconnected regions retain separate labels.
    Boundaries where solution compositions switch remain visible even when
    both fields have the same assemblage names.
    Connected segments separating the same two assemblages are drawn as one
    continuous line between endpoint vertices. Junctions and separate paths
    remain distinct; the equilibrium points are retained.
    ``line_capstyle`` accepts ``'butt'``, ``'round'`` or ``'projecting'``;
    ``line_joinstyle`` accepts ``'miter'``, ``'round'`` or ``'bevel'``.
    Both default to ``'round'``.
    Fields with conflicting or missing assemblage counts stay uncoloured and
    issue a warning. Each coexisting solution instance counts as one phase.
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

    Label each identified closed field with its assemblage. ``phase_aliases``
    maps model names to abbreviations (e.g. ``{'g': 'gt'}``); repeated solution
    instances appear as ``2hb``. C++ selects interior label positions, avoiding
    holes. Every assemblage and numeric label uses ``label_fontsize`` (points).
    Labels that cannot fit, including their background, become unique field
    numbers. Numbers that also cannot fit use leader lines. ``label_key_path``
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

    patches, counts, domain_patches = [], [], []
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
        collection.set_array(np.asarray(counts, dtype=float))
        ax.add_collection(collection)
        if colorbar:
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

    phase_names = _value(result, "phase_names", [])
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
                and (not merge_fields or _value(node, "id") in visible_nodes)
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

        for number, polygon in enumerate(geometry.polygons, 1):
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
    Return (figure, axes).
    """
    import numpy as np
    from matplotlib.contour import ContourSet

    if not np.isfinite(label_fontsize) or label_fontsize <= 0.0:
        raise ValueError("label_fontsize must be positive and finite.")
    options = dict(getattr(ax, "_pseudosection_display_options", {}))
    for key, value in display_options.items():
        if value is not None:
            if key in options and options[key] != value:
                raise ValueError("Contour display units must match the phase diagram.")
            options[key] = value
    diagram, names, _, _, _, coordinates, _ = _coordinate_transforms(result, **options)
    if _value(contours, "diagram_type") != diagram:
        raise ValueError("Contours and pseudosection must use the same diagram type.")
    segments = []
    for line in _value(contours, "lines", []):
        points = _value(line, "points", [])
        if len(points) >= 2:
            segments.append(
                coordinates([[_value(p, names[c]) for c in diagram] for p in points])
            )
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
            lines.clabel(fmt={0.0: str(label)}, fontsize=label_fontsize, inline=True)
    return ax.figure, ax
