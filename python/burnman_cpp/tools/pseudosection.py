"""Pseudosection figures; polygon geometry is assembled by the C++ library."""

from collections.abc import Mapping
import warnings

from .. import _core


def _value(record, name, default=None):
    return (
        record.get(name, default)
        if isinstance(record, Mapping)
        else getattr(record, name, default)
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
    show_nodes=False,
    show_unresolved=True,
    close_domain=True,
    merge_fields=True,
    tolerance=1.0e-8,
    pressure_unit="kbar",
    temperature_unit="C",
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
    Fields with conflicting or missing assemblage counts stay uncoloured and
    issue a warning. Each coexisting solution instance counts as one phase.
    Native ``excluded_regions`` are drawn with grey hatching: the solid model
    cannot be evaluated there within its required mechanically stable EOS domain.
    Such regions have no assemblage count and are distinct from solve failures.

    Pressure units are ``'Pa'``, ``'GPa'`` or ``'kbar'``; temperature units are
    ``'K'`` or ``'C'``. ``tolerance`` is a fraction of the calculation domain,
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

    pressure_scales = {"Pa": 1.0, "GPa": 1.0e9, "kbar": 1.0e8}
    if pressure_unit not in pressure_scales:
        raise ValueError("pressure_unit must be 'Pa', 'GPa' or 'kbar'.")
    if temperature_unit not in ("K", "C"):
        raise ValueError("temperature_unit must be 'K' or 'C'.")
    if not np.isfinite(label_fontsize) or label_fontsize <= 0.0:
        raise ValueError("label_fontsize must be positive and finite.")
    p_scale = pressure_scales[pressure_unit]
    t_offset = 273.15 if temperature_unit == "C" else 0.0
    if isinstance(result, Mapping):
        result = dict(result)
    geometry = _core.pseudosection_field_polygons(
        result, tolerance, close_domain, merge_fields
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    fig = ax.figure
    ax.pseudosection_geometry = geometry

    def coordinates(vertices):
        vertices = np.asarray(vertices)
        return np.column_stack((vertices[:, 1] - t_offset, vertices[:, 0] / p_scale))

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
    if geometry.boundary_segments:
        ax.add_collection(
            LineCollection(
                [coordinates(s) for s in geometry.boundary_segments],
                colors=line_color,
                linewidths=line_width,
                zorder=2,
            )
        )
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
                    _value(node, "temperature") - t_offset,
                    _value(node, "pressure") / p_scale,
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
                [_value(p, "temperature") - t_offset for p in incomplete],
                [_value(p, "pressure") / p_scale for p in incomplete],
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
                [_value(s, "temperature") - t_offset for s in failures],
                [_value(s, "pressure") / p_scale for s in failures],
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
    ax.set(
        xlim=tuple(t - t_offset for t in temperature_range),
        ylim=tuple(p / p_scale for p in pressure_range),
        xlabel=f"Temperature ({'°C' if temperature_unit == 'C' else 'K'})",
        ylabel=f"Pressure ({pressure_unit})",
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
            corners = ax.transData.inverted().transform(bounds.get_points())
            centre = corners.mean(axis=0)
            half = np.abs(corners[1] - corners[0]) * 0.5
            return polygon.contains_rectangle(
                [centre[1] * p_scale, centre[0] + t_offset],
                [half[1] * p_scale, half[0]],
            )

        for number, polygon in enumerate(geometry.polygons, 1):
            if polygon.n_phases <= 0 or not polygon.phases:
                continue
            name = assemblage_text(polygon.phases)
            if not name:
                continue
            location = (
                polygon.label_position[1] - t_offset,
                polygon.label_position[0] / p_scale,
            )
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
            rows.extend(
                f"| {number} | {name.replace('|', r'\|')} |"
                for number, name in ax.pseudosection_label_key.items()
            )
            destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return fig, ax
