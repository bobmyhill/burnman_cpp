#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Figure 4 comparison: relaxed and isomorphic thermal expansivity of pyrolite.

Use --diagram-json to grid an existing completed SLB24 pyrolite diagram.
Without it, calculate the paper's 0–40 GPa, 800–2800 K window first. The two
panels share the paper's red/white/blue scale (-5 to +5 in 10^-5 K^-1), phase
boundaries, and equilibrium isentropes starting at 1000, 1500 and 2000 K at
zero pressure. Save the property arrays as NPZ and the figure as PNG/PDF.

This uses the repository's SLB2024 pyrolite composition (3% initial ferric
iron), rather than the paper's SLB2021 HeFESTo parameterization. It illustrates
the same calculation, not identical numerical results. The solid-only model
cannot predict melting; optionally supply published solidus/liquidus curves
with --melting-curves (CSV columns pressure_GPa,solidus_K,liquidus_K).
The paper used a much finer 0.01 GPa / 1 K property grid; --resolution controls
the grid here independently of the saved boundary geometry.
Retain all accepted grid equilibria in grid_solutions.json.gz so more properties
can be evaluated later without repeating equilibrium solves.
Isentropes interpolate total entropy from these saved equilibria; their accuracy
therefore depends on the property-grid resolution.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import load_example

pyrolite = load_example(Path(__file__).with_name("example_pyrolite_phase_diagram.py"))
PAPER_URL = "https://doi.org/10.1093/gji/ggab394"


def calculate_grids(
    diagram,
    *,
    resolution=(201, 161),
    pressure_range=(0.0, 40e9),
    temperature_range=(800.0, 2800.0),
):
    """Re-equilibrate saved fields and return total/frozen alpha grids."""
    data = diagram if isinstance(diagram, dict) else diagram.to_dict()
    data = dict(data, settings=dict(data.get("settings", {}), verbose=False))
    if data.get("diagram_type", "PT") != "PT":
        raise ValueError("The Figure 4 comparison requires a PT diagram.")
    low = max(data["temperature_range"][0], temperature_range[0])
    high = min(data["temperature_range"][1], temperature_range[1])
    if low >= high:
        raise ValueError("The saved diagram does not overlap the temperature window.")
    plow = max(data["pressure_range"][0], pressure_range[0])
    phigh = min(data["pressure_range"][1], pressure_range[1])
    if plow >= phigh:
        raise ValueError("The saved diagram does not overlap the pressure window.")
    x = np.linspace(plow, phigh, resolution[0])
    y = np.linspace(low, high, resolution[1])
    grids = {}
    for name, relaxed in (("relaxed", True), ("unrelaxed", False)):
        if relaxed:
            grids[name] = bm.phase_diagram_grid(
                data,
                pyrolite.candidate_phases(),
                "thermal_expansivity",
                x=x,
                y=y,
                relaxed=True,
                store_solutions=True,
            )
        else:
            grids[name] = bm.evaluate_property_grid(
                grids["relaxed"],
                pyrolite.candidate_phases(),
                "thermal_expansivity",
                relaxed=False,
            )
        grid = grids[name]
        print(
            f"{name}: {np.count_nonzero(grid.success)}/{grid.success.size} points; "
            f"{grid.equilibrium_solves} solves; {len(grid.diagnostics)} diagnostics",
            flush=True,
        )
    return grids


def calculate_isentropes(grid, potential_temperatures=(1000.0, 1500.0, 2000.0)):
    """Contour saved total entropy, with no further equilibrium calculations.

    Interpolate S(P=0, T0) between adjacent valid grid temperatures, then contour
    that entropy on the PT grid. Unsolved points remain masked; interpolation
    never bridges missing surface states or extrapolates outside the grid.
    """
    import contourpy

    if grid.diagram_type != "PT" or not grid.has_solutions:
        raise ValueError("Isentropes require a PT grid with saved solutions.")
    x, y = grid.x, grid.y
    if x[0] != 0.0:
        raise ValueError(
            "Potential-temperature isentropes require a grid reaching zero pressure."
        )
    entropy = np.full((len(y), len(x)), np.nan)
    for row, col in np.argwhere(grid.equilibrium_success):
        entropy[row, col] = grid.solution_at(int(row), int(col)).state.entropy
    contours = contourpy.contour_generator(
        x=x,
        y=y,
        z=np.ma.masked_invalid(entropy),
        corner_mask=False,
        line_type="Separate",
    )
    records = []
    for t0 in potential_temperatures:
        record = dict(potential_temperature=float(t0), lines=[], diagnostics=[])
        records.append(record)
        if not np.isfinite(t0) or not y[0] <= t0 <= y[-1]:
            record["diagnostics"].append(
                f"Potential temperature {t0:g} K is outside the saved grid."
            )
            continue
        upper = int(np.searchsorted(y, t0))
        if y[upper] == t0:
            target = entropy[upper, 0]
        else:
            lower = upper - 1
            fraction = (t0 - y[lower]) / (y[upper] - y[lower])
            target = (1 - fraction) * entropy[lower, 0] + fraction * entropy[upper, 0]
        if not np.isfinite(target):
            record["diagnostics"].append(
                f"Saved zero-pressure entropy is unavailable at {t0:g} K."
            )
            continue
        record["entropy"] = float(target)  # Total J/K, on the saved bulk scale.
        record["lines"] = [
            dict(
                points=[dict(pressure=float(p), temperature=float(t)) for p, t in line]
            )
            for line in contours.lines(float(target))
            if len(line) >= 2
        ]
        if not record["lines"]:
            record["diagnostics"].append("No isentrope was found on the valid grid.")
        print(f"{t0:g} K isentrope: {len(record['lines'])} grid segments", flush=True)
    return records


def plot(diagram, grids, *, isentropes=(), melting_curves=None):
    """Return a Figure 4 style pair of panels; keep invalid points masked."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.08)
    cmap = plt.get_cmap("bwr_r").with_extremes(bad="0.85")
    norm = Normalize(-5, 5)
    data = diagram if isinstance(diagram, dict) else diagram.to_dict()
    polygons = bm.phase_diagram_field_polygons(diagram).polygons
    for ax, key, title in zip(
        axes,
        ("relaxed", "unrelaxed"),
        (r"Equilibrium $\alpha$", r"Isomorphic $\alpha_{\mathrm{iso}}$"),
    ):
        grid = grids[key]
        image = ax.pcolormesh(
            grid.x / 1e9,
            grid.y,
            np.ma.masked_invalid(grid.values["thermal_expansivity"] * 1e5),
            cmap=cmap,
            norm=norm,
            shading="nearest",
            rasterized=True,
        )
        bm.plot_phase_diagram(
            diagram,
            ax=ax,
            fill_alpha=0.0,
            colorbar=False,
            line_width=0.4,
            pressure_unit="GPa",
            temperature_unit="K",
            swap_axes=True,
            label_assemblages=False,
            label_key_path=None,
            show_unresolved=False,
        )
        for record in isentropes:
            for line in record["lines"]:
                ax.plot(
                    [p["pressure"] / 1e9 for p in line["points"]],
                    [p["temperature"] for p in line["points"]],
                    color="orange",
                    lw=1.1,
                    zorder=6,
                )
        if melting_curves is not None:
            for name in ("solidus_K", "liquidus_K"):
                ax.plot(
                    melting_curves["pressure_GPa"],
                    melting_curves[name],
                    color="limegreen",
                    lw=1.1,
                    zorder=6,
                )
        ax.set(
            xlim=(grid.x[0] / 1e9, grid.x[-1] / 1e9),
            ylim=(grid.y[0], grid.y[-1]),
            title=title,
        )
        # Label fields whose complete text fits inside the saved polygon.
        # Narrow fields remain visible as lines without dense numeric callouts.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for polygon in sorted(polygons, key=lambda p: p.area, reverse=True):
            p, t = polygon.label_position
            if polygon.area < 0.001 or not (
                grid.x[0] <= p <= grid.x[-1] and grid.y[0] <= t <= grid.y[-1]
            ):
                continue
            counts = Counter(
                data["phase_names"][i].split(" #")[0] for i in polygon.phases
            )
            labels = [f"{n if n > 1 else ''}{name}" for name, n in counts.items()]
            label = "\n".join(
                " ".join(labels[i : i + 3]) for i in range(0, len(labels), 3)
            )
            if not label:
                continue
            text = ax.text(
                p / 1e9, t, label, ha="center", va="center", fontsize=6.5, zorder=7
            )
            corners = ax.transData.inverted().transform(
                text.get_window_extent(renderer).get_points()
            )
            corners[:, 0] *= 1e9
            if not polygon.contains_rectangle(
                corners.mean(axis=0), np.diff(corners, axis=0)[0] / 2
            ):
                text.remove()
    axes[1].set_ylabel("")
    cax = fig.add_axes([0.32, 0.91, 0.36, 0.02])
    fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        extend="both",
        label=r"Thermal expansivity ($10^{-5}$ K$^{-1}$)",
    )
    fig.text(
        0.5,
        0.035,
        "SLB2024 pyrolite · comparison with Stixrude & Lithgow-Bertelloni Figure 4\n"
        "Orange: equilibrium isentropes · Grey: unavailable properties / model domain",
        ha="center",
        fontsize=9,
    )
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagram-json", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pyrolite_expansivity_output")
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=(401, 401), metavar=("NP", "NT")
    )
    parser.add_argument("--seeds", type=int, default=41)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while calculating the original phase diagram.",
    )
    parser.add_argument("--no-isentropes", action="store_true")
    parser.add_argument("--melting-curves", type=Path)
    args = parser.parse_args()
    if min(args.resolution) < 2:
        parser.error("--resolution needs two counts >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.diagram_json:
        diagram = bm.load_phase_diagram(args.diagram_json)
    else:
        result = pyrolite.calculate(
            args.seeds,
            verbose=args.verbose,
            pressure_range=(0.0, 40e9),
            temperature_range=(800.0, 2800.0),
        )
        bm.save_phase_diagram(result, args.output_dir / "phase_diagram.json")
        diagram = result.to_dict()
    grids = calculate_grids(diagram, resolution=args.resolution)
    bm.save_property_grid(grids["relaxed"], args.output_dir / "grid_solutions.json.gz")
    report = dict(
        paper=PAPER_URL,
        dataset="SLB24",
        resolution=args.resolution,
        source_resolved=diagram.get("resolved", False),
        source_diagnostics=diagram.get("diagnostics", []),
        source_spacing_diagnostics=diagram.get("spacing_diagnostics", []),
        diagnostics={key: grid.diagnostics for key, grid in grids.items()},
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "expansivity.npz",
        x=grids["relaxed"].x,
        y=grids["relaxed"].y,
        **{key: grid.values["thermal_expansivity"] for key, grid in grids.items()},
        field_numbers=grids["relaxed"].field_numbers,
    )
    isentropes = [] if args.no_isentropes else calculate_isentropes(grids["relaxed"])
    (args.output_dir / "isentropes.json").write_text(
        json.dumps(isentropes, indent=2) + "\n"
    )
    melting = (
        np.genfromtxt(args.melting_curves, delimiter=",", names=True)
        if args.melting_curves
        else None
    )
    fig, _ = plot(diagram, grids, isentropes=isentropes, melting_curves=melting)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"pyrolite_expansivity.{suffix}", dpi=200)
    print(f"Figure and arrays: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
