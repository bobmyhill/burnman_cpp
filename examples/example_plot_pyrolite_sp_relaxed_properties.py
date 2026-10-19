#!/usr/bin/env python3
# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Plot relaxed/unrelaxed alpha, beta_T and molar Cp from saved P-S equilibria.

Supply a phase-diagram JSON and --grid-solutions pointing to a grid_solutions
directory, its manifest.json, or an older grid_solutions.json.gz file. Restore
the original candidate models with --models (SLB2024 pyrolite by default).
Only response properties are evaluated: no equilibrium solves or retries run.
An expansivity.npz archive alone does not contain the retained equilibria
needed to calculate compressibility and heat capacity.

Relaxed responses allow phase amounts and compositions to adjust; unrelaxed
responses hold them fixed at the same saved equilibrium. Cp is per current
mole of assemblage, rather than per original 100 g bulk composition. Beta_T
is isothermal compressibility, not bulk modulus. Missing properties stay grey.
Each property uses the same colour scale for both modes. Alpha spans +/-5e-5
K^-1; beta_T and Cp span the central 99% of the displayed values, including
zero. Colour clipping never changes the saved numerical values.

Write a six-panel PNG/PDF, SI-unit arrays in properties.npz and arrays/*.npy,
labelled isotherms, and diagnostics. Chunked inputs keep equilibrium memory
bounded; --plot-resolution limits rendering only, preserving the saved grid.
Use --verbose for progress after each saved batch.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import load_example

grid_example = load_example(
    Path(__file__).with_name("example_grid_pyrolite_sp_expansivity.py")
)
PROPERTIES = (
    ("thermal_expansivity", r"$\alpha$ ($10^{-5}$ K$^{-1}$)", 1e5),
    ("isothermal_compressibility_reuss", r"$\beta_T$ (GPa$^{-1}$)", 1e9),
    ("molar_heat_capacity_p", r"$C_p$ (J mol$^{-1}$ K$^{-1}$)", 1.0),
)
MODES = ("relaxed", "unrelaxed")


def saved_batches(path):
    """Return full-grid axes and an iterator over retained equilibrium batches."""
    path = Path(path)
    if path.is_dir():
        path /= "manifest.json"
    if path.name == "manifest.json":
        metadata = json.loads(path.read_text())
        return (
            np.asarray(metadata["x"]),
            np.asarray(metadata["y"]),
            bm.load_property_grid_chunks(path),
        )
    grid = bm.load_property_grid(path)
    return grid.x, grid.y, iter([(slice(0, len(grid.y)), slice(0, len(grid.x)), grid)])


def evaluate_saved(path, phases, output_dir, *, verbose=False):
    """Evaluate both responses at saved states, retaining each property's gaps.

    Read solution batches without changing them; write disk-backed numeric
    arrays and stream diagnostics so neither grows with retained solution RAM.
    """
    x, y, batches = saved_batches(path)
    if min(len(x), len(y)) < 2:
        raise ValueError("Plotting requires at least two points on each grid axis.")
    arrays_dir = Path(output_dir) / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.save(arrays_dir / "x.npy", x)
    np.save(arrays_dir / "y.npy", y)
    names = [name for name, _, _ in PROPERTIES]
    arrays = {}
    for key in [f"{mode}_{name}" for mode in MODES for name in names] + ["temperature"]:
        arrays[key] = np.lib.format.open_memmap(
            arrays_dir / f"{key}.npy", mode="w+", dtype=float, shape=(len(y), len(x))
        )
        arrays[key][:] = np.nan
    completed = 0
    with (Path(output_dir) / "diagnostics.jsonl").open("w") as diagnostics:
        for rows, cols, grid in batches:
            if grid.diagram_type != "PS":
                raise ValueError("This example requires a pressure-entropy (PS) grid.")
            if not (
                np.array_equal(grid.x, x[cols]) and np.array_equal(grid.y, y[rows])
            ):
                raise ValueError("Saved chunk coordinates disagree with the grid axes.")
            for mode in MODES:
                replay = bm.evaluate_property_grid(
                    grid, phases, names + ["temperature"], relaxed=(mode == "relaxed")
                )
                for name in names:
                    arrays[f"{mode}_{name}"][rows, cols] = replay.values[name]
                if mode == "relaxed":
                    arrays["temperature"][rows, cols] = replay.values["temperature"]
                if replay.diagnostics:
                    diagnostics.write(
                        json.dumps(
                            dict(
                                mode=mode,
                                rows=[rows.start, rows.stop],
                                columns=[cols.start, cols.stop],
                                diagnostics=replay.diagnostics,
                            )
                        )
                        + "\n"
                    )
                del replay
            completed += len(grid.x) * len(grid.y)
            if verbose:
                print(
                    f"Properties: {completed}/{len(x) * len(y)} saved points",
                    flush=True,
                )
            del grid
    for array in arrays.values():
        array.flush()
    grid_example.save_array_archive(arrays_dir, Path(output_dir) / "properties.npz")
    return x, y, arrays


def plot(diagram, x, y, arrays, *, isotherms=(), max_plot_resolution=1001):
    """Compare response modes using a common scale per property on native P-S axes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    columns = np.linspace(0, len(x) - 1, min(len(x), max_plot_resolution), dtype=int)
    rows = np.linspace(0, len(y) - 1, min(len(y), max_plot_resolution), dtype=int)
    extent = (x[0] / 1e9, x[-1] / 1e9, y[0], y[-1])
    dx = (extent[1] - extent[0]) / (len(columns) - 1) / 2
    dy = (extent[3] - extent[2]) / (len(rows) - 1) / 2
    image_extent = (extent[0] - dx, extent[1] + dx, extent[2] - dy, extent[3] + dy)
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 15), sharex=True, sharey=True, layout="constrained"
    )
    for row, (name, label, factor) in enumerate(PROPERTIES):
        values = [
            grid_example.array_sample(arrays[f"{mode}_{name}"], np.ix_(rows, columns))
            * factor
            for mode in MODES
        ]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        if row == 0:
            low, high = -5.0, 5.0
        elif finite.size:
            low, high = np.percentile(finite, (0.5, 99.5))
            low, high = min(0.0, low), max(0.0, high)
            if low == high:
                high = low + 1.0
        else:
            low, high = 0.0, 1.0
        cmap = plt.get_cmap("bwr_r" if row == 0 else "viridis").with_extremes(
            bad="0.85"
        )
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            # Preserve signed infinities at the colour limits, masking only NaNs.
            image = ax.imshow(
                np.ma.masked_where(
                    np.isnan(values[col]), np.clip(values[col], low, high)
                ),
                extent=image_extent,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=Normalize(low, high),
                rasterized=True,
            )
            bm.plot_phase_diagram(
                diagram,
                ax=ax,
                fill_alpha=0.0,
                colorbar=False,
                line_width=0.35,
                pressure_unit="GPa",
                entropy_unit="J/K",
                swap_axes=True,
                label_assemblages=False,
                label_key_path=None,
                show_unresolved=False,
            )
            for record in isotherms:
                bm.plot_phase_diagram_contours(
                    diagram,
                    dict(record, diagram_type="PS"),
                    ax,
                    color="orange",
                    line_width=0.8,
                    label=f"{record['temperature']:g} K",
                    label_placement="single",
                )
            ax.set(
                xlim=extent[:2], ylim=extent[2:], title=f"{mode.capitalize()} {label}"
            )
            ax.set_xlabel("Pressure (GPa)" if row == 2 else "")
            if col:
                ax.set_ylabel("")
        fig.colorbar(image, ax=list(axes[row]), label=label, extend="both", shrink=0.9)
    fig.suptitle(
        "Pyrolite: responses at saved equilibria\n"
        "Orange: equilibrium isotherms · Grey: unavailable properties"
    )
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagram-json", type=Path, required=True)
    parser.add_argument("--grid-solutions", type=Path, required=True)
    parser.add_argument(
        "--models",
        default=str(Path(__file__).with_name("example_pyrolite_phase_diagram.py")),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pyrolite_sp_properties_output")
    )
    parser.add_argument("--plot-resolution", type=int, default=1001)
    parser.add_argument(
        "--isotherms",
        type=float,
        nargs="*",
        default=(1000.0, 1500.0, 2000.0),
        metavar="T_K",
        help="Kelvin levels; omit values to disable isotherms.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.plot_resolution < 2:
        parser.error("--plot-resolution must be >= 2")
    diagram = bm.load_phase_diagram(args.diagram_json)
    if diagram.get("diagram_type") != "PS":
        parser.error("--diagram-json must contain a pressure-entropy (PS) diagram")
    x, y, arrays = evaluate_saved(
        args.grid_solutions,
        load_example(args.models).candidate_phases(),
        args.output_dir,
        verbose=args.verbose,
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(
            dict(
                source_json=str(args.diagram_json.resolve()),
                grid_solutions=str(args.grid_solutions.resolve()),
                models=args.models,
                equilibrium_solves=0,
                shape=[len(y), len(x)],
                finite_points={
                    key: int(np.count_nonzero(np.isfinite(a)))
                    for key, a in arrays.items()
                },
                infinite_points={
                    key: int(np.count_nonzero(np.isinf(a))) for key, a in arrays.items()
                },
                detailed_diagnostics="diagnostics.jsonl",
            ),
            indent=2,
        )
        + "\n"
    )
    temperatures = SimpleNamespace(
        x=x,
        y=y,
        diagram_type="PS",
        has_solutions=True,
        values={"temperature": arrays["temperature"]},
    )
    isotherms = grid_example.calculate_isotherms(temperatures, args.isotherms)
    (args.output_dir / "isotherms.json").write_text(json.dumps(isotherms) + "\n")
    fig, _ = plot(
        diagram,
        x,
        y,
        arrays,
        isotherms=isotherms,
        max_plot_resolution=args.plot_resolution,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"pyrolite_sp_properties.{suffix}", dpi=200)
    print(f"Figure and arrays: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
