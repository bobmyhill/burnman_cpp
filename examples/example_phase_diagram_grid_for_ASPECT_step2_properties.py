#!/usr/bin/env python3
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

"""
This example is a follow-up to example_phase_diagram_grid_for_ASPECT.py.
It reads the saved P-S equilibria for the pyrolitic composition.
It then evaluates relaxed and unrelaxed
thermal expansivity alpha (1/K), isothermal compressibility beta_T (1/Pa)
and absolute isobaric heat capacity Cp (J/K) for the original 100 g bulk.

Relaxed responses allow phase amounts and endmember compositions to adjust
to pressure and temperature perturbations at fixed bulk composition.
Unrelaxed responses hold phase amounts and endmember compositions fixed.

The six panels show pressure in GPa (x axis) and total entropy in J/K (y axis).
Rows show alpha, beta_T and Cp. Columns show relaxed and unrelaxed responses.
Phase boundaries and the 1000, 1500 and 2000 K isotherms are overlaid.

Outputs share Step 1's phase_diagram_grid_for_ASPECT_output directory.
The figure is saved as pyrolite_sp_properties.png and .pdf. Full-resolution
SI-unit arrays are saved in properties.npz and arrays/*.npy, together with
isotherm coordinates and property-evaluation diagnostics. diagnostics.json
contains the summary; diagnostics.jsonl streams one JSON record per failed
property batch.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import burnman_cpp as bm
from burnman_cpp.minerals import load_dataset, model_sets
from burnman_cpp.tools.phase_diagram_helpers import save_figure

EXAMPLES = Path(__file__).resolve().parent
SOURCE_DIR = EXAMPLES / "phase_diagram_grid_for_ASPECT_output"
DIAGRAM = SOURCE_DIR / "refined.json"
GRID_SOLUTIONS = SOURCE_DIR / "grid_solutions"
MODEL_SET = model_sets.slb2024()
OUTPUT_DIR = SOURCE_DIR
PLOT_RESOLUTION = 1001
ISOTHERMS = (1000.0, 1500.0, 2000.0)  # K; use () to omit isotherms.

PROPERTIES = (
    ("thermal_expansivity", r"$\alpha$ ($10^{-5}$ K$^{-1}$)", 1e5),
    ("isothermal_compressibility_reuss", r"$\beta_T$ (GPa$^{-1}$)", 1e9),
    ("heat_capacity_p", r"$C_p$ (J K$^{-1}$)", 1.0),
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
                    grid,
                    phases,
                    names + ["temperature"],
                    relaxed=(mode == "relaxed"),
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
    np.savez_compressed(Path(output_dir) / "properties.npz", x=x, y=y, **arrays)
    return x, y, arrays


def calculate_isotherms(grid, temperatures, *, chunk_rows=256):
    """Contour saved temperatures on P-S axes, preserving gaps between samples."""
    import contourpy

    generator = contourpy.contour_generator(
        x=grid.x,
        y=grid.y,
        z=np.ma.masked_invalid(grid.values["temperature"]),
        corner_mask=False,
        chunk_size=(max(1, chunk_rows - 1), 0),
        line_type="Separate",
    )
    return [
        dict(
            temperature=float(temperature),
            resolved=True,
            diagnostics=[],
            lines=[
                dict(
                    points=[dict(pressure=float(p), entropy=float(s)) for p, s in path]
                )
                for path in generator.lines(float(temperature))
            ],
        )
        for temperature in temperatures
    ]


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
            arrays[f"{mode}_{name}"][np.ix_(rows, columns)] * factor for mode in MODES
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


if __name__ == "__main__":
    diagram = bm.load_phase_diagram(DIAGRAM)
    if diagram.get("diagram_type") != "PS":
        raise ValueError("DIAGRAM must contain a pressure-entropy (PS) diagram.")
    models = diagram.get("workflow", {}).get("models", {})
    if "phase_names" in models:
        dataset = models["dataset"].rsplit(".", 1)[-1]
        phases, _ = load_dataset(dataset, models["phase_names"])
    else:
        if models and models != MODEL_SET.to_dict():
            raise ValueError(
                "Saved model set does not match the configured SLB2024 models."
            )
        phases = MODEL_SET.phases
    x, y, arrays = evaluate_saved(
        GRID_SOLUTIONS,
        phases,
        OUTPUT_DIR,
        verbose=True,
    )
    (OUTPUT_DIR / "diagnostics.json").write_text(
        json.dumps(
            dict(
                source_json=str(DIAGRAM.resolve()),
                grid_solutions=str(GRID_SOLUTIONS.resolve()),
                models=models or MODEL_SET.to_dict(),
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
    isotherms = calculate_isotherms(temperatures, ISOTHERMS)
    (OUTPUT_DIR / "isotherms.json").write_text(json.dumps(isotherms) + "\n")
    fig, _ = plot(
        diagram,
        x,
        y,
        arrays,
        isotherms=isotherms,
        max_plot_resolution=PLOT_RESOLUTION,
    )
    save_figure(fig, OUTPUT_DIR / "pyrolite_sp_properties.png", formats=("pdf",))
    print(f"Figure and arrays: {OUTPUT_DIR.resolve()}", flush=True)
