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
"""Calculate, refine and save equilibrium grids through one public workflow."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from .. import _core
from .phase_diagram import plot_phase_diagram, save_phase_diagram
from .phase_diagram_helpers import _figure_formats, save_figure
from .properties import (
    phase_diagram_grid_chunks,
    retry_property_grid_chunks,
    save_property_grid,
)
from .potential_grid import (
    DERIVATIVES,
    _natural_potential,
    load_potential_grid,
    save_potential_grid,
)
from .potential_infill import infill_potential_grid
from .potential_smoothing import smooth_potential_grid

_COORDINATES = dict(P="pressure", T="temperature", S="entropy", V="volume")


@dataclass
class PhaseDiagramCalculation:
    """Calculated diagram, original candidate models and paths to saved data.

    grid_manifest is None when gridding is disabled or the refined diagram is
    not completely resolved. raw_potential/infilled_potential/smoothed_potential
    are paths to the requested NPZ arrays, or None for stages that did not run.
    diagnostics explains why gridding was skipped, or records grid problems.
    Complete disk grids may still contain failed or excluded equilibrium nodes;
    inspect their equilibrium_success masks and diagnostics before processing.
    """

    diagram: object
    phases: list
    initial_diagram: Path
    refined_diagram: Path
    grid_manifest: Path | None
    diagnostics: list
    timings: dict
    raw_potential: Path | None = None
    infilled_potential: Path | None = None
    smoothed_potential: Path | None = None
    diagram_plot: Path | None = None
    diagram_plots: tuple[Path, ...] = ()


def _counts(values, name):
    counts = np.asarray(values)
    if (
        counts.shape != (2,)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 2).any()
    ):
        raise ValueError(f"{name} must contain two integer counts >= 2.")
    return tuple(int(n) for n in counts)


def _candidates(dataset, phase_names):
    """Select ordinary solution models and pure phases, collapsing aliases.

    Omit pure endmembers already contained in a selected solution. An explicit
    phase_names sequence overrides this automatic catalogue selection and its
    order is preserved. A calibrated ModelSet can also be supplied directly.
    """
    if isinstance(dataset, str):
        dataset = import_module(f"burnman_cpp.minerals.{dataset}")
    if hasattr(dataset, "phases"):
        if phase_names is not None:
            raise ValueError("phase_names cannot override a ModelSet.")
        return list(dataset.phases), dataset.to_dict()
    if phase_names is not None:
        names = list(phase_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("phase_names must be nonempty and unique.")
        return [getattr(dataset, name)() for name in names], dict(
            dataset=dataset.__name__, phase_names=names
        )
    selected = {}
    for name in sorted(vars(dataset)):
        factory = getattr(dataset, name)
        if name.startswith("_") or not callable(factory):
            continue
        # Dataset exports include constructors as well as mineral factories.
        if getattr(factory, "__module__", None) != dataset.__name__:
            continue
        material = factory()
        if isinstance(material, _core.RelaxedSolution):
            continue
        selected.setdefault(material.name, (name, material))
    endmembers = {
        name
        for _, phase in selected.values()
        if isinstance(phase, _core.Solution)
        for name in phase.endmember_names
    }
    selected = [
        (name, phase)
        for name, phase in selected.values()
        if isinstance(phase, _core.Solution) or phase.name not in endmembers
    ]
    if not selected:
        raise ValueError("The dataset contains no candidate mineral factories.")
    return [p for _, p in selected], dict(
        dataset=dataset.__name__, phase_names=[n for n, _ in selected]
    )


def _corner(corner, variables, composition, phases, settings, search):
    """Convert a two-coordinate reference equilibrium only when necessary."""
    values = {str(key).upper(): float(value) for key, value in corner.items()}
    if (
        len(values) != 2
        or not values.keys() <= _COORDINATES.keys()
        or not np.isfinite(list(values.values())).all()
    ):
        raise ValueError("Each corner needs exactly two finite P, T, S or V values.")
    if values.get("T", 0.0) < 0 or values.get("V", 1.0) <= 0:
        raise ValueError("Corner T must be nonnegative and V must be positive.")
    if not set(variables) <= values.keys():
        state = _core.stable_equilibrium(
            composition,
            phases,
            settings=settings,
            **search,
            **{_COORDINATES[key]: value for key, value in values.items()},
        )
        if not state.success or state.outside_model_domain:
            raise RuntimeError(f"Corner equilibrium failed: {state.message}")
        values.update({key: getattr(state, name) for key, name in _COORDINATES.items()})
    return values


def _write_json(path, data):
    """Replace metadata only after its serialization succeeds."""
    text = json.dumps(data, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _grid_to_disk(diagram, phases, resolution, directory, chunk_size, verbose):
    """Stream bounded solution batches, repair backwards, then retain the audit."""
    directory.mkdir()
    x, y = [
        np.linspace(*bounds, count)
        for bounds, count in zip(diagram.coordinate_ranges, resolution)
    ]
    manifest = dict(
        format="burnman_cpp.property_grid_chunks",
        schema_version=1,
        diagram_type=diagram.diagram_type,
        x=x.tolist(),
        y=y.tolist(),
        shape=[len(y), len(x)],
        chunks_file="chunks.jsonl",
        complete=False,
    )
    path = directory / "manifest.json"
    _write_json(path, manifest)
    completed = 0
    with (directory / "chunks.jsonl").open("w", encoding="utf-8") as journal:
        for index, (rows, cols, grid) in enumerate(
            phase_diagram_grid_chunks(
                diagram, phases, ["temperature"], x=x, y=y, chunk_size=chunk_size
            )
        ):
            filename = f"chunk_{index:06d}.json.gz"
            save_property_grid(grid, directory / filename)
            journal.write(
                json.dumps(
                    dict(
                        file=filename,
                        rows=[rows.start, rows.stop],
                        columns=[cols.start, cols.stop],
                    )
                )
                + "\n"
            )
            journal.flush()
            completed += grid.equilibrium_success.size
            if verbose:
                print(f"Saved grid equilibria: {completed}/{len(x)*len(y)}", flush=True)
    diagnostics, solved, solves = [], 0, 0
    if verbose:
        print(
            "Retrying failed grid nodes backwards and auditing final states", flush=True
        )
    for rows, cols, grid in retry_property_grid_chunks(diagram, phases, path):
        solved += int(grid.equilibrium_success.sum())
        solves += grid.equilibrium_solves
        if grid.diagnostics:
            diagnostics.append(
                dict(
                    rows=[rows.start, rows.stop],
                    columns=[cols.start, cols.stop],
                    messages=grid.diagnostics,
                )
            )
    manifest.update(
        complete=True,
        equilibrium_successes=solved,
        equilibrium_solves=solves,
        diagnostics=diagnostics,
    )
    _write_json(path, manifest)
    return path, diagnostics


def _process_potential(
    manifest, phases, symbol, directory, make_infilled, make_smoothed, sigma
):
    """Save each requested stage independently, preserving original arrays."""
    raw = load_potential_grid(manifest, phases, potential=symbol)
    paths = [save_potential_grid(raw, directory / "potential_raw.npz"), None, None]
    info = {}
    if make_infilled:
        filled, info = infill_potential_grid(
            raw, fill_derivatives=True, return_info=True
        )
        paths[1] = save_potential_grid(filled, directory / "potential_infilled.npz")
        # Detailed masks are recovered by comparing raw and infilled arrays.
        info = {key: value for key, value in info.items() if key != "estimated_masks"}
        if make_smoothed:
            if all(
                np.isfinite(getattr(filled, name)).all()
                for name in ("values", *DERIVATIVES)
            ):
                smoothed = smooth_potential_grid(filled, sigma=sigma)
                paths[2] = save_potential_grid(
                    smoothed, directory / "potential_smoothed.npz"
                )
            else:
                info["smoothing_skipped"] = (
                    "Infilling left nonfinite values or derivatives."
                )
    return paths, info


def calculate_phase_diagram_grid(
    dataset,
    composition,
    diagram,
    lower_corner,
    upper_corner,
    *,
    seeds=(11, 11),
    boundary_resolution=(201, 201),
    grid_resolution=(101, 101),
    output_dir="phase_diagram_output",
    phase_names=None,
    settings=None,
    pressure_range=None,
    temperature_range=None,
    chunk_size=4096,
    verbose=False,
    make_grid=False,
    make_potential_grid=False,
    make_infilled_potential_grid=False,
    make_smoothed_potential_grid=False,
    smoothing_sigma=2.0,
    plot=False,
    plot_kwargs=None,
    plot_formats="png",
):
    """Create and refine a diagram, then save equilibria for a resolved result.

    dataset is a mineral module/name (e.g. SLB24), or a calibrated ModelSet.
    composition is a Composition object or elemental mole-amount dictionary.
    Oxide masses can be supplied with Composition(..., unit_type="mass").
    phase_names optionally selects named factories in their intended order.
    Automatic catalogue selection uses ordinary solutions plus pure phases,
    removes aliases and pure endmembers represented by those solutions. Inspect
    the saved candidate list when selecting models for a particular rock.

    diagram selects two of P,T,S,V, e.g. "PS" or "V-T". Each corner supplies
    two SI conditions, e.g. {"P": 1e9, "T": 1000}. Convert corners through full
    equilibrium when their variables differ from the diagram. S (J/K), V (m³)
    and potentials are totals on the explicit bulk scale; P is Pa and T is K.
    Bounds are sorted after conversion. Optional pressure_range/temperature_range
    override plotted P/T limits, or guide inverse solves when unplotted.

    seeds, boundary_resolution and grid_resolution are (x,y) integer counts in
    the requested diagram order. Boundary counts limit reaction-line spacing;
    they do not add a rectangular grid of assemblage discovery seeds. The saved
    diagram/grid follow native PT,PS,PV,TS,TV,SV order, also recorded in metadata.

    Save initial.json and refined.json even on unresolved runs. Grid only when
    resolved, spacing_resolved and polygon checks all pass; otherwise return
    grid_manifest=None and diagnostics. This is the implemented resolution
    criterion, not proof that every possible disconnected field was discovered.
    A grid_solutions manifest journals compressed batches, retained equilibria
    and a reverse retry/final audit. No infilling or smoothing is automatic.
    Use load_potential_grid(run.grid_manifest, run.phases) for postprocessing.
    An existing calculation directory is rejected to preserve previous results.

    make_grid=True saves only the equilibrium samples. Set it to False to stop
    after diagram refinement. The three subsequent stage flags default to False,
    and each requires the preceding stage's flag to be True:
    make_potential_grid saves raw values and five relaxed derivatives to
    potential_raw.npz, retaining NaNs; make_infilled_potential_grid saves a
    separate potential_infilled.npz; make_smoothed_potential_grid saves
    potential_smoothed.npz. smoothing_sigma sets the width (default 2 cells),
    without enabling smoothing by itself.

    Independent variables select G(P,T), H(P,S), F(V,T) or E(V,S), in either
    coordinate order. PV/TS sections support equilibrium gridding but cannot
    make potential grids. Every stage preserves source arrays and original
    equilibrium masks. If infilling leaves nonfinite entries, smoothing is
    skipped and reported. All stages require a completely resolved diagram.

    plot=True saves phase_diagram.<format> after refinement, including for
    unresolved diagrams or runs without gridding. plot_formats accepts a string
    or list of strings (default "png"), e.g. ["png", "svg", "pdf"].
    diagram_plots lists all output paths; diagram_plot is the first one.
    plot_kwargs forwards styling and units
    to plot_phase_diagram(). The plot and default assemblage key stay inside
    output_dir. Plotting requires Matplotlib and performs no equilibrium solves.
    """
    variables = "".join(c for c in diagram.upper() if c not in "-_ ")
    native = next(
        (
            d
            for d in ("PT", "PS", "PV", "TS", "TV", "SV")
            if len(variables) == 2 and set(d) == set(variables)
        ),
        None,
    )
    if native is None:
        raise ValueError("diagram must select two different variables from P,T,S,V.")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be a boolean.")
    if plot:
        plot_formats = _figure_formats(plot_formats)
        if not plot_formats:
            raise ValueError("plot_formats must contain at least one format.")
    plot_options = dict(plot_kwargs or {})
    stages = [
        ("make_grid", make_grid),
        ("make_potential_grid", make_potential_grid),
        ("make_infilled_potential_grid", make_infilled_potential_grid),
        ("make_smoothed_potential_grid", make_smoothed_potential_grid),
    ]
    for index, (name, enabled) in enumerate(stages):
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean.")
        if enabled and index and not stages[index - 1][1]:
            raise ValueError(f"{name}=True requires {stages[index - 1][0]}=True.")
    potential = _natural_potential(native)
    if make_potential_grid and potential is None:
        raise ValueError(
            "Potential gridding requires PT, PS, VT or VS coordinates; "
            "use make_potential_grid=False for PV or TS diagrams."
        )
    if make_smoothed_potential_grid:
        widths = np.asarray(smoothing_sigma, float)
        if widths.ndim == 0:
            widths = np.repeat(widths, 2)
        if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
            raise ValueError(
                "smoothing_sigma must be positive and finite, scalar or (y,x)."
            )
    seeds = _counts(seeds, "seeds")
    boundary_resolution = _counts(boundary_resolution, "boundary_resolution")
    grid_resolution = _counts(grid_resolution, "grid_resolution")
    if make_smoothed_potential_grid and min(grid_resolution) < 3:
        raise ValueError("Smoothing requires at least three grid points per axis.")
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")
    bulk = dict(getattr(composition, "atomic_composition", composition))
    if not bulk or not np.isfinite(list(bulk.values())).all() or min(bulk.values()) < 0:
        raise ValueError("composition needs finite nonnegative elemental mole amounts.")
    if not any(bulk.values()):
        raise ValueError("composition must contain a positive amount.")
    search = {}
    for name, bounds in (
        ("pressure_range", pressure_range),
        ("temperature_range", temperature_range),
    ):
        if bounds is not None:
            bounds = np.asarray(bounds, float)
            if (
                bounds.shape != (2,)
                or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must contain two finite increasing limits.")
            search[name] = bounds.tolist()
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Calculation directory is not empty: {directory}")
    phases, provenance = _candidates(dataset, phase_names)
    opts = (
        _core.PhaseDiagramSettings.from_dict(settings.to_dict())
        if settings
        else _core.PhaseDiagramSettings()
    )
    opts.verbose = bool(verbose)
    for variable, count in zip(variables, seeds):
        setattr(opts, _COORDINATES[variable] + "_seeds", count)
    timings = {}
    started = perf_counter()
    corners = [
        _corner(c, variables, bulk, phases, opts, search)
        for c in (lower_corner, upper_corner)
    ]
    bounds = search | {
        _COORDINATES[v] + "_range": sorted(c[v] for c in corners) for v in variables
    }
    bounds.update({k: v for k, v in search.items() if k[0].upper() in variables})
    if any(pair[0] >= pair[1] for pair in bounds.values()):
        raise ValueError("Converted corners must span both diagram coordinates.")
    timings["corners"] = perf_counter() - started
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        workflow=dict(
            models=provenance,
            corners=corners,
            requested_diagram=variables,
            native_diagram=native,
            boundary_resolution=list(boundary_resolution),
            grid_resolution=list(grid_resolution),
        )
    )
    if verbose:
        print(
            f"Calculating {native} phase diagram with {len(phases)} candidates",
            flush=True,
        )
    started = perf_counter()
    initial = _core.phase_diagram(bulk, phases, settings=opts, diagram=native, **bounds)
    timings["diagram"] = perf_counter() - started
    initial_path = save_phase_diagram(
        initial, directory / "initial.json", metadata=metadata
    )
    started = perf_counter()
    resolution = (
        boundary_resolution if native == variables else boundary_resolution[::-1]
    )
    refined = _core.refine_phase_diagram(
        bulk, phases, initial, opts, resolution=resolution
    )
    timings["refinement"] = perf_counter() - started
    refined_path = save_phase_diagram(
        refined, directory / "refined.json", metadata=metadata
    )
    diagram_plot = None
    diagram_plots = ()
    if plot:
        started = perf_counter()
        options = {
            "label_key_path": directory / "phase_diagram_labels.md"
        } | plot_options
        figure, _ = plot_phase_diagram(refined, **options)
        diagram_plot = directory / f"phase_diagram.{plot_formats[0]}"
        diagram_plots = save_figure(
            figure, diagram_plot, formats=plot_formats[1:], bbox_inches="tight"
        )
        timings["plot"] = perf_counter() - started
    polygons = _core.phase_diagram_field_polygons(refined)
    diagnostics = list(
        dict.fromkeys(
            [*refined.diagnostics, *refined.spacing_diagnostics, *polygons.diagnostics]
        )
    )
    grid_path = None
    potential_paths, processing_info = [None, None, None], {}
    ready = refined.resolved and refined.spacing_resolved and not diagnostics
    if make_grid and ready:
        started = perf_counter()
        resolution = grid_resolution if native == variables else grid_resolution[::-1]
        grid_path, diagnostics = _grid_to_disk(
            refined,
            phases,
            resolution,
            directory / "grid_solutions",
            chunk_size,
            verbose,
        )
        timings["grid"] = perf_counter() - started
        if make_potential_grid:
            started = perf_counter()
            potential_paths, processing_info = _process_potential(
                grid_path,
                phases,
                potential,
                directory,
                make_infilled_potential_grid,
                make_smoothed_potential_grid,
                smoothing_sigma,
            )
            timings["postprocessing"] = perf_counter() - started
            if "smoothing_skipped" in processing_info:
                diagnostics.append(processing_info["smoothing_skipped"])
    elif not ready:
        diagnostics = diagnostics or [
            "Diagram is not completely resolved; gridding skipped."
        ]
        if verbose:
            print("Diagram is not completely resolved; gridding skipped.", flush=True)
    _write_json(
        directory / "report.json",
        dict(
            timings_seconds=timings,
            gridding_started=grid_path is not None,
            diagnostics=diagnostics,
            postprocessing=processing_info,
        ),
    )
    return PhaseDiagramCalculation(
        refined,
        phases,
        initial_path,
        refined_path,
        grid_path,
        diagnostics,
        timings,
        *potential_paths,
        diagram_plot,
        diagram_plots,
    )
