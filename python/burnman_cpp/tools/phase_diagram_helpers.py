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

"""Reusable runners and output helpers for phase diagram calculations.

Examples supply their bulk, models, settings and scientific annotations. These
helpers handle the repeated command-line and output steps. Keeping this module
with the package also lets examples be loaded by file, without changing the
caller's Python import path permanently.
"""

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import burnman_cpp as bm
from .phase_diagram import _value

__all__ = [
    "example_parser",
    "load_example",
    "plot_phase_diagram_example",
    "refine_saved",
    "report_phase_diagram",
    "run_phase_diagram_example",
    "save_figure",
    "write_phase_diagram",
]


def load_example(name):
    """Load a model/example module by import name or Python filename."""
    path = Path(name)
    if path.suffix != ".py":
        return importlib.import_module(name)
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def example_parser(
    description,
    output_dir,
    *,
    seeds=None,
    label_fontsize=7.0,
    replot=False,
    refine=False,
    resolution=False,
):
    """Build the common CLI, leaving scientific options to the example."""
    parser = argparse.ArgumentParser(description=description)
    if seeds is not None:
        parser.add_argument("--seeds", type=int, default=seeds)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=label_fontsize)
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    if replot:
        parser.add_argument(
            "--plot-json", type=Path, help="Replot saved results without solving."
        )
    if refine:
        parser.add_argument(
            "--refine-json", type=Path, help="Resume saved equilibrium lines."
        )
    if resolution:
        parser.add_argument(
            "--resolution",
            type=int,
            nargs=2,
            metavar=("NX", "NY"),
            help="With --refine-json, refine lines to these axis point counts.",
        )
    return parser


def report_phase_diagram(result, output_dir):
    """Print the same field, line, junction and solve counts for every example."""
    print(
        f"{len(_value(result, 'fields', []))} fields, "
        f"{len(_value(result, 'boundaries', []))} edges, "
        f"{sum(_value(n, 'kind') == 'junction' for n in _value(result, 'nodes', []))} junctions; "
        f"{_value(result, 'equilibrium_solves', 0)} native equilibrium solves.",
        flush=True,
    )
    print(f"Resolved: {_value(result, 'resolved')}. Outputs: {output_dir}", flush=True)
    for message in _value(result, "diagnostics", []):
        print(message, flush=True)
    spacing = _value(result, "spacing_diagnostics", [])
    if spacing:
        print("Requested line spacing was not reached:", flush=True)
        for message in spacing:
            print(message, flush=True)


def save_figure(figure, path, *, formats=("svg",), dpi=200, **options):
    """Save a figure in its requested formats and release its plotting memory."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (path.suffix,) + tuple("." + item for item in formats):
        figure.savefig(path.with_suffix(suffix), dpi=dpi, **options)
    plt.close(figure)


def plot_phase_diagram_example(
    result,
    path,
    label_fontsize=7.0,
    *,
    axes=None,
    scope=None,
    notes="",
    figsize=(12, 10),
    formats=("svg",),
    key_path=None,
    bbox_inches="tight",
    **options,
):
    """Plot a section, with an optional geological caption and field-status line."""
    import matplotlib.pyplot as plt

    path = Path(path)
    key_path = key_path or path.with_name(path.stem + "_assemblages.md")
    ax = None
    if scope:
        figure, ax = plt.subplots(figsize=figsize)
        figure.subplots_adjust(bottom=0.20)
    figure, ax = bm.plot_phase_diagram(
        result,
        ax=ax,
        label_fontsize=label_fontsize,
        label_key_path=key_path,
        **options,
    )
    ax.set(**(axes or {}))
    if scope:
        if ax.get_legend() is not None:
            ax.get_legend().set_bbox_to_anchor((0.0, -0.11))
        ax.grid(alpha=0.12)
        fields = [
            p for p in ax.phase_diagram_geometry.polygons if not p.outside_model_domain
        ]
        status = (
            f"{sum(p.has_open_boundary for p in fields)} open fields; "
            f"{sum(p.n_phases <= 0 or not p.phases for p in fields)} unidentified fields; "
            f"{len(_value(result, 'diagnostics', []))} numerical diagnostics in JSON"
        )
        figure.text(
            0.1,
            0.015,
            f"{scope}\n{status}; finite seeds may miss narrow fields.\n{notes}{key_path.name}",
            fontsize=8,
        )
    save_figure(
        figure, path, formats=formats, bbox_inches=None if scope else bbox_inches
    )


def refine_saved(
    source,
    phases,
    *,
    composition=None,
    verbose=False,
    resolution=None,
    settings=None,
    **display,
):
    """Continue saved equilibria, preserving their bulk scale and solver settings."""
    data = source if isinstance(source, dict) else bm.load_phase_diagram(source)
    if settings is None:
        settings = bm.PhaseDiagramSettings.from_dict(data.get("settings", {}))
    settings.verbose = verbose
    if resolution is not None:
        display["resolution"] = resolution
    return bm.refine_phase_diagram(
        composition if composition is not None else data["composition_start"],
        phases,
        data,
        settings,
        **display,
    )


def write_phase_diagram(
    result, stem, *, metadata=None, plot=None, label_fontsize=7.0, save_json=None
):
    """Write one diagram's JSON and optional figures, then report its diagnostics."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if save_json is None:
        bm.save_phase_diagram(result, stem.with_suffix(".json"), metadata=metadata)
    else:
        save_json(result, stem.with_suffix(".json"))
    if plot is not None:
        plot(result, stem.with_suffix(".png"), label_fontsize)
    report_phase_diagram(result, stem.parent)
    return result


def run_phase_diagram_example(
    parser, calculate, *, stem, save_json=None, plot=None, refine=None, args=None
):
    """Master CLI: calculate/refine/replot, write outputs, and report diagnostics.

    calculate receives the parsed example arguments. The optional refine and plot
    functions keep their familiar source/verbose and result/path/fontsize arguments,
    so examples can also be used directly from Python or by the patching script.
    Pass args when a specialised example has already parsed additional options.
    """
    if args is None:
        args = parser.parse_args()
    resolution = getattr(args, "resolution", None)
    if resolution is not None:
        if not args.refine_json:
            parser.error("--resolution requires --refine-json.")
        if min(resolution) < 2:
            parser.error("--resolution counts must be at least 2.")
    if getattr(args, "plot_json", None):
        result = bm.load_phase_diagram(args.plot_json)
        save_json = None  # Retain the saved metadata when only replotting.
    elif getattr(args, "refine_json", None):
        options = dict(verbose=args.verbose)
        if resolution is not None:
            options["resolution"] = resolution
        result = refine(args.refine_json, **options)
    else:
        result = calculate(args)
    return write_phase_diagram(
        result,
        args.output_dir / stem,
        save_json=save_json,
        plot=None if args.no_plots else plot,
        label_fontsize=args.label_fontsize,
    )
