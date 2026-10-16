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

"""Refine all phase lines in a saved diagram, then save JSON and a figure.

Use the original example's candidate_phases() to restore its models, e.g.:
python examples/example_refine_phase_diagram.py metasediment_phase_diagram.json \
    --models example_metasediment_phase_diagram --resolution 101 201

Resolution counts axis points: here 100 divisions of x and 200 divisions of y.
All new equilibrium points are calculated and validated in C++.
Resolved diagrams retain their connections and labels, adding only solved points.
Unresolved diagrams also run recovery. Spacing failures are reported separately
and do not invalidate a resolved diagram.
Add --verbose to report line refinement and field-checking progress.
"""

import argparse
from functools import partial
from pathlib import Path

import burnman_cpp as bm
from burnman_cpp.tools.phase_diagram_helpers import (
    load_example,
    phase_diagram_timer,
    plot_phase_diagram_example,
    refine_saved,
    write_phase_diagram,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--models",
        required=True,
        help="Original module or .py file supplying candidate_phases().",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        required=True,
        metavar=("NX", "NY"),
        help="Axis point counts (at least 2); 101 201 gives 100 x and 200 y divisions.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("refined_phase_diagram_output")
    )
    for name, default, choices in (
        ("pressure", "kbar", ("Pa", "GPa", "kbar")),
        ("temperature", "C", ("C", "K")),
        ("entropy", "J/K", ("J/K", "kJ/K", "kB/atom")),
        ("volume", "m3", ("m3", "cm3", "kg/m3")),
    ):
        parser.add_argument(f"--{name}-unit", default=default, choices=choices)
    parser.add_argument("--swap-axes", action="store_true")
    parser.add_argument("--label-fontsize", type=float, default=7.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Print refinement progress."
    )
    args = parser.parse_args()
    with phase_diagram_timer(args.verbose):
        saved = bm.load_phase_diagram(args.source)
        candidates = load_example(args.models).candidate_phases()
        settings = bm.PhaseDiagramSettings.from_dict(saved.get("settings", {}))
        settings.verbose = args.verbose
        if args.verbose:
            print(f"Loaded {args.source}. Starting refinement.", flush=True)
        display = {
            key: getattr(args, key)
            for key in (
                "pressure_unit",
                "temperature_unit",
                "entropy_unit",
                "volume_unit",
                "swap_axes",
            )
        }
        result = refine_saved(
            saved,
            candidates,
            settings=settings,
            verbose=args.verbose,
            resolution=args.resolution,
            **display,
        )
        write_phase_diagram(
            result,
            args.output_dir / "phase_diagram",
            metadata=saved,
            label_fontsize=args.label_fontsize,
            plot=(
                None
                if args.no_plots
                else partial(
                    plot_phase_diagram_example,
                    **display,
                    key_path=args.output_dir / "assemblages.md",
                    bbox_inches=None,
                )
            ),
        )


if __name__ == "__main__":
    main()
